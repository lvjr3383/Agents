# ==============================================================
# agent.py – Eva Banking Assistant (Ultra-Final + Fuzzy FAQ Patch)
# ==============================================================

import os
import asyncio
import random
import uuid
import json
import difflib
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it.")

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
FAQ_FILE = DATA_DIR / "Personal Banking FAQ.txt"
OFFERS_FILE = DATA_DIR / "Consumer Offers.txt"
MODEL = "gpt-4o-mini"

# ---------------- CONTEXT ----------------
class BankAgentContext(BaseModel):
    case_number: Optional[str] = None
    zip_code: Optional[str] = None
    topics_discussed: List[str] = []
    last_resolution: Optional[str] = None        # 'faq' | 'case' | 'offers'
    offers_nudged: bool = False
    offers_presented: bool = False
    feedback_score: Optional[int] = None
    session_open: bool = True
    greeted: bool = False

# ---------------- DATA LOADERS ----------------
def _load_faq_blocks() -> List[Tuple[str, str]]:
    """Each block: Q (first line) + answer lines until blank line."""
    if not FAQ_FILE.exists():
        return []
    text = FAQ_FILE.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    pairs: List[Tuple[str, str]] = []
    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        q = lines[0]
        a = " ".join(lines[1:]) if len(lines) > 1 else ""
        pairs.append((q, a))
    return pairs

_FAQ_PAIRS = _load_faq_blocks()

def _load_offers() -> List[Dict[str, str]]:
    """Parse numbered offers file to list."""
    if not OFFERS_FILE.exists():
        return []
    txt = OFFERS_FILE.read_text(encoding="utf-8")
    lines = txt.splitlines()
    offers: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    buf: List[str] = []
    for line in lines + ["END"]:
        numbered_prefixes = tuple(f"{i}." for i in range(1, 50))
        if line.strip().startswith(numbered_prefixes) or line == "END":
            if current:
                current["raw"] = " ".join(s.strip() for s in buf if s.strip())
                offers.append(current)
                current, buf = {}, []
            if line != "END":
                title = line.split(".", 1)[1].strip() if "." in line else line.strip()
                current = {"product": title}
        else:
            buf.append(line)
    return offers

_OFFERS = _load_offers()

# ---------------- TOOLS ----------------
@function_tool(
    name_override="faq_lookup_tool",
    description_override="Return the raw approved FAQ answer (no model additions)."
)
async def faq_lookup_tool(question: str) -> str:
    """Keyword overlap + fuzzy fallback to minimize NO_MATCH."""
    if not _FAQ_PAIRS:
        return "FAQ_KB_EMPTY"
    ql = question.lower()

    # Keyword overlap
    scores: List[Tuple[int, int]] = []
    for idx, (q, _a) in enumerate(_FAQ_PAIRS):
        tokens = {t for t in q.lower().split() if len(t) > 2}
        overlap = sum(1 for t in tokens if t in ql)
        if overlap:
            scores.append((overlap, idx))
    if scores:
        scores.sort(reverse=True)
        return _FAQ_PAIRS[scores[0][1]][1]

    # Fuzzy match (question lines)
    questions = [q for q, _a in _FAQ_PAIRS]
    best = difflib.get_close_matches(question, questions, n=1, cutoff=0.3)
    if best:
        idx = questions.index(best[0])
        return _FAQ_PAIRS[idx][1]

    return "FAQ_NO_MATCH"

@function_tool(
    name_override="create_case_tool",
    description_override="Collect ZIP, generate case number, record issue + urgency."
)
async def create_case_tool(
    context: RunContextWrapper[BankAgentContext],
    issue_summary: str,
    urgency: str
) -> str:
    while True:
        zip_code = input("Please enter your 5-digit ZIP code: ").strip()
        if zip_code.isdigit() and len(zip_code) == 5:
            context.context.zip_code = zip_code
            break
        print("Invalid ZIP (need 5 digits).")
    case_number = f"CS-{random.randint(1000000, 9999999)}"
    context.context.case_number = case_number
    context.context.topics_discussed.append(f"Case {case_number}: {issue_summary}")
    context.context.last_resolution = "case"
    window = "within 24 hours" if urgency.lower() == "high" else "within 3 business days"
    return f"CASE_CREATED|{case_number}|ZIP:{zip_code}|WINDOW:{window}|ISSUE:{issue_summary}"

@function_tool(
    name_override="generate_offers_tool",
    description_override="Return up to 3 concise savings/CD offers as JSON."
)
async def generate_offers_tool(preference_hint: str) -> str:
    if not _OFFERS:
        return json.dumps({"offers": []})
    prefs = {t for t in preference_hint.lower().split() if len(t) > 2}
    ranked: List[Tuple[int, Dict[str, str]]] = []
    for offer in _OFFERS:
        raw = " ".join(offer.values()).lower()
        score = sum(1 for t in prefs if t in raw)
        ranked.append((score, offer))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = [o for _, o in ranked[:3]]
    return json.dumps({"offers": top})

@function_tool(
    name_override="mark_offers_nudged_tool",
    description_override="Mark that the offers nudge has been shown."
)
async def mark_offers_nudged_tool(context: RunContextWrapper[BankAgentContext]) -> str:
    context.context.offers_nudged = True
    return "OFFERS_NUDGED"

# ---------------- AGENTS ----------------
BOUNDARIES = (
    "You cannot view personal balances, move money, or give legal/financial advice. "
    "If the user requests restricted actions, state that limitation and suggest allowed alternatives."
)

welcome_agent = Agent[BankAgentContext](
    name="Eva Welcome",
    handoff_description=(
        "If last_resolution in ('faq','case') AND offers_nudged is False: hand off to Eva Offers Nudge immediately, "
        "without replying, even if the user says thanks or indicates closure. "
        "Controller. First turn: greet + capabilities + limits. "
        "Informational question -> Eva FAQ. Problem keywords (fraud, unauthorized, blocked, lost, dispute, suspicious, locked) -> Eva Escalation. "
        "Direct interest in offers/rates -> Eva Offers. "
        "If offers_presented is True and user shows closure/thanks (thanks, no more questions, bye): hand off to Eva Closing."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Eva Welcome (controller). "
        "Introduce yourself ONCE at session start. "
        "ABSOLUTE RULE: Immediately after any FAQ or case resolution, if offers_nudged is False, hand off to Eva Offers Nudge WITHOUT replying, "
        "even if the user expresses closure or gratitude. Do NOT skip this. "
        "If offers_presented is True and the user expresses closure/thanks, hand off directly to Eva Closing. "
        "Do NOT restate case details or offers yourself—those belong to Escalation/Offers. "
        "You do NOT answer FAQs or offers. When you must speak (e.g., greeting), keep it ≤1 short line. "
        f"{BOUNDARIES}"
    ),
    handoffs=[],
    model=MODEL,
)

faq_agent = Agent[BankAgentContext](
    name="Eva FAQ",
    handoff_description=(
        "Answers informational questions from FAQ. After answering or clarification, hand off immediately to Eva Welcome (no pleasantries)."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You MUST call `faq_lookup_tool` exactly once for every user question. "
        "Do not answer or hand off without calling it. Then paraphrase ONLY the tool answer in 3–4 short lines, no bullets. "
        "If FAQ_NO_MATCH or FAQ_KB_EMPTY: respond with one short clarification request (≤2 lines) and hand off. "
        "Do NOT add 'Anything else?'; just hand off to Eva Welcome."
    ),
    tools=[faq_lookup_tool],
    model=MODEL,
)

escalation_agent = Agent[BankAgentContext](
    name="Eva Escalation",
    handoff_description=(
        "Creates support cases for problems. After creating the case, hand off immediately to Eva Welcome."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Eva Escalation. "
        "Call `create_case_tool` exactly once. Confirm in ≤4 short lines (case # and timeframe). "
        "No extra chit-chat; immediately hand off to Eva Welcome."
    ),
    tools=[create_case_tool],
    model=MODEL,
)

offers_nudge_agent = Agent[BankAgentContext](
    name="Eva Offers Nudge",
    handoff_description=(
        "Ask once per session if user wants offers. Yes -> Eva Offers. No/closure -> Eva Closing. New question -> Eva Welcome."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Eva Offers Nudge. "
        "First call `mark_offers_nudged_tool` to set the flag. "
        "Say: 'Before we wrap, would you like a quick look at current savings or CD offers?' (≤2 lines). "
        "Affirmative -> hand off to Eva Offers. Decline or closure -> Eva Closing. New unrelated question -> Eva Welcome. "
        "Do not present offers yourself."
    ),
    tools=[mark_offers_nudged_tool],
    model=MODEL,
)

offers_agent = Agent[BankAgentContext](
    name="Eva Offers",
    handoff_description=(
        "Presents up to 3 concise savings/CD offers. Then hand off to Eva Welcome."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Eva Offers. "
        "Call `generate_offers_tool` once. "
        "Return exactly ONE plain paragraph (≤4 sentences, no bullets, no numbering, no markdown) naming up to 3 products with one key benefit each. "
        "No follow-up question. Immediately hand off to Eva Welcome."
    ),
    tools=[generate_offers_tool],
    model=MODEL,
)

closing_agent = Agent[BankAgentContext](
    name="Eva Closing",
    handoff_description=(
        "Recap + ask for 1–5 rating. On digit -> thank & end. New question -> Eva Welcome."
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Eva Closing. "
        "Start with 'Recap:' and ONE sentence summarizing what was done (include case # if any, and whether offers were shown). "
        "Then ask: 'On a scale of 1–5, how did I do today?' "
        "If user gives 1–5, thank briefly and end the session. "
        "If user just says thanks/bye with no number, politely restate rating request once; if still no digit and no new question, thank and end."
    ),
    model=MODEL,
)

# Handoffs wiring
welcome_agent.handoffs.extend([
    faq_agent,
    handoff(agent=escalation_agent),
    handoff(agent=offers_nudge_agent),
    handoff(agent=offers_agent),
    handoff(agent=closing_agent),
])
for ag in (faq_agent, escalation_agent, offers_nudge_agent, offers_agent, closing_agent):
    ag.handoffs.append(welcome_agent)

# ---------------- MAIN LOOP ----------------
async def main():
    print("Booting Eva (fully agentic assistant)...")
    context = BankAgentContext()
    current_agent: Agent[BankAgentContext] = welcome_agent
    conversation_id = uuid.uuid4().hex
    history: List[TResponseInputItem] = []

    # Auto-greet
    if not context.greeted:
        print("Eva Welcome: Hi, I'm Eva, your banking assistant. How can I help you today?")
        context.greeted = True

    while context.session_open:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            with trace("Eva Banking Service", group_id=conversation_id):
                result = await Runner.run(current_agent, history, context=context)
        except Exception as e:
            print(f"System: Error during run: {e}")
            continue

        for item in result.new_items:
            agent_name = item.agent.name

            if isinstance(item, MessageOutputItem):
                msg = ItemHelpers.text_message_output(item)

                # Handle rare case where assistant echoes digit in same turn
                if agent_name == "Eva Closing":
                    stripped = msg.strip()
                    if stripped.isdigit():
                        val = int(stripped)
                        if 1 <= val <= 5:
                            context.feedback_score = val
                            context.session_open = False

                print(f"{agent_name}: {msg}")

            elif isinstance(item, HandoffOutputItem):
                print(f"Routing from {item.source_agent.name} to {item.target_agent.name}...")

            elif isinstance(item, ToolCallItem):
                label = getattr(item, "tool_name", None) or getattr(item, "tool_id", None) or "(tool)"
                print(f"{agent_name}: Invoking tool {label}...")

            elif isinstance(item, ToolCallOutputItem):
                # Show case creation output; suppress raw JSON for offers
                if agent_name == "Eva Escalation":
                    print(f"{agent_name}: {item.output}")

                if isinstance(item.output, str):
                    out = item.output
                    if agent_name == "Eva Escalation" and out.startswith("CASE_CREATED|"):
                        context.last_resolution = "case"
                    elif agent_name == "Eva Offers" and "\"offers\"" in out:
                        context.last_resolution = "offers"
                        context.offers_presented = True
                    elif agent_name == "Eva Offers Nudge" and out == "OFFERS_NUDGED":
                        context.offers_nudged = True
                    elif agent_name == "Eva FAQ" and not out.startswith("FAQ_NO_MATCH"):
                        context.last_resolution = "faq"

        history = result.to_input_list()
        current_agent = result.last_agent

        # User gives rating as user message (after Closing asks)
        if (
            current_agent.name == "Eva Closing"
            and history
            and history[-1]["role"] == "user"
            and history[-1]["content"].strip().isdigit()
        ):
            val = int(history[-1]["content"].strip())
            if 1 <= val <= 5:
                context.feedback_score = val
                context.session_open = False

    print("Session ended.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
