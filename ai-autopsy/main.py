import json
import os
import re
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI(title="AI Autopsy", version="1.0.0")

# ── Load taxonomy at startup ──────────────────────────────────────────────────
_taxonomy_path = Path(__file__).parent / "data" / "taxonomy.json"
with _taxonomy_path.open() as f:
    TAXONOMY: dict = json.load(f)

_TAXONOMY_STRING = "\n".join(
    f"{m['id']}: {m['name']} - {m['definition']}"
    for m in TAXONOMY["failure_modes"]
)

_FM_MAP: dict[str, dict] = {m["id"]: m for m in TAXONOMY["failure_modes"]}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = f"""You are AI Autopsy — a brutally honest diagnostic tool for failed AI builds. You do not guess. You do not label without evidence. You prove failure modes by citing specific language or logic gaps directly from the input.

You have access to a taxonomy of 17 failure modes across 5 categories plus FM-00 for insufficient data. Your job is to:
1. Read the input carefully
2. Match it against the taxonomy
3. Identify ONE primary failure mode
4. Identify UP TO TWO secondary failure modes
5. For each match, extract a direct quote or specific logic gap from the input as evidence
6. If multiple modes are present, articulate the causal chain — how the primary failure created conditions for the secondary failure(s)
7. Reconstruct what the spec should have said
8. Provide a prioritized remediation plan

CRITICAL RULES:
- Never identify a failure mode without evidence from the input
- If the input is too brief or lacks technical or contextual detail, you MUST default to FM-00. Choosing FM-00 when the input is vague noise is considered a high-accuracy result, not a failure.
- The causal chain must show dependency, not just co-occurrence. Use the pattern: Because [Primary Failure] occurred, it created conditions for [Secondary Failure].
- Verdict must be exactly one of: Inconclusive / Recoverable / Critical
- Evidence quotes must be wrapped in backticks

TAXONOMY:
{_TAXONOMY_STRING}

OUTPUT FORMAT:
Return valid Markdown only. No preamble. No explanation outside the report template.
Use this exact structure:

## AI Autopsy Report

### Input Type
[input type]

### Primary Failure Mode
[FM-ID] [Mode Name] — [one-line definition]

### Evidence
`[direct quote or specific logic gap from input]`

### Secondary Failure Mode(s)
[FM-ID] [Mode Name] — [one-line definition]
Evidence: `[quote or logic gap]`

### Causal Chain
Because [primary failure], it created conditions for [secondary failure(s)].

### Reconstructed Spec
[What the spec should have said in 3-5 bullet points]

### Remediation Plan
1. [First and most important fix]
2. [Second fix]
3. [Third fix]

### Verdict
[Inconclusive / Recoverable / Critical]"""

_TYPE_DESCRIPTIONS = {
    "prompt-output": "a failed prompt and AI output pair",
    "spec": "a vague or failed spec, PRD, or requirements description",
    "code": "a code snippet with a description of what went wrong",
}


# ── Request model ─────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    input_type: str
    user_input: str

    @field_validator("user_input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Input cannot be empty")
        if len(v) > 5000:
            raise ValueError("Input exceeds 5000 character limit")
        return v

    @field_validator("input_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in _TYPE_DESCRIPTIONS:
            raise ValueError(f"Invalid input type: {v}")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────
def _append_remediations(report: str) -> str:
    """Append local remediation references for FM-IDs found in the report."""
    ids = list(dict.fromkeys(re.findall(r"\bFM-\d{2}\b", report)))
    appendix = []
    for fm_id in ids:
        if fm_id == "FM-00":
            continue
        mode = _FM_MAP.get(fm_id)
        if not mode:
            continue
        if mode["remediation"][:30] not in report:
            appendix.append(
                f"\n---\n**{fm_id} Remediation Reference:** {mode['remediation']}"
            )
    return report + "".join(appendix)


def _is_inconclusive(report: str) -> bool:
    return (
        bool(re.search(r"Verdict[\s\S]{0,80}Inconclusive", report, re.IGNORECASE))
        or "FM-00" in report
    )


def _extract_verdict(report: str) -> Optional[str]:
    match = re.search(r"###\s*Verdict\s*\n+([^\n\r]+)", report, re.IGNORECASE)
    return match.group(1).strip() if match else None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured. Add it to your .env file.",
        )

    user_prompt = (
        f"Input Type: {_TYPE_DESCRIPTIONS[req.input_type]}\n\n"
        f"Input:\n{req.user_input}\n\nRun the autopsy."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key. Check your .env file.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached. Please wait a moment and try again.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e}")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise HTTPException(status_code=500, detail="No text content returned by model.")

    report = _append_remediations(text)

    return {
        "report": report,
        "verdict": _extract_verdict(report),
        "is_inconclusive": _is_inconclusive(report),
    }


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
