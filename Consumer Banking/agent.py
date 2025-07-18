import asyncio
import random
import uuid
from pydantic import BaseModel
from typing import Optional, List

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

# --- 1. CONTEXT MODEL ---
class BankAgentContext(BaseModel):
    case_number: Optional[str] = None
    topics_discussed: List[str] = []
    feedback: Optional[str] = None

# --- 2. TOOLS ---
@function_tool(
    name_override="faq_lookup_tool",
    description_override="Lookup personal banking FAQs by semantic similarity."
)
async def faq_lookup_tool(question: str) -> str:
    """
    Return the best matching FAQ answer for the user's question.
    """
    # In a production system, replace this stubbed logic with real embeddings search
    faqs = {
        "hours": "Our branches are open Mon-Fri 9am-5pm EST.",
        "online banking": "You can enroll in online banking at examplebank.com.",
        "fees": "We have no monthly maintenance fees with a minimum balance of $500."  # etc.
    }
    for key, answer in faqs.items():
        if key in question.lower():
            return answer
    return "I'm sorry, I don't have an answer to that FAQ."

@function_tool(
    description_override="Create a support case for the user based on their query and urgency."
)
async def create_case_tool(
    context: RunContextWrapper[BankAgentContext],
    query: str,
    urgency: str
) -> str:
    """
    Create a support case and record it in the context.
    """
    # Prompt for ZIP code in CLI
    zip_code = input("Please enter your 5-digit ZIP code: ").strip()
    # Generate a fake case number
    case_number = f"CS-{random.randint(1000000, 9999999)}"
    context.context.case_number = case_number
    context.context.topics_discussed.append(f"Case {case_number}: {query}")
    resolution = "within 24 hours" if urgency == "high" else "within 3 business days"
    return (
        f"Thank you. A case ({case_number}) has been created for '{query}'. "
        f"A specialist in {zip_code} will contact you {resolution}."
    )

# --- 3. AGENT DEFINITIONS ---
# FAQ Specialist
faq_agent = Agent[BankAgentContext](
    name="FAQ Agent",
    handoff_description="Handles general banking questions via FAQ lookup.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are a banking FAQ specialist. "
        "Use the `faq_lookup_tool` to answer the user's question."
    ),
    tools=[faq_lookup_tool],
)

# Escalation Specialist
escalation_agent = Agent[BankAgentContext](
    name="Escalation Agent",
    handoff_description="Handles customer issues by creating a support case.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are a banking support escalation specialist. "
        "Use the `create_case_tool` to create a support case when needed."
    ),
    tools=[create_case_tool],
)

# Triage Agent
triage_agent = Agent[BankAgentContext](
    name="Triage Agent",
    handoff_description="Routes user queries to the correct specialist agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are a triage agent. "
        "If the user asks a general question, hand off to FAQ Agent. "
        "If the user expresses an issue (e.g., 'fraud', 'blocked', 'error'), hand off to Escalation Agent."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=escalation_agent),
    ],
)

# Allow specialists to hand back to triage for unrelated queries
faq_agent.handoffs.append(triage_agent)
escalation_agent.handoffs.append(triage_agent)

# --- 4. CONVERSATION LOOP ---
async def main():
    print("Booting banking assistant...")
    context = BankAgentContext()
    current_agent: Agent[BankAgentContext] = triage_agent
    conversation_id = uuid.uuid4().hex
    history: list[TResponseInputItem] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})
        with trace("Banking Service", group_id=conversation_id):
            result = await Runner.run(current_agent, history, context=context)

        # Display outputs
        for item in result.new_items:
            name = item.agent.name
            if isinstance(item, MessageOutputItem):
                print(f"{name}: {ItemHelpers.text_message_output(item)}")
            elif isinstance(item, HandoffOutputItem):
                print(f"Routing from {item.source_agent.name} to {item.target_agent.name}...")
            elif isinstance(item, ToolCallItem):
                print(f"{name}: Invoking tool {item.tool_id}...")
            elif isinstance(item, ToolCallOutputItem):
                print(f"{name}: {item.output}")

        # Prepare for next turn
        history = result.to_input_list()
        current_agent = result.last_agent

if __name__ == "__main__":
    asyncio.run(main())
