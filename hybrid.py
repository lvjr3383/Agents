import os
import openai
import numpy as np
import json
import random
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List

# --- 1. STRUCTURED MEMORY (CONTEXT) ---
class BankAgentContext(BaseModel):
    case_number: Optional[str] = None
    topics_discussed: List[str] = []
    feedback: Optional[str] = None

# --- 2. SETUP ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key: raise ValueError("OPENAI_API_KEY not found")
openai.api_key = api_key
client = openai.OpenAI()
faq_embeddings = []

# --- 3. HELPER FUNCTIONS & TOOLS ---
def process_faq_file(filename="Data/Personal Banking FAQ.txt"):
    with open(filename, 'r', encoding='utf-8') as f: content = f.read()
    qa_pairs = []
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines or not lines[0]: continue
        question, answer = lines[0].strip(), ' '.join(line.strip() for line in lines[1:])
        qa_pairs.append({'question': question, 'answer': answer})
    return qa_pairs

def create_embeddings(qa_pairs):
    global faq_embeddings
    for pair in qa_pairs:
        response = client.embeddings.create(input=pair['question'], model="text-embedding-3-small")
        pair['embedding'] = response.data[0].embedding
    faq_embeddings = qa_pairs

def cosine_similarity(v1, v2):
    # Added epsilon for numerical stability
    epsilon = 1e-9
    return np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + epsilon)

def faq_lookup_tool(context: BankAgentContext, user_query: str):
    context.topics_discussed.append(f"Q: {user_query}")
    similarity_threshold=0.70
    if not faq_embeddings: return json.dumps({"answer": "Knowledge base not available."})
    query_response = client.embeddings.create(input=user_query, model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding
    best_match_score, best_match_answer = -1, "NO_ANSWER_FOUND"
    for faq in faq_embeddings:
        similarity = cosine_similarity(np.array(query_embedding), np.array(faq['embedding']))
        if similarity > best_match_score:
            best_match_score = similarity
            if best_match_score > similarity_threshold: best_match_answer = faq['answer']
    return json.dumps({"answer": best_match_answer})

def create_case_tool(context: BankAgentContext, query: str, urgency: str):
    zip_code = input(f"Ava: To create a case for '{query}', I first need your 5-digit ZIP code. Can you please provide it? \nYou (ZIP): ").strip()
    case_number = f"CS{random.randint(1000000, 9999999)}"; context.case_number = case_number
    context.topics_discussed.append(f"Case Created ({case_number}) for: {query}")
    resolution_time = "within 24 hours" if urgency == "high" else "within 3 business days"
    return json.dumps({"confirmation": f"Thank you. A case ({case_number}) has been created. A specialist from the {zip_code} area will contact you {resolution_time}."})

def get_offers():
    try:
        with open("Data/Consumer Offers.txt", 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: return "Sorry, offers are unavailable."

def get_summary(messages: list) -> str:
    history_for_summary = []
    # FINAL BUG FIX: Use isinstance to robustly handle the mixed-type message list.
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get('content')
        else:
            content = msg.content
        if content and "You are Ava, a helpful digital assistant" not in content:
            history_for_summary.append(str(content))
    
    summary_prompt = (
        "You are the voice of Ava, the assistant. Summarize the help you provided to the user in a single, friendly sentence from your perspective. "
        "For example: 'Just to recap, I helped you with Zelle money transfers and ordering checks, and also created a case for your blocked account.' "
        "Be concise and do not use a list format."
    )
    summary_messages = [{"role":"system", "content": summary_prompt}, {"role":"user", "content": "\n".join(history_for_summary)}]
    
    summary_response = client.chat.completions.create(model="gpt-4o", messages=summary_messages)
    return summary_response.choices[0].message.content

def save_chat_log(context: BankAgentContext, messages: list):
    os.makedirs("chat_logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"chat_logs/chat_log_{ts}.txt"
    with open(fname, "w", encoding='utf-8') as f:
        f.write("--- CONVERSATION CONTEXT ---\n"); f.write(context.model_dump_json(indent=2))
        f.write("\n\n--- CHAT HISTORY ---\n")
        for msg in messages:
            if isinstance(msg, dict):
                role, content, tool_calls = msg.get('role'), msg.get('content'), msg.get('tool_calls')
            else:
                role, content, tool_calls = msg.role, msg.content, msg.tool_calls
            content = content or ''
            if tool_calls: content += str(tool_calls)
            f.write(f"[{role}]: {content}\n")
    print(f"\n[Chat log saved to {fname}]")

# --- 4. THE HYBRID AGENT LOOP ---
def main():
    print("Just a moment, your personal Fake Bank agent Ava is firing up...")
    context = BankAgentContext()
    qa_data = process_faq_file()
    create_embeddings(qa_data)
    
    cst_tz = ZoneInfo("America/Chicago"); current_time_cst = datetime.now(cst_tz)
    hour = current_time_cst.hour
    if 5 <= hour < 12: greeting = "Good morning"
    elif 12 <= hour < 18: greeting = "Good afternoon"
    else: greeting = "Good evening"
    print(f"\n👋 {greeting}! I’m Ava, your Fake Bank digital assistant.")
    
    tools = [
        {"type": "function", "function": {"name": "faq_lookup_tool", "description": "Looks up answers to frequently asked questions about personal banking.", "parameters": {"type": "object", "properties": {"user_query": {"type": "string"}}, "required": ["user_query"]}}},
        {"type": "function", "function": {"name": "create_case_tool", "description": "Creates a customer support case for a problem that cannot be solved by the FAQ.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "urgency": {"type": "string", "enum": ["high", "normal"]}},"required": ["query", "urgency"]}}},
    ]
    
    system_prompt = (
        "You are Ava, a helpful digital assistant for Fake Bank. Your job is to understand the user's request and decide which tool to use. "
        "For general questions, use the `faq_lookup_tool`. "
        "If the user has a problem (e.g., 'blocked', 'fraud') or if the FAQ tool returns 'NO_ANSWER_FOUND', use the `create_case_tool`. You MUST infer the urgency from the user's language. "
        "After every response, ask 'Is there anything else I can help with?'"
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]: break

        closing_triggers = ["no", "nope", "no thanks", "i'm good", "that's all", "no more questions", "im done", "all set", "that's it"]
        if any(trigger in user_input.lower() for trigger in closing_triggers):
            messages.append({"role": "user", "content": user_input})
            
            print("\nAva: Glad I could help. Before you go, would you be interested in hearing about some of our special savings offers?")
            offer_response = input("\nYou (yes/no): ").strip()
            messages.append({"role": "user", "content": offer_response})
            
            yes_triggers = ["yes", "yep", "sure", "ok", "yeah", "y"]
            if any(trigger in offer_response.lower() for trigger in yes_triggers):
                print("\nAva: Great! Here are some options that might interest you.\n")
                print(get_offers())
                messages.append({"role": "assistant", "content": "[Presented offers to the user]"})
                
                user_ack = input("\nYou: ").strip()
                messages.append({"role": "user", "content": user_ack})
            
            summary_sentence = get_summary(messages)
            
            final_closing_message = (
                f"\nAva: {summary_sentence} \n\nI hope I was able to resolve all your issues today. "
                f"On a scale of 1 to 5 (where 1 is poor and 5 is exceptional), how would you rate my performance?"
            )
            print(final_closing_message)
            messages.append({"role": "assistant", "content": final_closing_message})

            feedback_input = input("\nYou (1-5): ").strip()
            context.feedback = feedback_input
            messages.append({"role": "user", "content": f"Feedback rating: {feedback_input}"})
            
            print("\nAva: Thank you for your feedback and for choosing Fake Bank. Have a wonderful day!")
            messages.append({"role": "assistant", "content": "Thank you for your feedback!"})
            save_chat_log(context, messages)
            break

        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools, tool_choice="auto")
            response_message = response.choices[0].message
            messages.append(response_message)
            
            if response_message.tool_calls:
                if response_message.content: print(f"Ava: {response_message.content}")
                for tool_call in response_message.tool_calls:
                    function_name, args = tool_call.function.name, json.loads(tool_call.function.arguments)
                    
                    tool_output_json = ""
                    if function_name == "faq_lookup_tool":
                        tool_output_json = faq_lookup_tool(context=context, **args)
                    elif function_name == "create_case_tool":
                        tool_output_json = create_case_tool(context=context, **args)

                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": tool_output_json})

                final_response = client.chat.completions.create(model="gpt-4o", messages=messages)
                print(f"Ava: {final_response.choices[0].message.content}")
                messages.append(final_response.choices[0].message)
            else:
                print(f"Ava: {response_message.content}")
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()