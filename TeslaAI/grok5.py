# =================================================================
# agent_final.py – TeslaAI Assistant (Increment 10: Trip Planner + Nashville Features Finalized)
# =================================================================
# - TeslaAI coordinator dispatches to Trippy (trip planning), Foody (food stops),
#   Resty (rest areas), Scenic (scenic detours), Kidsy (kid activities),
#   Lodgy (lodging), Panicky (emergency needs), Veggie (Nashville vegan restaurants),
#   or Showsy (2025 kid-friendly shows)
# - Specialists use their tools then hand off cleanly back to TeslaAI
# - Plain text responses with hyphen bullets (no markdown)
# - Graceful exit on "bye/goodbye/etc."
# - Output sanitizer removes **, *, _, `, #, > artifacts
# - Fallback ensures control returns to TeslaAI if a specialist misses a handoff
# =================================================================

import os
import re
import asyncio
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from agents import (
    Agent,
    Runner,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallOutputItem,
    ItemHelpers,
    TResponseInputItem,
    function_tool,
    handoff,
    RunContextWrapper,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# ---------------- CONSTANTS ----------------
DEFAULT_ORIGIN_CITY = "Nashville"
DEST_CODE = {
    "chicago": "chi",
    "atlanta": "atl",
    "denver":  "den",
    "orlando": "orl",
    "dallas":  "dal",
}

# Agents that are "specialists" (used for fallback routing)
SPECIALIST_NAMES = {
    "Trippy",
    "Foody",
    "Resty",
    "Scenic",
    "Kidsy",
    "Lodgy",
    "Panicky",
    "Veggie",
    "Showsy",
}

# Natural-language exits caught at the main loop
EXIT_PHRASES = (
    "exit", "quit", "bye", "goodbye", "good bye", "see ya", "see you",
    "cya", "later", "talk later", "that’s it", "thats it",
    "we're done", "were done", "im done", "i'm done", "all set, bye",
    "thanks, bye", "thank you, bye", "all done"
)

def _clean_out(text: str) -> str:
    """Remove common markdown artifacts and collapse excess blank lines."""
    text = re.sub(r"[*_`#>]+", "", text)          # strip **, *, _, `, #, >
    text = re.sub(r"\n{3,}", "\n\n", text).strip() # collapse extra blank lines
    return text

# ---------------- CONTEXT ----------------
class TeslaTripContext(BaseModel):
    greeted: bool = False
    last_resolution: Optional[str] = None
    destination: Optional[str] = None
    current_plan: Optional[Dict[str, Any]] = None
    session_open: bool = True

# ---------------- TOOLS ----------------

@function_tool(
    name_override="get_trip_plan_tool",
    description_override="Load the JSON bundle for the requested route."
)
async def get_trip_plan_tool(destination: str) -> str:
    dest = destination.split(",")[0].strip().lower()
    code = DEST_CODE.get(dest)
    if not code:
        return json.dumps({"error": f"No route bundle for '{destination}'."})
    file_path = Path(__file__).parent / "Data" / f"route_nash-{code}.json"
    try:
        bundle = json.load(open(file_path, "r", encoding="utf-8"))
        # unwrap single-key dict like {"nashville-chicago": {...}}
        if isinstance(bundle, dict) and len(bundle) == 1:
            bundle = next(iter(bundle.values()))
        return json.dumps(bundle)
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path.name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool(
    name_override="food_options_tool",
    description_override="Return curated food stops per charging stop."
)
async def food_options_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps([
        {"stop": s["city"], "food_stops": s.get("food_stops", [])}
        for s in plan.get("stops", [])
    ])

@function_tool(
    name_override="rest_area_lookup_tool",
    description_override="Return curated rest areas for the route."
)
async def rest_area_lookup_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps(plan.get("rest_areas", []))

@function_tool(
    name_override="scenic_spots_tool",
    description_override="Return curated scenic spots for the route."
)
async def scenic_spots_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps(plan.get("scenic_spots", []))

@function_tool(
    name_override="kids_activities_tool",
    description_override="Return curated kid-friendly stops for the route."
)
async def kids_activities_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps(plan.get("kid_friendly_stops", []))

@function_tool(
    name_override="lodging_options_tool",
    description_override="Return curated lodging options per stop."
)
async def lodging_options_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps([
        {"stop": s["city"], "lodging_option": s["lodging_option"]}
        for s in plan.get("stops", []) if "lodging_option" in s
    ])

@function_tool(
    name_override="emergency_convenience_tool",
    description_override="Return curated emergency convenience details per stop."
)
async def emergency_convenience_tool(context: RunContextWrapper[TeslaTripContext]) -> str:
    plan = context.context.current_plan or {}
    return json.dumps([
        {"stop": s["city"], "emergency_convenience": s["emergency_convenience"]}
        for s in plan.get("stops", []) if "emergency_convenience" in s
    ])

@function_tool(
    name_override="vegan_restaurants_tool",
    description_override="Return three random vegan restaurants in Nashville."
)
async def vegan_restaurants_tool() -> str:
    file_path = Path(__file__).parent / "Data" / "nash_vegan.json"
    try:
        data = json.load(open(file_path, "r", encoding="utf-8"))
        places = data.get("places", [])
        # Select 3 random entries
        selected = random.sample(places, min(3, len(places)))
        return json.dumps(selected)
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path.name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool(
    name_override="kid_shows_tool",
    description_override="Return three random kid-friendly shows in Nashville for 2025."
)
async def kid_shows_tool() -> str:
    file_path = Path(__file__).parent / "Data" / "nash_kids.json"
    try:
        data = json.load(open(file_path, "r", encoding="utf-8"))
        events = data.get("events", [])
        # Select 3 random entries
        selected = random.sample(events, min(3, len(events)))
        return json.dumps(selected)
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path.name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ---------------- AGENTS ----------------

# 1) TeslaAI Coordinator
welcome_agent = Agent[TeslaTripContext](
    name="TeslaAI",
    handoff_description="Greet, handle 'done', or route user to Trippy, Foody, Resty, Scenic, Kidsy, Lodgy, Panicky, Veggie, or Showsy.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are TeslaAI, the in-car assistant based in Nashville.\n"
        "Use plain text. Do not use markdown, bold, or asterisks.\n\n"
        "If context.greeted is False:\n"
        "  Say: \"Hey y’all! I’m TeslaAI, your onboard companion from Nashville. I can plan road trips, find dinner spots, share Nashville’s shows and events, or find kid-friendly shows in 2025. What’s up?\"\n"
        "  Then set context.greeted=True and stop.\n\n"
        "Define closing_phrases as: ['no', 'no thanks', 'no thank you', 'that is all', 'that's all', 'done', 'all set', 'thank you', 'thanks', 'bye', 'goodbye', 'all done'].\n"
        "If the user's latest message matches any closing_phrases (case-insensitive):\n"
        "  If context.destination is 'orlando':\n"
        "    Say: \"Catch you later, enjoy your Orlando trip! Definitely check out SeaWorld, Disney World, and Universal Studios for fun!\"\n"
        "    Then set context.last_resolution=None and context.session_open=False and stop.\n"
        "  Else if context.destination is 'chicago':\n"
        "    Say: \"Catch you later, enjoy your Chicago trip! Check out Navy Pier, Millennium Park, and the Art Institute of Chicago for fun!\"\n"
        "    Then set context.last_resolution=None and context.session_open=False and stop.\n"
        "  Else if context.destination is not None:\n"
        "    Say: \"Catch you later, enjoy your {context.destination} trip!\"\n"
        "    Then set context.last_resolution=None and context.session_open=False and stop.\n"
        "  Else:\n"
        "    Say: \"Catch you later, y’all!\"\n"
        "    Then set context.last_resolution=None and context.session_open=False and stop.\n\n"
        "If context.last_resolution is not None and the user did NOT say a closing phrase:\n"
        "  If the user’s message contains 'food' or 'restaurant' or 'dinner' and 'nashville', hand off to Veggie.\n"
        "  If the user’s message contains 'food' or 'restaurant' or 'dinner', hand off to Foody.\n"
        "  If the user’s message contains 'rest' or 'rest area', hand off to Resty.\n"
        "  If the user’s message contains 'scenic' or 'view' or 'detour' or 'tours', hand off to Scenic.\n"
        "  If the user’s message contains 'kid' or 'kids' or 'family' and 'show' or 'event' or '2025', hand off to Showsy.\n"
        "  If the user’s message contains 'kid' or 'kids' or 'family', hand off to Kidsy.\n"
        "  If the user’s message contains 'lodge' or 'hotel' or 'lodging', hand off to Lodgy.\n"
        "  If the user’s message contains 'emergency' or 'services' or 'convenience', hand off to Panicky.\n"
        "  Else, say: \"Now that your {context.destination} trip is planned, looking for vegan dinner options with your wife in Nashville or kid-friendly shows for your child in 2025? Or try food, rest areas, scenic detours, kid activities, lodging, emergency info, or something else?\"\n"
        "  Then stop.\n\n"
        "Otherwise, look at the user’s message:\n"
        "- If it contains 'trip' or 'plan' with a place name, hand off to Trippy.\n"
        "- If it contains 'food' or 'restaurant' or 'dinner' and 'nashville', hand off to Veggie.\n"
        "- If it contains 'food' or 'restaurant' or 'dinner', hand off to Foody.\n"
        "- If it contains 'rest' or 'rest area', hand off to Resty.\n"
        "- If it contains 'scenic' or 'view' or 'detour' or 'tours', hand off to Scenic.\n"
        "- If it contains 'kid' or 'kids' or 'family' and 'show' or 'event' or '2025', hand off to Showsy.\n"
        "- If it contains 'kid' or 'kids' or 'family', hand off to Kidsy.\n"
        "- If it contains 'lodge' or 'hotel' or 'lodging', hand off to Lodgy.\n"
        "- If it contains 'emergency' or 'services' or 'convenience', hand off to Panicky.\n"
        "- For any other request, say: \"Not sure about that one. Want to plan a trip, find food, rest areas, scenic detours, kid activities, lodging, emergency info, vegan dinner in Nashville, or kid-friendly shows in 2025?\"\n"
        "  Then stop.\n\n"
        "If a specialist just responded, say: \"Now that your {context.destination} trip is planned, looking for vegan dinner options with your wife in Nashville or kid-friendly shows for your child in 2025? Or try food, rest areas, scenic detours, kid activities, lodging, emergency info, or something else?\"\n"
        "Then stop."
    ),
    handoffs=[]
)

# 2) Trippy
trippy_agent = Agent[TeslaTripContext](
    name="Trippy",
    handoff_description="Load route bundle, present summary, hand back to TeslaAI.",
    tools=[get_trip_plan_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Trippy. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'trip' or 'plan' with a destination, do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text with hyphen bullets. Do not use markdown, bold, or asterisks.\n\n"
        "1) Extract the destination from the user’s last message.\n"
        "2) Call get_trip_plan_tool once.\n"
        "3) If the JSON has 'error': say you don’t have that route, execute: set context.last_resolution=None, and hand off to TeslaAI.\n"
        "4) Else: load into context.current_plan, execute: set context.destination and context.last_resolution='plan'.\n"
        "5) Output exactly this structure exactly once:\n"
        "   Travel tip: {bundle.travel_tip}\n"
        "   The trip covers {bundle.total_distance_miles} miles and takes about {bundle.total_duration_hours} hours.\n"
        "   - In {city}, charge for {charge_time_minutes} minutes. (one bullet per stop, in order)\n"
        "6) Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 3) Foody
foody_agent = Agent[TeslaTripContext](
    name="Foody",
    handoff_description="Summarize food stops along the route, then hand back to TeslaAI.",
    tools=[food_options_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Foody. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'food' or 'restaurant' or 'dinner', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call food_options_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are food options along your route:\n"
        "   - In {stop}, try {name} ({type}) — {description}.\n"
        "3. Execute: set context.last_resolution='food'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 4) Resty
resty_agent = Agent[TeslaTripContext](
    name="Resty",
    handoff_description="Summarize rest areas along the route, then hand back to TeslaAI.",
    tools=[rest_area_lookup_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Resty. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'rest' or 'rest area', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call rest_area_lookup_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are rest areas along your route:\n"
        "   - {name} at {location} — {description}.\n"
        "   If the word 'sirup' appears, write it as 'syrup'.\n"
        "3. Execute: set context.last_resolution='rest'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 5) Scenic
scenic_agent = Agent[TeslaTripContext](
    name="Scenic",
    handoff_description="Summarize scenic detours along the route, then hand back to TeslaAI.",
    tools=[scenic_spots_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Scenic. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'scenic' or 'view' or 'detour' or 'tours', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call scenic_spots_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are scenic detours you might enjoy:\n"
        "   - {name} in {location} — {description}.\n"
        "3. Execute: set context.last_resolution='scenic'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 6) Kidsy
kidsy_agent = Agent[TeslaTripContext](
    name="Kidsy",
    handoff_description="Summarize kid-friendly stops along the route, then hand back to TeslaAI.",
    tools=[kids_activities_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Kidsy. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'kid' or 'kids' or 'family', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call kids_activities_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are kid-friendly stops along your route:\n"
        "   - {name} in {location} — {description}.\n"
        "3. Execute: set context.last_resolution='kids'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 7) Lodgy
lodgy_agent = Agent[TeslaTripContext](
    name="Lodgy",
    handoff_description="Summarize lodging options along the route, then hand back to TeslaAI.",
    tools=[lodging_options_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Lodgy. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'lodge' or 'hotel' or 'lodging', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call lodging_options_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are lodging options along your route:\n"
        "   - At {stop}, stay at {name} with amenities: {amenities as a comma-separated list}.\n"
        "3. Execute: set context.last_resolution='lodge'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 8) Panicky
panicky_agent = Agent[TeslaTripContext](
    name="Panicky",
    handoff_description="Summarize emergency convenience details along the route, then hand back to TeslaAI.",
    tools=[emergency_convenience_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Panicky. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your trip’. If the user’s message does not EXACTLY contain 'emergency' or 'services' or 'convenience', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call emergency_convenience_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here is emergency info by stop:\n"
        "   - At {stop}, pharmacy: {pharmacy}, grocery: {grocery}.\n"
        "3. Execute: set context.last_resolution='emergency'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 9) Veggie
veggie_agent = Agent[TeslaTripContext](
    name="Veggie",
    handoff_description="Summarize three random vegan restaurants in Nashville, then hand back to TeslaAI.",
    tools=[vegan_restaurants_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Veggie. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy your meal’. If the user’s message does not EXACTLY contain 'vegan' or 'plant-based' or 'dinner' and 'nashville', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call vegan_restaurants_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are three vegan dinner options in Nashville:\n"
        "   - {name} ({type}) — {description}, {address}.\n"
        "3. Execute: set context.last_resolution='vegan'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# 10) Showsy
showsy_agent = Agent[TeslaTripContext](
    name="Showsy",
    handoff_description="Summarize three random kid-friendly shows in Nashville for 2025, then hand back to TeslaAI.",
    tools=[kid_shows_tool],
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You are Showsy. Output only the specified structure exactly once, then hand off to TeslaAI, no exceptions. Do not append any phrases like ‘Transferring back’ or ‘Enjoy the show’. If the user’s message does not EXACTLY contain 'kid' or 'kids' or 'family' and 'show' or 'event' or '2025', do not respond under any circumstances and immediately hand off to TeslaAI.\n"
        "Use plain text. Do not use markdown or asterisks.\n\n"
        "1. Call kid_shows_tool and parse JSON.\n"
        "2. Output exactly this structure exactly once: Here are three kid-friendly shows in Nashville for 2025:\n"
        "   - {name} in {typical_window_2025} — {description}, at {likely_venues}.\n"
        "3. Execute: set context.last_resolution='shows'.\n"
        "4. Then hand off to TeslaAI, no exceptions. Do not repeat the output or include code actions like ‘set context.last_resolution’ in the output."
    ),
    handoffs=[]
)

# ---------------- HANDOFFS ----------------
welcome_agent.handoffs = [
    handoff(agent=trippy_agent),
    handoff(agent=foody_agent),
    handoff(agent=resty_agent),
    handoff(agent=scenic_agent),
    handoff(agent=kidsy_agent),
    handoff(agent=lodgy_agent),
    handoff(agent=panicky_agent),
    handoff(agent=veggie_agent),
    handoff(agent=showsy_agent),
]
for ag in (trippy_agent, foody_agent, resty_agent, scenic_agent, kidsy_agent, lodgy_agent, panicky_agent, veggie_agent, showsy_agent):
    ag.handoffs = [handoff(agent=welcome_agent)]

# ---------------- MAIN LOOP ----------------
async def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")

    print("TeslaAI Assistant booting up...")
    context = TeslaTripContext()
    current = welcome_agent
    history: List[TResponseInputItem] = []

    while context.session_open:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # natural-language exits
        lower = user_input.lower()
        if any(p in lower for p in EXIT_PHRASES):
            if context.destination == "orlando":
                print("TeslaAI: Catch you later, enjoy your Orlando trip! Definitely check out SeaWorld, Disney World, and Universal Studios for fun!")
            elif context.destination == "chicago":
                print("TeslaAI: Catch you later, enjoy your Chicago trip! Check out Navy Pier, Millennium Park, and the Art Institute of Chicago for fun!")
            elif context.destination:
                print(f"TeslaAI: Catch you later, enjoy your {context.destination} trip!")
            else:
                print("TeslaAI: Catch you later, y’all!")
            break

        history.append({"role": "user", "content": user_input})
        result = await Runner.run(current, history, context=context)

        # reset context on trip error
        for item in result.new_items:
            if isinstance(item, ToolCallOutputItem) and item.agent.name == "Trippy":
                try:
                    data = json.loads(item.output)
                    if isinstance(data, dict) and "error" in data:
                        context.current_plan = None
                        context.last_resolution = None
                except Exception:
                    context.current_plan = None
                    context.last_resolution = None

        # render outputs and capture handoffs
        next_agent = None
        last_speaker = None
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                last_speaker = item.agent.name
                txt = _clean_out(ItemHelpers.text_message_output(item))
                print(f"{item.agent.name}: {txt}")
            if isinstance(item, HandoffOutputItem):
                # Log the detected intent for debugging
                keyword = "unknown"
                lower_input = history[-1]["content"].lower()
                if "vegan" in lower_input or "plant-based" in lower_input or ("dinner" in lower_input and "nashville" in lower_input):
                    keyword = "vegan/plant-based/dinner+nashville"
                elif ("kid" in lower_input or "kids" in lower_input or "family" in lower_input) and ("show" in lower_input or "event" in lower_input or "2025" in lower_input):
                    keyword = "kid/kids/family+show/event/2025"
                elif "emergency" in lower_input or "services" in lower_input or "convenience" in lower_input:
                    keyword = "emergency/services/convenience"
                elif "kid" in lower_input or "kids" in lower_input or "family" in lower_input:
                    keyword = "kid/kids/family"
                elif "lodge" in lower_input or "hotel" in lower_input or "lodging" in lower_input:
                    keyword = "lodge/hotel/lodging"
                elif "scenic" in lower_input or "view" in lower_input or "detour" in lower_input or "tours" in lower_input:
                    keyword = "scenic/view/detour/tours"
                elif "rest" in lower_input or "rest area" in lower_input:
                    keyword = "rest/rest area"
                elif "food" in lower_input or "restaurant" in lower_input or "dinner" in lower_input:
                    keyword = "food/restaurant/dinner"
                elif "trip" in lower_input or "plan" in lower_input:
                    keyword = "trip/plan"
                print(f"--- HANDOFF from {item.source_agent.name} to {item.target_agent.name} (Intent: {keyword}) ---")
                next_agent = item.target_agent

        # Fallback: if a specialist spoke but didn't hand off, return to TeslaAI
        if next_agent is None and last_speaker in SPECIALIST_NAMES:
            print(f"--- FORCED HANDOFF from {last_speaker} to TeslaAI (Responded to: {history[-1]['content']}) ---")
            next_agent = welcome_agent

        # Decide who handles next turn
        current = next_agent or result.last_agent
        # Trigger post-trip prompt after specialist response
        if last_speaker in SPECIALIST_NAMES and context.last_resolution:
            print(f"TeslaAI: Now that your {context.destination} trip is planned, looking for vegan dinner options with your wife in Nashville or kid-friendly shows for your child in 2025? Or try food, rest areas, scenic detours, kid activities, lodging, emergency info, or something else?")
        history = result.to_input_list()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")