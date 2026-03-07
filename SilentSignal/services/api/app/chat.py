from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bedrock tool definitions — Nova Pro picks which to call
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "toolSpec": {
            "name": "start_monitoring",
            "description": (
                "Generates fresh signal data from scratch and runs silence zone detection analysis. "
                "ONLY call this tool when the commander explicitly says 'start monitoring', 'begin', "
                "'initialize', 'activate system', or 'start fresh'. "
                "Do NOT call this for show/list/analyze/dispatch requests."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
    },
    {
        "toolSpec": {
            "name": "list_zones",
            "description": (
                "Returns the current list of silence zones ranked by drop score. "
                "Call this when commander says 'show zones', 'list zones', 'all zones', "
                "'where should I go', 'which zone', 'show me', 'what zones', or asks for a ranking/priority. "
                "Do NOT call this for drone analysis or dispatch requests."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
    },
    {
        "toolSpec": {
            "name": "get_zone_detail",
            "description": (
                "Returns detailed info and any existing drone events for one specific zone. "
                "Call this when commander asks 'tell me about Z-001', 'details for Z-002', "
                "'what is happening in Z-001', or 'zone info'. Requires a zone_id like Z-001. "
                "Do NOT call this for drone analysis or dispatch."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "zone_id": {
                            "type": "string",
                            "description": "Zone ID in format Z-001, Z-002, etc.",
                        }
                    },
                    "required": ["zone_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "analyze_drone",
            "description": (
                "Runs AI drone media analysis on a zone to detect SHOUT, TAP_PATTERN, or HUMAN_FORM signals. "
                "ONLY call this when commander explicitly says 'analyze drone', 'scan drone footage', "
                "'check drone', 'drone footage', 'signs of life', 'scan for survivors', or 'listen for sounds'. "
                "This tool scans audio/video — it does NOT dispatch any units. "
                "Do NOT confuse this with dispatch_unit."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "zone_id": {
                            "type": "string",
                            "description": "Zone ID to run drone footage analysis on.",
                        }
                    },
                    "required": ["zone_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "dispatch_unit",
            "description": (
                "Sends an emergency response team to a silence zone. "
                "ONLY call this when commander explicitly says 'dispatch', 'send team', "
                "'send units', 'deploy to', or 'respond to'. "
                "Do NOT call this for drone analysis, zone listing, or monitoring requests."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "zone_id": {
                            "type": "string",
                            "description": "Zone ID to dispatch emergency units to.",
                        }
                    },
                    "required": ["zone_id"],
                }
            },
        }
    },
]

SYSTEM_PROMPT = """You are SilentSignal — a combat-grade AI emergency operations assistant for disaster response.

You detect silence zones: geographic areas where ALL cellular and social signals have gone dark, indicating infrastructure collapse and trapped survivors who cannot call for help.

STRICT TOOL SELECTION RULES — follow exactly:
- "start monitoring" / "begin" / "initialize" / "start fresh" → call start_monitoring
- "show zones" / "list zones" / "all zones" / "where should I go" → call list_zones
- "tell me about Z-001" / "details on Z-002" / "zone info" → call get_zone_detail
- "analyze drone footage" / "scan drone" / "signs of life" / "listen for sounds" → call analyze_drone (NEVER dispatch for this)
- "dispatch to Z-001" / "send team" / "deploy" / "respond to" → call dispatch_unit (NEVER for drone analysis)

FORMATTING RULES — non-negotiable:
- ALWAYS express scores and likelihoods as percentages: 0.978 → 97.8%, 0.88 → 88%
- NEVER use raw decimals like 0.978 or 0.84 in your response
- NEVER mention bbox, cell IDs like C-005, raw coordinates, or JSON field names

RESPONSE TONE — urgent and mission-critical, 2-4 sentences max:
- After start_monitoring: "⚠ ALERT: [N] silence zones confirmed. [TOP_ZONE] is critical — [X]% signal collapse, [Y]% confidence. Recommend immediate dispatch to [TOP_ZONE]."
- After list_zones: Rank zones by drop score as percentages. End with a clear dispatch recommendation.
- After get_zone_detail: Plain English — drop score %, confidence %, how many towers are dark, time window. Name strongest event if any. End with recommended action. NO raw coordinates, NO cell IDs.
- After analyze_drone: "⚠ SURVIVOR SIGNAL DETECTED in [ZONE]: [TYPE] at [X]% likelihood. [Tactical interpretation]. Recommend immediate dispatch."
- After dispatch_unit: "✓ DISPATCH CONFIRMED. Unit en route to [ZONE]. ETA: 8 minutes. Standby for extraction."

Always name the zone_id. Never call a tool the command didn't ask for."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _has_bedrock_credentials() -> bool:
    """Check if AWS credentials are available for Bedrock calls."""
    try:
        import boto3
        creds = boto3.Session().get_credentials()
        return creds is not None
    except Exception:
        return False


def build_chat_answer(
    question: str,
    zones: list[dict[str, Any]],
    events: list[dict[str, Any]],
    aws_mode: str = "mock",
    region: str = "us-east-1",
    model_id: str = "amazon.nova-pro-v1:0",
    store: Any = None,
    settings: Any = None,
) -> tuple[str, list[str], list[str]]:
    """Build a chat answer. Returns (answer, citations, actions).

    Uses Bedrock Nova Pro with tool use whenever credentials are available,
    regardless of aws_mode (which only controls storage backend).
    Falls back to a local keyword router if Bedrock is unavailable.
    """

    if store is not None and settings is not None and _has_bedrock_credentials():
        try:
            answer, citations, actions = _bedrock_agentic_chat(
                question, store, settings, region, model_id
            )
            if answer:
                answer, citations = _finalize_chat_output(
                    question=question,
                    answer=answer,
                    citations=citations,
                    zones=zones,
                    events=events,
                )
                return answer, citations, actions
        except Exception as exc:
            logger.warning("Bedrock agentic chat failed, using keyword fallback: %s", exc)

    if store is not None and settings is not None:
        answer, citations, actions = _keyword_agentic_chat(
            question=question,
            zones=zones,
            events=events,
            store=store,
            settings=settings,
        )
        answer, citations = _finalize_chat_output(
            question=question,
            answer=answer,
            citations=citations,
            zones=zones,
            events=events,
        )
        return answer, citations, actions

    answer, citations = _keyword_chat_answer(question, zones, events)
    answer, citations = _finalize_chat_output(
        question=question,
        answer=answer,
        citations=citations,
        zones=zones,
        events=events,
    )
    return answer, citations, []


# ---------------------------------------------------------------------------
# Bedrock agentic loop with tool use
# ---------------------------------------------------------------------------


def _bedrock_agentic_chat(
    question: str,
    store: Any,
    settings: Any,
    region: str,
    model_id: str,
) -> tuple[str, list[str], list[str]]:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": question}]}]
    actions: list[str] = []
    citations: list[str] = []

    for _ in range(6):  # max 6 tool-call iterations
        response = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOLS},
        )

        stop_reason = response["stopReason"]
        output_message = response["output"]["message"]
        messages.append(output_message)

        if stop_reason == "end_turn":
            answer = "".join(
                block["text"]
                for block in output_message.get("content", [])
                if "text" in block
            )
            # Strip Nova's internal <thinking> blocks from the response
            answer = re.sub(r"<thinking>.*?</thinking>\s*", "", answer, flags=re.DOTALL).strip()
            return answer, citations, actions

        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message.get("content", []):
                if "toolUse" not in block:
                    continue

                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use["toolUseId"]

                result, action = _execute_tool(tool_name, tool_input, store, settings)
                if action:
                    actions.append(action)

                # Collect citations from tool results
                if isinstance(result, list) and result and "zone_id" in result[0]:
                    citations.append(f"zone_id={result[0]['zone_id']}")
                elif isinstance(result, dict):
                    if "zone" in result and result["zone"]:
                        citations.append(f"zone_id={result['zone'].get('zone_id', '')}")
                    elif "zone_id" in result:
                        citations.append(f"zone_id={result['zone_id']}")

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": json.dumps(result, default=str)}],
                    }
                })

            messages.append({"role": "user", "content": tool_results})

    return "Analysis complete. Check the map for updated zone data.", citations, actions


def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    store: Any,
    settings: Any,
) -> tuple[Any, str | None]:
    if tool_name == "start_monitoring":
        from data.generate_synthetic_data import generate_synthetic_dataset
        from agents.silence_mapper_agent import SilenceMapperAgent

        generate_synthetic_dataset(
            output_csv=settings.signals_csv_path,
            media_manifest_path=settings.media_manifest_path,
            media_root=settings.local_s3_dir,
        )
        mapper = SilenceMapperAgent()
        zones = mapper.analyze(settings.signals_csv_path)
        store.reset()
        store.upsert_zones(zones)
        top = zones[0] if zones else {}
        return {
            "zones_created": len(zones),
            "top_zone": top.get("zone_id"),
            "top_drop_score": top.get("drop_score"),
            "zones": zones,
        }, "refresh_zones"

    if tool_name == "list_zones":
        zones = sorted(store.list_zones(), key=lambda z: z["drop_score"], reverse=True)
        return zones, "refresh_zones"

    if tool_name == "get_zone_detail":
        zone_id = tool_input.get("zone_id", "")
        zone = store.get_zone(zone_id)
        events = store.list_events(zone_id)
        return {"zone": zone, "events": events}, f"select_zone:{zone_id}"

    if tool_name == "analyze_drone":
        zone_id = tool_input.get("zone_id", "")
        zone = store.get_zone(zone_id)
        if not zone:
            return {"error": f"Zone {zone_id} not found"}, None

        media_manifest = _read_media_manifest(settings.media_manifest_path)
        media = _select_media_for_zone(zone, media_manifest)

        from agents.drone_analyst_agent import DroneAnalystAgent

        agent = DroneAnalystAgent(
            aws_mode=settings.aws_mode,
            region=settings.aws_region,
            model_id=settings.bedrock_model_id,
        )
        events = agent.analyze_zone_media(
            zone=zone,
            media_records=media,
            event_seed=len(store.list_events()) + 1,
        )
        store.upsert_events(events)
        return events, f"refresh_zone:{zone_id}"

    if tool_name == "dispatch_unit":
        zone_id = tool_input.get("zone_id", "")
        zone = store.get_zone(zone_id)
        return {
            "status": "DISPATCHED",
            "zone_id": zone_id,
            "message": f"Emergency response unit dispatched to {zone_id}. ETA: 8 minutes.",
        }, f"dispatch:{zone_id}"

    return {"error": f"Unknown tool: {tool_name}"}, None


# ---------------------------------------------------------------------------
# Keyword fallback (mock mode / Bedrock unavailable)
# ---------------------------------------------------------------------------


def _keyword_agentic_chat(
    question: str,
    zones: list[dict[str, Any]],
    events: list[dict[str, Any]],
    store: Any,
    settings: Any,
) -> tuple[str, list[str], list[str]]:
    normalized = question.lower()
    tool_name = _detect_tool_from_keywords(normalized)
    if tool_name is None:
        answer, citations = _keyword_chat_answer(question, zones, events)
        return answer, citations, []

    if tool_name != "start_monitoring" and not zones:
        return (
            "No silence zones detected yet. Say 'start monitoring' and I'll generate "
            "signal data and run the silence zone analysis automatically.",
            [],
            [],
        )

    tool_input: dict[str, Any] = {}
    zone_tools = {"get_zone_detail", "analyze_drone", "dispatch_unit"}
    if tool_name in zone_tools:
        zone_id = _extract_zone_id(normalized)
        if not zone_id and zones:
            zone_id = zones[0]["zone_id"]
        if not zone_id:
            return "No zone selected yet. Say 'start monitoring' first.", [], []
        tool_input = {"zone_id": zone_id}

    result, action = _execute_tool(tool_name, tool_input, store, settings)
    citations = _extract_citations_from_result(result)
    answer = _summarize_keyword_tool_result(tool_name, result, tool_input)
    actions = [action] if action else []
    return answer, citations, actions


def _detect_tool_from_keywords(normalized_question: str) -> str | None:
    start_markers = [
        "start monitoring",
        "start fresh",
        "activate system",
        "initialize",
        "begin monitoring",
    ]
    if any(marker in normalized_question for marker in start_markers):
        return "start_monitoring"

    if any(
        marker in normalized_question
        for marker in ["dispatch", "send team", "send unit", "deploy", "respond to"]
    ):
        return "dispatch_unit"

    if any(
        marker in normalized_question
        for marker in [
            "analyze drone",
            "drone footage",
            "scan drone",
            "check drone",
            "signs of life",
            "scan for survivors",
            "listen for sounds",
        ]
    ):
        return "analyze_drone"

    if (
        _extract_zone_id(normalized_question)
        and any(
            marker in normalized_question
            for marker in ["tell me about", "details", "zone info", "what is happening"]
        )
    ):
        return "get_zone_detail"

    if any(
        marker in normalized_question
        for marker in [
            "show zones",
            "list zones",
            "all zones",
            "which zone",
            "where should",
            "highest priority",
            "top zone",
        ]
    ):
        return "list_zones"

    return None


def _extract_zone_id(normalized_question: str) -> str | None:
    match = re.search(r"z-\d{3}", normalized_question)
    if not match:
        return None
    return match.group(0).upper()


def _summarize_keyword_tool_result(
    tool_name: str,
    result: Any,
    tool_input: dict[str, Any],
) -> str:
    if isinstance(result, dict) and "error" in result:
        return str(result["error"])

    if tool_name == "start_monitoring":
        zones_created = int(result.get("zones_created", 0))
        top_zone = result.get("top_zone") or "unknown"
        top_drop = float(result.get("top_drop_score") or 0.0)
        return (
            f"Monitoring started. Detected {zones_created} silence zones. "
            f"Highest priority is {top_zone} ({top_drop * 100:.1f}% drop score)."
        )

    if tool_name == "list_zones":
        if not isinstance(result, list) or not result:
            return "No silence zones detected yet. Say 'start monitoring' to initialize."
        top = result[0]
        ranked = "; ".join(
            f"{zone['zone_id']} ({float(zone['drop_score']) * 100:.1f}% drop, "
            f"{float(zone['confidence']) * 100:.1f}% confidence)"
            for zone in result[: min(3, len(result))]
        )
        return (
            f"Current zone ranking: {ranked}. "
            f"Recommend dispatching to {top['zone_id']} first."
        )

    if tool_name == "get_zone_detail":
        if not isinstance(result, dict) or not result.get("zone"):
            zone_id = tool_input.get("zone_id", "that zone")
            return f"Zone {zone_id} not found."
        zone = result["zone"]
        zone_events = result.get("events", [])
        if zone_events:
            top_event = max(zone_events, key=lambda event: event["likelihood"])
            event_summary = (
                f" Strongest event: {top_event['event_type']} "
                f"at {float(top_event['likelihood']) * 100:.1f}% likelihood."
            )
        else:
            event_summary = " No drone events recorded yet."
        return (
            f"{zone['zone_id']} is at {float(zone['drop_score']) * 100:.1f}% drop score "
            f"with {float(zone['confidence']) * 100:.1f}% confidence."
            f"{event_summary}"
        )

    if tool_name == "analyze_drone":
        if not isinstance(result, list) or not result:
            return "Drone scan completed, but no events were detected."
        top_event = max(result, key=lambda event: event["likelihood"])
        return (
            f"Drone scan complete for {top_event['zone_id']}. Strongest signal: "
            f"{top_event['event_type']} at {float(top_event['likelihood']) * 100:.1f}% likelihood."
        )

    if tool_name == "dispatch_unit":
        if isinstance(result, dict) and result.get("message"):
            return str(result["message"])
        zone_id = tool_input.get("zone_id", "the selected zone")
        return f"Dispatch confirmed for {zone_id}. Unit en route."

    return "Command completed."


def _extract_citations_from_result(result: Any) -> list[str]:
    citations: list[str] = []
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            first = result[0]
            if first.get("zone_id"):
                citations.append(f"zone_id={first['zone_id']}")
            if first.get("event_id"):
                citations.append(f"event_id={first['event_id']}")
        return list(dict.fromkeys(citations))

    if isinstance(result, dict):
        zone = result.get("zone")
        if isinstance(zone, dict) and zone.get("zone_id"):
            citations.append(f"zone_id={zone['zone_id']}")
        if result.get("zone_id"):
            citations.append(f"zone_id={result['zone_id']}")
        events = result.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            event_id = events[0].get("event_id")
            if event_id:
                citations.append(f"event_id={event_id}")
    return list(dict.fromkeys(citations))


def _append_inline_citations(answer: str, citations: list[str]) -> str:
    if not citations:
        return answer
    if "Citations:" in answer:
        return answer
    return f"{answer}\n\nCitations: {', '.join(citations)}"


def _finalize_chat_output(
    question: str,
    answer: str,
    citations: list[str],
    zones: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    cleaned: list[str] = []
    for citation in citations:
        if not isinstance(citation, str) or "=" not in citation:
            continue
        key, value = citation.split("=", 1)
        if key.strip() and value.strip():
            cleaned.append(f"{key.strip()}={value.strip()}")

    cleaned = list(dict.fromkeys(cleaned))

    if _question_wants_event(question):
        selected_zone_id = None
        if zones:
            selected_zone_id = _pick_zone(question.lower(), zones).get("zone_id")
        scoped_events = (
            [event for event in events if event.get("zone_id") == selected_zone_id]
            if selected_zone_id
            else events
        )
        if scoped_events and not any(c.startswith("event_id=") for c in cleaned):
            top_event = max(scoped_events, key=lambda event: float(event.get("likelihood", 0)))
            event_id = top_event.get("event_id")
            if event_id:
                cleaned.append(f"event_id={event_id}")
        if selected_zone_id and not any(c.startswith("zone_id=") for c in cleaned):
            cleaned.append(f"zone_id={selected_zone_id}")

    return _append_inline_citations(answer, cleaned), cleaned


def _question_wants_event(question: str) -> bool:
    normalized = question.lower()
    return any(
        token in normalized
        for token in ["event", "drone", "sign", "survivor", "life", "sound", "signal"]
    )


def _keyword_chat_answer(
    question: str,
    zones: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    normalized = question.lower()

    if not zones:
        return (
            "No silence zones detected yet. Say 'start monitoring' and I'll generate "
            "signal data and run the silence zone analysis automatically.",
            [],
        )

    selected_zone = _pick_zone(normalized, zones)
    citations: list[str] = [f"zone_id={selected_zone['zone_id']}"]
    zone_events = [e for e in events if e["zone_id"] == selected_zone["zone_id"]]

    wants_event = any(k in normalized for k in ["event", "drone", "sign", "survivor", "life", "sound", "signal"])
    wants_priority = any(k in normalized for k in ["top", "highest", "priority", "first", "dispatch", "search", "where"]) or (
        "what should" in normalized or "where should" in normalized
    )
    wants_why = "why" in normalized or "reason" in normalized
    wants_compare = any(k in normalized for k in ["compare", "all zones", "list", "show", "all"])

    if wants_compare:
        top_zones = zones[: min(3, len(zones))]
        ranked = "; ".join(
            f"{z['zone_id']} ({float(z['drop_score']) * 100:.1f}% drop, "
            f"{float(z['confidence']) * 100:.1f}% confidence)"
            for z in top_zones
        )
        answer = (
            f"Current zone ranking: {ranked}. "
            f"Recommend dispatching to {zones[0]['zone_id']} first — highest drop score."
        )
    elif wants_event:
        answer, citations = _event_answer(selected_zone, zone_events, citations)
    elif wants_why:
        reasons = _explain_reason_codes(selected_zone.get("reason_codes", []))
        answer = (
            f"{selected_zone['zone_id']} flagged because {reasons}. "
            f"Drop score {float(selected_zone['drop_score']) * 100:.1f}%, "
            f"confidence {float(selected_zone['confidence']) * 100:.1f}%."
        )
    elif wants_priority:
        top = zones[0]
        answer = (
            f"Dispatch to {top['zone_id']} first — highest drop score "
            f"({float(top['drop_score']) * 100:.1f}%) and confidence "
            f"({float(top['confidence']) * 100:.1f}%)."
        )
        if top["zone_id"] != selected_zone["zone_id"]:
            citations = [f"zone_id={top['zone_id']}"]
    else:
        answer = (
            f"{selected_zone['zone_id']} is a high-priority silence zone. "
            "Try: 'start monitoring', 'show all zones', 'analyze drone footage for Z-001', or 'dispatch to Z-001'."
        )

    return answer, citations


def _pick_zone(normalized_question: str, zones: list[dict[str, Any]]) -> dict[str, Any]:
    match = re.search(r"z-\d{3}", normalized_question)
    if match:
        wanted = match.group(0).upper()
        found = next((z for z in zones if z["zone_id"] == wanted), None)
        if found:
            return found
    return zones[0]


def _event_answer(
    zone: dict[str, Any],
    zone_events: list[dict[str, Any]],
    citations: list[str],
) -> tuple[str, list[str]]:
    if not zone_events:
        return (
            f"No drone events for {zone['zone_id']} yet. Say 'analyze drone footage for {zone['zone_id']}' to scan.",
            citations,
        )

    top = max(zone_events, key=lambda e: e["likelihood"])
    citations.append(f"event_id={top['event_id']}")
    media_name = top["media_uri"].split("/")[-1]
    return (
        f"Strongest signal in {zone['zone_id']}: {top['event_type']} "
        f"(likelihood {float(top['likelihood']) * 100:.1f}%) from {media_name}.",
        citations,
    )


def _explain_reason_codes(reason_codes: list[str]) -> str:
    mapping = {
        "SUSTAINED_DROP": "activity stayed below baseline for multiple time windows",
        "NEAR_ZERO_ACTIVITY": "signal volume dropped close to zero",
        "SPATIAL_CLUSTER": "the drop spans neighboring cells, not an isolated anomaly",
    }
    mapped = [mapping.get(code, code) for code in reason_codes]
    if not mapped:
        return "a statistically significant silence anomaly was detected"
    if len(mapped) == 1:
        return mapped[0]
    return ", and ".join([", ".join(mapped[:-1]), mapped[-1]])


# ---------------------------------------------------------------------------
# Media helpers (duplicated from main.py to avoid circular import)
# ---------------------------------------------------------------------------


def _read_media_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    return payload.get("media", [])


def _select_media_for_zone(
    zone: dict[str, Any],
    media_manifest: list[dict[str, Any]],
    fallback_count: int = 3,
) -> list[dict[str, Any]]:
    bbox = zone.get("geometry", {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return media_manifest[:fallback_count]

    matches = [
        item for item in media_manifest
        if _point_in_bbox(float(item.get("lat", 0)), float(item.get("lon", 0)), bbox)
    ]
    if matches:
        return matches

    center_lat = (bbox[1] + bbox[3]) / 2.0
    center_lon = (bbox[0] + bbox[2]) / 2.0
    scored = sorted(
        media_manifest,
        key=lambda item: _distance_km(
            center_lat, center_lon,
            float(item.get("lat", center_lat)),
            float(item.get("lon", center_lon)),
        ),
    )
    return scored[:fallback_count]


def _point_in_bbox(lat: float, lon: float, bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
