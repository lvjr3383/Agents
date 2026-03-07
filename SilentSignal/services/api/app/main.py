from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agents.drone_analyst_agent import DroneAnalystAgent
from agents.silence_mapper_agent import SilenceMapperAgent
from data.generate_synthetic_data import generate_synthetic_dataset

from .chat import build_chat_answer
from .config import Settings, get_settings
from .dependencies import get_object_store, get_record_store
from .models import (
    AnalyzeZoneResponse,
    ChatRequest,
    ChatResponse,
    GenerateDataResponse,
    RunAnalysisResponse,
    SilenceZone,
)
from .storage import ChatRecord, RecordStore, now_utc_iso


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.generated_dir.mkdir(parents=True, exist_ok=True)
    settings.local_s3_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="SilentSignal API", version="0.1.0", lifespan=lifespan)


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/dashboard",
    StaticFiles(directory=str(settings.dashboard_dir), html=True),
    name="dashboard",
)


@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "SilentSignal API"}


@app.post("/reset")
def reset_app(store: RecordStore = Depends(get_record_store)) -> dict[str, str]:
    store.reset()
    return {"status": "reset", "message": "All zones, events, and chat history cleared."}


@app.post("/generate-data", response_model=GenerateDataResponse)
def generate_data(
    settings: Settings = Depends(get_settings),
) -> GenerateDataResponse:
    summary = generate_synthetic_dataset(
        output_csv=settings.signals_csv_path,
        media_manifest_path=settings.media_manifest_path,
        media_root=settings.local_s3_dir,
        seed=42,
    )
    return GenerateDataResponse(
        signals_csv=str(settings.signals_csv_path),
        media_manifest=str(settings.media_manifest_path),
        rows_written=summary["rows_written"],
        media_assets=summary["media_assets"],
    )


@app.post("/run-analysis", response_model=RunAnalysisResponse)
def run_analysis(
    reset_store: bool = Query(default=True),
    settings: Settings = Depends(get_settings),
    store: RecordStore = Depends(get_record_store),
) -> RunAnalysisResponse:
    if not settings.signals_csv_path.exists():
        generate_data(settings)

    silence_mapper = SilenceMapperAgent()
    zones = silence_mapper.analyze(settings.signals_csv_path)

    if reset_store:
        store.reset()

    store.upsert_zones(zones)
    parsed = [SilenceZone(**zone) for zone in zones]
    return RunAnalysisResponse(zones_created=len(parsed), zones=parsed)


@app.get("/zones", response_model=list[SilenceZone])
def list_zones(store: RecordStore = Depends(get_record_store)) -> list[SilenceZone]:
    return [SilenceZone(**zone) for zone in store.list_zones()]


@app.get("/zones/{zone_id}")
def get_zone_detail(
    zone_id: str,
    store: RecordStore = Depends(get_record_store),
) -> dict[str, Any]:
    zone = store.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")
    events = store.list_events(zone_id)
    return {"zone": zone, "events": events}


@app.get("/zones/{zone_id}/events")
def get_zone_events(
    zone_id: str,
    store: RecordStore = Depends(get_record_store),
) -> list[dict[str, Any]]:
    return store.list_events(zone_id)


@app.post("/zones/{zone_id}/analyze-drone", response_model=AnalyzeZoneResponse)
def analyze_zone(
    zone_id: str,
    settings: Settings = Depends(get_settings),
    store: RecordStore = Depends(get_record_store),
) -> AnalyzeZoneResponse:
    zone = store.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    media_manifest = _read_media_manifest(settings.media_manifest_path)
    media_for_zone = _select_media_for_zone(zone, media_manifest)
    if not media_for_zone:
        raise HTTPException(status_code=404, detail="No media found for selected zone")

    event_seed = len(store.list_events()) + 1
    drone_analyst = DroneAnalystAgent(
        aws_mode=settings.aws_mode,
        region=settings.aws_region,
        model_id=settings.bedrock_model_id,
    )
    events = drone_analyst.analyze_zone_media(
        zone=zone,
        media_records=media_for_zone,
        event_seed=event_seed,
    )

    store.upsert_events(events)
    return AnalyzeZoneResponse(
        events_created=len(events),
        events=[event for event in events],
    )


@app.post("/chat", response_model=ChatResponse)
def chat_with_data(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: RecordStore = Depends(get_record_store),
) -> ChatResponse:
    zones = store.list_zones()
    events = store.list_events()
    answer, citations, actions = build_chat_answer(
        request.question,
        zones,
        events,
        aws_mode=settings.aws_mode,
        region=settings.aws_region,
        model_id=settings.bedrock_model_id,
        store=store,
        settings=settings,
    )
    store.append_chat(
        ChatRecord(
            timestamp=now_utc_iso(),
            question=request.question,
            answer=answer,
            citations=citations,
        )
    )
    return ChatResponse(answer=answer, citations=citations, actions=actions)


@app.get("/chat/history")
def get_chat_history(
    store: RecordStore = Depends(get_record_store),
) -> dict[str, Any]:
    # Chat history is currently only persisted in local mode JSON.
    if hasattr(store, "_load"):
        payload = store._load()  # type: ignore[attr-defined]
        return {"items": payload.get("chats", [])}
    return {"items": []}


# ---------------------------- helpers ----------------------------


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

    matches = []
    for item in media_manifest:
        lat = float(item.get("lat", 0.0))
        lon = float(item.get("lon", 0.0))
        if _point_in_bbox(lat, lon, bbox):
            matches.append(item)

    if matches:
        return matches

    center_lat = (bbox[1] + bbox[3]) / 2.0
    center_lon = (bbox[0] + bbox[2]) / 2.0

    scored = sorted(
        media_manifest,
        key=lambda item: _distance_km(
            center_lat,
            center_lon,
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
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c
