from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SilenceZone(BaseModel):
    zone_id: str
    geometry: dict[str, Any]
    start_time: str
    end_time: str
    drop_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str]


class DroneEvent(BaseModel):
    event_id: str
    zone_id: str
    media_uri: str
    event_type: Literal["SHOUT", "TAP_PATTERN", "HUMAN_FORM", "UNKNOWN"]
    time_offset_start: float = Field(ge=0)
    time_offset_end: float = Field(ge=0)
    likelihood: float = Field(ge=0, le=1)
    notes: str


class GenerateDataResponse(BaseModel):
    signals_csv: str
    media_manifest: str
    rows_written: int
    media_assets: int


class RunAnalysisResponse(BaseModel):
    zones_created: int
    zones: list[SilenceZone]


class AnalyzeZoneResponse(BaseModel):
    events_created: int
    events: list[DroneEvent]


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    actions: list[str] = []
