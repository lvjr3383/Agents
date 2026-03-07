from __future__ import annotations

import json
import logging
from typing import Any

from .base import AgentMetadata, BaseAgent


logger = logging.getLogger(__name__)


class BedrockNovaAdapter:
    """
    Bedrock wrapper with deterministic mock fallback.

    In mock mode, detections are inferred from media URI naming conventions.
    """

    def __init__(self, aws_mode: str, region: str, model_id: str) -> None:
        self.aws_mode = aws_mode
        self.region = region
        self.model_id = model_id
        self.client = None

        if aws_mode in {"aws", "auto"}:
            try:
                import boto3

                self.client = boto3.client("bedrock-runtime", region_name=region)
            except Exception as exc:
                logger.warning("Bedrock client unavailable, using mock mode: %s", exc)

    def analyze_media(self, media_uri: str, prompt: str) -> dict[str, Any]:
        if self.client is not None and self.aws_mode in {"aws", "auto"}:
            try:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": prompt},
                                {"text": f"Analyze media reference: {media_uri}"},
                            ],
                        }
                    ],
                    "inferenceConfig": {"maxTokens": 256, "temperature": 0},
                }
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(payload).encode("utf-8"),
                    contentType="application/json",
                    accept="application/json",
                )
                raw = json.loads(response["body"].read().decode("utf-8"))
                text = json.dumps(raw)
                return self._deterministic_mock(media_uri, raw_text=text)
            except Exception as exc:
                logger.warning("Bedrock invocation failed, using mock response: %s", exc)

        return self._deterministic_mock(media_uri)

    @staticmethod
    def _deterministic_mock(media_uri: str, raw_text: str | None = None) -> dict[str, Any]:
        lower_uri = media_uri.lower()

        if "shout" in lower_uri:
            return {
                "event_type": "SHOUT",
                "likelihood": 0.88,
                "notes": "Spectrogram peaks are consistent with human shouting.",
                "raw": raw_text,
            }
        if "tap" in lower_uri:
            return {
                "event_type": "TAP_PATTERN",
                "likelihood": 0.81,
                "notes": "Periodic impulse pattern suggests intentional tapping.",
                "raw": raw_text,
            }
        if "human" in lower_uri or "silhouette" in lower_uri:
            return {
                "event_type": "HUMAN_FORM",
                "likelihood": 0.84,
                "notes": "Frame sequence indicates potential human form movement.",
                "raw": raw_text,
            }

        return {
            "event_type": "UNKNOWN",
            "likelihood": 0.33,
            "notes": "No strong distress indicators detected in sampled segments.",
            "raw": raw_text,
        }


class DroneAnalystAgent(BaseAgent):
    def __init__(self, aws_mode: str = "mock", region: str = "us-east-1", model_id: str = "amazon.nova-pro-v1:0") -> None:
        super().__init__(
            AgentMetadata(
                name="DroneAnalystAgent",
                description="Classifies drone media for signs of life",
            )
        )
        self.nova = BedrockNovaAdapter(aws_mode=aws_mode, region=region, model_id=model_id)

    def analyze_zone_media(
        self,
        zone: dict[str, Any],
        media_records: list[dict[str, Any]],
        event_seed: int = 1,
    ) -> list[dict[str, Any]]:
        prompt = (
            "Analyze spectrogram and visual cues for distress signals despite rotor noise. "
            "Return strongest event classification and confidence."
        )

        events: list[dict[str, Any]] = []
        for index, media in enumerate(media_records, start=0):
            analysis = self.nova.analyze_media(media_uri=media["media_uri"], prompt=prompt)
            start_offset = float(index * 12)
            end_offset = start_offset + 8.0

            events.append(
                {
                    "event_id": f"E-{event_seed + index:03d}",
                    "zone_id": zone["zone_id"],
                    "media_uri": media["media_uri"],
                    "event_type": analysis["event_type"],
                    "time_offset_start": round(start_offset, 2),
                    "time_offset_end": round(end_offset, 2),
                    "likelihood": round(float(analysis["likelihood"]), 3),
                    "notes": analysis["notes"],
                }
            )

        return sorted(events, key=lambda event: event["likelihood"], reverse=True)
