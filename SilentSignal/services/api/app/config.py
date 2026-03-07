from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _parse_origins(raw: str) -> list[str]:
    if raw.strip() == "*":
        return ["*"]
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    generated_dir: Path
    local_s3_dir: Path
    local_dynamo_path: Path
    signals_csv_path: Path
    media_manifest_path: Path
    dashboard_dir: Path
    aws_mode: str
    aws_region: str
    s3_bucket: str
    dynamodb_table: str
    bedrock_model_id: str
    api_host: str
    api_port: int
    cors_origins: list[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    aws_mode = os.getenv("AWS_MODE", "mock").lower().strip()
    data_dir = PROJECT_ROOT / "data"

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        generated_dir=data_dir / "generated",
        local_s3_dir=PROJECT_ROOT / os.getenv("LOCAL_S3_DIR", "data/s3_mock"),
        local_dynamo_path=PROJECT_ROOT
        / os.getenv("LOCAL_DYNAMO_PATH", "data/generated/mock_dynamo.json"),
        signals_csv_path=PROJECT_ROOT
        / os.getenv("SIGNALS_CSV_PATH", "data/generated/signals.csv"),
        media_manifest_path=PROJECT_ROOT
        / os.getenv("MEDIA_MANIFEST_PATH", "data/generated/media_manifest.json"),
        dashboard_dir=PROJECT_ROOT / "apps" / "dashboard",
        aws_mode=aws_mode if aws_mode in {"mock", "aws", "auto"} else "mock",
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        s3_bucket=os.getenv("S3_BUCKET", "silent-signal-media"),
        dynamodb_table=os.getenv("DYNAMODB_TABLE", "silent-signal-mvp"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        cors_origins=_parse_origins(os.getenv("CORS_ORIGINS", "*")),
    )
