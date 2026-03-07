from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ChatRecord:
    timestamp: str
    question: str
    answer: str
    citations: list[str]


class RecordStore(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_zones(self, zones: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_zones(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_zone(self, zone_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def upsert_events(self, events: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_events(self, zone_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def append_chat(self, record: ChatRecord) -> None:
        raise NotImplementedError


class LocalRecordStore(RecordStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._save({"zones": [], "events": [], "chats": []})

    def _load(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"zones": [], "events": [], "chats": []}
        return json.loads(self.db_path.read_text())

    def _save(self, payload: dict[str, Any]) -> None:
        self.db_path.write_text(json.dumps(payload, indent=2))

    def reset(self) -> None:
        self._save({"zones": [], "events": [], "chats": []})

    def upsert_zones(self, zones: list[dict[str, Any]]) -> None:
        payload = self._load()
        by_id = {zone["zone_id"]: zone for zone in payload["zones"]}
        for zone in zones:
            by_id[zone["zone_id"]] = zone
        payload["zones"] = sorted(
            by_id.values(), key=lambda zone: zone["drop_score"], reverse=True
        )
        self._save(payload)

    def list_zones(self) -> list[dict[str, Any]]:
        payload = self._load()
        return sorted(payload["zones"], key=lambda zone: zone["drop_score"], reverse=True)

    def get_zone(self, zone_id: str) -> dict[str, Any] | None:
        for zone in self._load()["zones"]:
            if zone["zone_id"] == zone_id:
                return zone
        return None

    def upsert_events(self, events: list[dict[str, Any]]) -> None:
        payload = self._load()
        by_id = {event["event_id"]: event for event in payload["events"]}
        for event in events:
            by_id[event["event_id"]] = event
        payload["events"] = sorted(
            by_id.values(), key=lambda event: event["likelihood"], reverse=True
        )
        self._save(payload)

    def list_events(self, zone_id: str | None = None) -> list[dict[str, Any]]:
        events = self._load()["events"]
        if zone_id:
            events = [event for event in events if event["zone_id"] == zone_id]
        return sorted(events, key=lambda event: event["likelihood"], reverse=True)

    def append_chat(self, record: ChatRecord) -> None:
        payload = self._load()
        payload["chats"].append(
            {
                "timestamp": record.timestamp,
                "question": record.question,
                "answer": record.answer,
                "citations": record.citations,
            }
        )
        self._save(payload)


class AwsRecordStore(RecordStore):
    def __init__(self, table_name: str, region: str) -> None:
        try:
            import boto3
            from boto3.dynamodb.conditions import Attr
        except ImportError as exc:
            raise RuntimeError("boto3 is required for AWS mode") from exc

        self._boto3 = boto3
        self._Attr = Attr
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def reset(self) -> None:
        logger.warning("AwsRecordStore.reset() is a no-op to avoid destructive deletes")

    def upsert_zones(self, zones: list[dict[str, Any]]) -> None:
        for zone in zones:
            self.table.put_item(
                Item={
                    "pk": f"ZONE#{zone['zone_id']}",
                    "sk": "META",
                    "entity": "zone",
                    **zone,
                }
            )

    def list_zones(self) -> list[dict[str, Any]]:
        result = self.table.scan(FilterExpression=self._Attr("entity").eq("zone"))
        zones = [self._strip_meta(item) for item in result.get("Items", [])]
        return sorted(zones, key=lambda zone: zone["drop_score"], reverse=True)

    def get_zone(self, zone_id: str) -> dict[str, Any] | None:
        result = self.table.get_item(Key={"pk": f"ZONE#{zone_id}", "sk": "META"})
        item = result.get("Item")
        return self._strip_meta(item) if item else None

    def upsert_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.table.put_item(
                Item={
                    "pk": f"ZONE#{event['zone_id']}",
                    "sk": f"EVENT#{event['event_id']}",
                    "entity": "event",
                    **event,
                }
            )

    def list_events(self, zone_id: str | None = None) -> list[dict[str, Any]]:
        if zone_id:
            response = self.table.query(
                KeyConditionExpression="pk = :pk AND begins_with(sk, :event_prefix)",
                ExpressionAttributeValues={
                    ":pk": f"ZONE#{zone_id}",
                    ":event_prefix": "EVENT#",
                },
            )
            items = response.get("Items", [])
        else:
            response = self.table.scan(FilterExpression=self._Attr("entity").eq("event"))
            items = response.get("Items", [])
        events = [self._strip_meta(item) for item in items]
        return sorted(events, key=lambda event: event["likelihood"], reverse=True)

    def append_chat(self, record: ChatRecord) -> None:
        chat_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self.table.put_item(
            Item={
                "pk": "CHAT",
                "sk": chat_id,
                "entity": "chat",
                "timestamp": record.timestamp,
                "question": record.question,
                "answer": record.answer,
                "citations": record.citations,
            }
        )

    @staticmethod
    def _strip_meta(item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        payload.pop("pk", None)
        payload.pop("sk", None)
        payload.pop("entity", None)
        return payload


class ObjectStore(ABC):
    @abstractmethod
    def read_json(self, path_or_uri: str) -> dict[str, Any]:
        raise NotImplementedError


class LocalObjectStore(ObjectStore):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def read_json(self, path_or_uri: str) -> dict[str, Any]:
        path = self._resolve(path_or_uri)
        return json.loads(path.read_text())

    def _resolve(self, path_or_uri: str) -> Path:
        if path_or_uri.startswith("s3://"):
            _, _, remainder = path_or_uri.partition("s3://")
            _, _, key = remainder.partition("/")
            return self.root_dir / key
        return Path(path_or_uri)


class AwsObjectStore(ObjectStore):
    def __init__(self, bucket: str, region: str) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for AWS mode") from exc
        self.client = boto3.client("s3", region_name=region)
        self.bucket = bucket

    def read_json(self, path_or_uri: str) -> dict[str, Any]:
        bucket = self.bucket
        key = path_or_uri
        if path_or_uri.startswith("s3://"):
            _, _, remainder = path_or_uri.partition("s3://")
            parsed_bucket, _, parsed_key = remainder.partition("/")
            bucket = parsed_bucket
            key = parsed_key

        response = self.client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
