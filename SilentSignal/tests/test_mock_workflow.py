from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class MockWorkflowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        temp_root = Path(self.tempdir.name)

        self.env_keys = [
            "AWS_MODE",
            "LOCAL_DYNAMO_PATH",
            "LOCAL_S3_DIR",
            "SIGNALS_CSV_PATH",
            "MEDIA_MANIFEST_PATH",
        ]
        self.original_env = {key: os.environ.get(key) for key in self.env_keys}

        os.environ["AWS_MODE"] = "mock"
        os.environ["LOCAL_DYNAMO_PATH"] = str(temp_root / "mock_dynamo.json")
        os.environ["LOCAL_S3_DIR"] = str(temp_root / "s3_mock")
        os.environ["SIGNALS_CSV_PATH"] = str(temp_root / "signals.csv")
        os.environ["MEDIA_MANIFEST_PATH"] = str(temp_root / "media_manifest.json")

        # Reset memoized settings/stores so each test uses fresh temp paths.
        from services.api.app.config import get_settings
        from services.api.app.dependencies import get_object_store, get_record_store

        get_settings.cache_clear()
        get_record_store.cache_clear()
        get_object_store.cache_clear()

        import services.api.app.main as app_main

        self.app_main = importlib.reload(app_main)
        self.client = TestClient(self.app_main.app)

    def tearDown(self) -> None:
        self.client.close()
        self.tempdir.cleanup()

        from services.api.app.config import get_settings
        from services.api.app.dependencies import get_object_store, get_record_store

        get_settings.cache_clear()
        get_record_store.cache_clear()
        get_object_store.cache_clear()

        for key, original in self.original_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def test_mock_mode_end_to_end_flow(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)

        generated = self.client.post("/generate-data")
        self.assertEqual(generated.status_code, 200)
        generated_payload = generated.json()
        self.assertGreater(generated_payload["rows_written"], 0)
        self.assertGreater(generated_payload["media_assets"], 0)

        analyzed = self.client.post("/run-analysis")
        self.assertEqual(analyzed.status_code, 200)
        analyzed_payload = analyzed.json()
        self.assertGreater(analyzed_payload["zones_created"], 0)
        self.assertGreaterEqual(analyzed_payload["zones_created"], 2)

        zones = self.client.get("/zones")
        self.assertEqual(zones.status_code, 200)
        zones_payload = zones.json()
        self.assertGreater(len(zones_payload), 0)

        top_zone = zones_payload[0]
        self.assertIn("zone_id", top_zone)
        self.assertIn("drop_score", top_zone)
        self.assertIn("confidence", top_zone)

        drone = self.client.post(f"/zones/{top_zone['zone_id']}/analyze-drone")
        self.assertEqual(drone.status_code, 200)
        drone_payload = drone.json()
        self.assertGreater(drone_payload["events_created"], 0)

        event_types = {event["event_type"] for event in drone_payload["events"]}
        self.assertTrue(event_types.intersection({"SHOUT", "TAP_PATTERN", "HUMAN_FORM"}))

        chat = self.client.post(
            "/chat",
            json={"question": f"What is the top event for {top_zone['zone_id']}?"},
        )
        self.assertEqual(chat.status_code, 200)
        chat_payload = chat.json()
        self.assertIn("Citations:", chat_payload["answer"])
        self.assertTrue(
            any(citation.startswith("zone_id=") for citation in chat_payload["citations"])
        )
        self.assertTrue(
            any(citation.startswith("event_id=") for citation in chat_payload["citations"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
