import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

os.environ["INTEGRATION_TESTING"] = "true"

from app.main import app


def test_anomalies_and_health_contracts():
    with TestClient(app) as client:
        now = datetime.now(timezone.utc)
        payload = []
        for idx in range(16):
            payload.append(
                {
                    "event_id": str(uuid4()),
                    "store_id": "STORE_9",
                    "camera_id": "CAM_ENTRY_01",
                    "visitor_id": f"VIS_{idx}",
                    "event_type": "ENTRY",
                    "timestamp": (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
                    "zone_id": "ENTRY",
                    "dwell_ms": 0,
                    "is_staff": False,
                    "confidence": 0.9,
                    "metadata": {"queue_depth": 0, "sku_zone": "SERUM", "session_seq": 1},
                }
            )
        client.post("/events/ingest", json=payload)
        anomalies = client.get("/stores/STORE_9/anomalies")
        health = client.get("/health")

        assert anomalies.status_code == 200
        body = anomalies.json()
        assert "store_id" in body
        assert "active_anomalies" in body
        assert any(item["anomaly_type"] in {"CONVERSION_DROP", "DEAD_ZONE"} for item in body["active_anomalies"])

        assert health.status_code == 200
        health_body = health.json()
        assert "warning_codes" in health_body
        assert isinstance(health_body["warning_codes"], list)
