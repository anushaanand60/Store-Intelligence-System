import os
from uuid import uuid4

from fastapi.testclient import TestClient

os.environ["INTEGRATION_TESTING"] = "true"

from app.main import app


def _seed_events(client: TestClient):
    payload = [
        {
            "event_id": str(uuid4()),
            "store_id": "STORE_1",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_A",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:22:10Z",
            "zone_id": "ENTRY",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.91,
            "metadata": {"queue_depth": 1, "sku_zone": "MOISTURISER", "session_seq": 1},
        },
        {
            "event_id": str(uuid4()),
            "store_id": "STORE_1",
            "camera_id": "CAM_ZONE_01",
            "visitor_id": "VIS_A",
            "event_type": "ZONE_ENTER",
            "timestamp": "2026-03-03T14:22:40Z",
            "zone_id": "SKINCARE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"queue_depth": 1, "sku_zone": "MOISTURISER", "session_seq": 1},
        },
        {
            "event_id": str(uuid4()),
            "store_id": "STORE_1",
            "camera_id": "CAM_ZONE_01",
            "visitor_id": "VIS_A",
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": "2026-03-03T14:23:40Z",
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.95,
            "metadata": {"queue_depth": 2, "sku_zone": "MOISTURISER", "session_seq": 1},
        },
    ]
    response = client.post("/events/ingest", json=payload)
    assert response.status_code == 200


def test_metrics_endpoint_returns_requested_fields():
    with TestClient(app) as client:
        _seed_events(client)
        response = client.get("/stores/STORE_1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "unique_visitors",
            "live_occupancy",
            "total_entries",
            "total_exits",
            "hourly_staff_counts",
            "store_conversion_rate_percentage",
            "queue_depth",
            "abandonment_rate",
        }
        assert body["unique_visitors"] >= 1
        assert isinstance(body["hourly_staff_counts"], int)
        assert body["queue_depth"] >= 0


def test_funnel_and_heatmap_contracts():
    with TestClient(app) as client:
        _seed_events(client)
        funnel = client.get("/stores/STORE_1/funnel")
        heatmap = client.get("/stores/STORE_1/heatmap")

        assert funnel.status_code == 200
        funnel_body = funnel.json()
        assert isinstance(funnel_body, list)
        assert len(funnel_body) == 4
        assert funnel_body[0] == {"stage": "Entry", "count": funnel_body[0]["count"], "drop_off_percentage": 0.0}
        assert [stage["stage"] for stage in funnel_body] == ["Entry", "Zone Visit", "Billing Queue", "Purchase"]

        assert heatmap.status_code == 200
        heatmap_body = heatmap.json()
        assert "zones" in heatmap_body
        assert "visit_frequency_score" in heatmap_body
        assert isinstance(heatmap_body["data_confidence"], bool)


def test_soft_identity_matching_merges_camera_aliases():
    with TestClient(app) as client:
        payload = [
            {
                "event_id": str(uuid4()),
                "store_id": "STORE_1",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_E_1_000001",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": "ENTRY",
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.91,
                "metadata": {"queue_depth": 1, "sku_zone": "ENTRY", "session_seq": 1},
            },
            {
                "event_id": str(uuid4()),
                "store_id": "STORE_1",
                "camera_id": "CAM_ZONE_01",
                "visitor_id": "VIS_Z_9_000001",
                "event_type": "ZONE_ENTER",
                "timestamp": "2026-03-03T14:27:10Z",
                "zone_id": "SKINCARE",
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.92,
                "metadata": {"queue_depth": 1, "sku_zone": "SKINCARE", "session_seq": 1},
            },
        ]
        response = client.post("/events/ingest", json=payload)
        assert response.status_code == 200

        metrics = client.get("/stores/STORE_1/metrics")
        assert metrics.status_code == 200
        body = metrics.json()
        assert body["unique_visitors"] == 1
        assert body["total_entries"] == 1
