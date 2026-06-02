import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTEGRATION_TESTING"] = "true"

from main import app

def _seed_engine():
    engine = app.state.behavior_engine
    ts = 1_700_000_000.0
    engine.process_track_frame("CAM_3", 1, [200.0, 200.0, 320.0, 480.0], ts)
    engine.process_track_frame("CAM_1", 1, [240.0, 220.0, 340.0, 500.0], ts + 0.1)
    engine.process_track_frame("CAM_2", 1, [280.0, 260.0, 380.0, 520.0], ts + 0.2)
    engine.process_track_frame("CAM_5", 1, [320.0, 280.0, 420.0, 560.0], ts + 0.3)

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["redis"] == "connected"
        assert "loop_latency_ms" in body

def test_metrics_schema_and_types():
    with TestClient(app) as client:
        _seed_engine()
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "live_occupancy",
            "total_entries",
            "total_exits",
            "hourly_staff_counts",
            "store_conversion_rate_percentage",
        }
        assert isinstance(body["live_occupancy"], int)
        assert isinstance(body["total_entries"], int)
        assert isinstance(body["total_exits"], int)
        assert isinstance(body["hourly_staff_counts"], int)
        assert isinstance(body["store_conversion_rate_percentage"], float)

def test_funnel_contract():
    with TestClient(app) as client:
        _seed_engine()
        response = client.get("/funnel")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 4
        assert body[0]["zone"] == "Entrance (CAM 3)"
        assert body[1]["zone"] == "Skincare/Aisles (CAM 1/4)"
        assert body[2]["zone"] == "Cosmetics Area (CAM 2)"
        assert body[3]["zone"] == "Checkout Terminal (CAM 5)"
        drops = [stage["drop_off_percentage"] for stage in body]
        assert drops[0] <= drops[1] <= drops[2] <= drops[3]
