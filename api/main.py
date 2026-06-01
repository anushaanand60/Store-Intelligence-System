import json
import logging
import os
from fastapi import FastAPI, HTTPException
import time
from contextlib import asynccontextmanager
from datetime import datetime
import pandas as pd

from anomaly_engine import StoreIntelligenceAnomalyDetector
from behavior_engine import StoreIntelligenceBehaviorEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("store_intelligence.api")

class InMemoryRedis:
    def __init__(self):
        self.kv = {}
        self.streams = {}

    async def ping(self):
        return True

    async def set(self, k, v):
        self.kv[k] = str(v)

    async def get(self, k):
        return self.kv.get(k)

def load_zone_map(layout_path):
    zone_map = {
        "CAM_1": [{"zone": "Skincare", "bbox": [100, 100, 1750, 950], "is_pos": False}],
        "CAM_2": [{"zone": "Cosmetics", "bbox": [100, 100, 1750, 950], "is_pos": False}],
        "CAM_3": [{"zone": "Entrance", "bbox": [100, 100, 1750, 950], "is_pos": False}],
        "CAM_4": [{"zone": "Aisles", "bbox": [100, 100, 1750, 950], "is_pos": False}],
        "CAM_5": [{"zone": "Checkout", "bbox": [100, 100, 1750, 950], "is_pos": True}],
    }
    try:
        if os.path.exists(layout_path):
            xls = pd.ExcelFile(layout_path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                cols = {c.lower().strip(): c for c in df.columns}
                required = {"camera", "zone", "x1", "y1", "x2", "y2"}
                if not required.issubset(set(cols.keys())):
                    continue
                for _, row in df.iterrows():
                    cam = str(row[cols["camera"]]).strip().upper().replace(" ", "_")
                    if cam in {"CAM1", "CAM2", "CAM3", "CAM4", "CAM5"}:
                        cam = cam.replace("CAM", "CAM_")
                    if cam not in zone_map:
                        continue
                    zone_map[cam].append({
                        "zone": str(row[cols["zone"]]).strip(),
                        "bbox": [
                            float(row[cols["x1"]]),
                            float(row[cols["y1"]]),
                            float(row[cols["x2"]]),
                            float(row[cols["y2"]]),
                        ],
                        "is_pos": "pos" in str(row[cols["zone"]]).lower() or "checkout" in str(row[cols["zone"]]).lower(),
                    })
    except Exception as exc:
        logger.info(json.dumps({"event": "layout_parse_fallback", "error": str(exc)}))
    return zone_map

def load_transactions(csv_candidates):
    for path in csv_candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            invoice_col = next((cols[c] for c in cols if "invoice" in c), None)
            time_col = next((cols[c] for c in cols if "time" in c or "date" in c), None)
            if not invoice_col or not time_col:
                continue
            ts = pd.to_datetime(df[time_col], errors="coerce")
            hourly = (
                df.assign(_hour=ts.dt.hour)
                .dropna(subset=["_hour"])
                .groupby("_hour")[invoice_col]
                .nunique()
                .to_dict()
            )
            return {int(k): int(v) for k, v in hourly.items()}
        except Exception:
            continue
    return {}

@asynccontextmanager
async def lifespan(app):
    redis_client = InMemoryRedis()
    app.state.redis = redis_client
    app.state.zone_map = load_zone_map(os.getenv("LAYOUT_PATH", "/app/data/store_layout.xlsx"))
    app.state.transactions_by_hour = load_transactions([
        "/app/data/Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
    ])
    app.state.behavior_engine = StoreIntelligenceBehaviorEngine(app.state.zone_map)
    app.state.anomaly_detector = StoreIntelligenceAnomalyDetector()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    t0 = time.perf_counter_ns()
    await app.state.redis.ping()
    latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    return {
        "status": "healthy",
        "redis": "connected_mock",
        "loop_latency_ms": round(latency_ms, 4)
    }

@app.get("/metrics")
async def metrics():
    now_hour = datetime.now().hour
    invoices = int(app.state.transactions_by_hour.get(now_hour, 0))
    return {
        "live_occupancy": 0,
        "total_entries": 0,
        "total_exits": 0,
        "hourly_staff_counts": len(app.state.behavior_engine.staff_registry),
        "store_conversion_rate_percentage": 0.0,
        "mock_hour_invoices_loaded": invoices
    }