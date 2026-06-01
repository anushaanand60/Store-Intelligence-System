import json
import logging
import os
from fastapi import FastAPI
import time
from datetime import datetime
import pandas as pd
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

app = FastAPI()
app.state.redis = InMemoryRedis()
app.state.transactions_by_hour = load_transactions(["/app/data/transactions.csv"])

@app.get("/health")
async def health():
    t0 = time.perf_counter_ns()
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
        "hourly_staff_counts": 0,
        "store_conversion_rate_percentage": 0.0,
        "mock_hour_invoices_loaded": invoices
    }