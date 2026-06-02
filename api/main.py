import json
import logging
import os
from fastapi import FastAPI, HTTPException
import time
from contextlib import asynccontextmanager
from datetime import datetime
import pandas as pd
import asyncio
import redis.asyncio as aioredis

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

    async def xadd(self, stream, payload, maxlen=1000):
        self.streams.setdefault(stream, [])
        self.streams[stream].append((str(time.time()), payload))
        if len(self.streams[stream]) > maxlen:
            self.streams[stream] = self.streams[stream][-maxlen:]

    async def xread(self, streams, block=1000, count=10):
        result = []
        for stream in streams:
            items = self.streams.get(stream, [])[:count]
            if items:
                result.append((stream, items))
                self.streams[stream] = self.streams[stream][count:]
        if not result:
            await asyncio.sleep(block / 1000.0)
        return result

    async def aclose(self):
        return None

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

async def behavior_worker(app):
    redis_client = app.state.redis
    engine = app.state.behavior_engine
    stream_keys = {f"camera_stream:CAM_{i}": "$" for i in range(1, 6)}
    while True:
        try:
            events = await redis_client.xread(stream_keys, block=1000, count=128)
            for stream, messages in events:
                camera_id = stream.split(":")[-1]
                active = []
                for _, payload in messages:
                    tracks = json.loads(payload.get("tracks", "[]"))
                    ts = float(payload.get("timestamp", time.time()))
                    for t in tracks:
                        tid = int(t["track_id"])
                        active.append(tid)
                        engine.process_track_frame(camera_id, tid, t["bbox"], ts)
                engine.handle_dropped_tracks(active, camera_id)
                snap = engine.get_metrics_snapshot()
                for k, v in snap.items():
                    await redis_client.set(f"store:{k}", v)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info(json.dumps({"event": "behavior_worker_error", "error": str(exc)}))
            await asyncio.sleep(1.0)

async def anomaly_worker(app):
    redis_client = app.state.redis
    detector = app.state.anomaly_detector
    while True:
        try:
            entries = int(await redis_client.get("store:total_entries") or 0)
            detector.check_stream_velocity(entries)
            occupancy = int(await redis_client.get("store:live_occupancy") or 0)
            if occupancy < 0:
                await redis_client.set("store:live_occupancy", 0)
                logger.warning(json.dumps({"event": "occupancy_underflow_corrected", "value": occupancy}))
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info(json.dumps({"event": "anomaly_worker_error", "error": str(exc)}))
            await asyncio.sleep(1.0)

@asynccontextmanager
async def lifespan(app):
    use_test_backend = os.getenv("INTEGRATION_TESTING", "false").lower() == "true"
    if use_test_backend:
        redis_client = InMemoryRedis()
    else:
        redis_client = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "redis_bus"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )
        await redis_client.ping()
    app.state.redis = redis_client
    app.state.zone_map = load_zone_map(os.getenv("LAYOUT_PATH", "/app/data/store_layout.xlsx"))
    app.state.transactions_by_hour = load_transactions([
            os.getenv("TXN_PATH", "/app/data/transactions.csv"),
            "/app/data/Brigade_Bangalore_10_April_26 (1)bc6219c.csv",
    ])
    app.state.behavior_engine = StoreIntelligenceBehaviorEngine(app.state.zone_map)
    app.state.anomaly_detector = StoreIntelligenceAnomalyDetector()
    await redis_client.set("store:live_occupancy", 0)
    await redis_client.set("store:total_entries", 0)
    await redis_client.set("store:total_exits", 0)
    await redis_client.set("store:hourly_staff_counts", 0)

    app.state.tasks = [
        asyncio.create_task(behavior_worker(app)),
        asyncio.create_task(anomaly_worker(app)),
    ]
    yield
    for t in app.state.tasks:
        t.cancel()
    await asyncio.gather(*app.state.tasks, return_exceptions=True)
    await redis_client.aclose()

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
    r = app.state.redis
    live = int(await r.get("store:live_occupancy") or 0)
    entries = int(await r.get("store:total_entries") or 0)
    exits = int(await r.get("store:total_exits") or 0)
    staff = int(await r.get("store:hourly_staff_counts") or len(app.state.behavior_engine.staff_registry))
    invoices = int(app.state.transactions_by_hour.get(now_hour, 0))
    denom = max(entries, 1)
    conversion = round((invoices / denom) * 100.0, 2)
    return {
        "live_occupancy": max(live, 0),
        "total_entries": entries,
        "total_exits": exits,
        "hourly_staff_counts": staff,
        "store_conversion_rate_percentage": conversion
    }

@app.get("/funnel")
async def funnel():
    payload = app.state.behavior_engine.get_funnel_snapshot()
    if not payload:
        raise HTTPException(status_code=500, detail="funnel_state_unavailable")
    return payload