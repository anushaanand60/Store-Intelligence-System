from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import sys
import redis
import redis.asyncio as aioredis
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core_logic import ProductionStateEngine, state_engine
from models import InboundEventModel


class InMemoryRedis:
    def __init__(self) -> None:
        self.kv: Dict[str, str] = {}
        self.streams: Dict[str, List[Dict[str, Any]]] = {}

    async def ping(self) -> bool:
        return True

    async def set(self, key: str, value: Any) -> None:
        self.kv[key] = str(value)

    async def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    async def xadd(self, stream: str, payload: Dict[str, Any], maxlen: int = 5000) -> str:
        self.streams.setdefault(stream, [])
        message_id = f"{int(time.time() * 1000)}-{len(self.streams[stream])}"
        self.streams[stream].append({"id": message_id, "payload": payload})
        if len(self.streams[stream]) > maxlen:
            self.streams[stream] = self.streams[stream][-maxlen:]
        return message_id

    async def xread(self, streams: Dict[str, str], block: int = 1000, count: int = 100):
        result = []
        for stream_name in streams:
            items = self.streams.get(stream_name, [])
            if not items:
                continue
            batch = items[:count]
            result.append((stream_name, [(item["id"], item["payload"]) for item in batch]))
            self.streams[stream_name] = items[count:]
        if not result:
            await asyncio.sleep(block / 1000.0)
        return result

    async def aclose(self) -> None:
        return None


def _load_transactions(path: Optional[str]) -> List[datetime]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    columns = {column.lower().strip(): column for column in df.columns}
    if "order_date" not in columns or "order_time" not in columns:
        return []
    result: List[datetime] = []
    for _, row in df.iterrows():
        combined = f"{row[columns['order_date']]} {row[columns['order_time']]}"
        ts = pd.to_datetime(combined, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            continue
        native = ts.to_pydatetime()
        if native.tzinfo is None:
            native = native.replace(tzinfo=timezone.utc)
        result.append(native.astimezone(timezone.utc))
    return result


def _coerce_ingest_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        return [payload]
    return []


async def _publish_events(redis_client: Any, events: List[Dict[str, Any]]) -> None:
    for event in events:
        await redis_client.xadd("store:intelligence:events", event, maxlen=5000)


async def _consume_stream(app: FastAPI) -> None:
    redis_client = app.state.redis
    last_id = "0-0"
    print("[Consumer] State-sync stream bridge attached to message broker.", flush=True)

    while True:
        try:
            batches = await redis_client.xread({"store:intelligence:events": last_id}, block=1000, count=100)
            if not batches:
                continue

            for _, messages in batches:
                parsed_records = []
                for msg_id, raw_payload in messages:
                    try:
                        clean_record = {}
                        for k, v in raw_payload.items():
                            if k == "metadata" and isinstance(v, str) and v:
                                clean_record[k] = json.loads(v)
                            elif k == "is_staff" and isinstance(v, str):
                                clean_record[k] = v.lower() == "true"
                            elif k == "dwell_ms" and isinstance(v, str) and v:
                                clean_record[k] = int(v)
                            elif k == "confidence" and isinstance(v, str) and v:
                                clean_record[k] = float(v)
                            else:
                                clean_record[k] = v
                        parsed_records.append(clean_record)
                    except Exception:
                        continue

                if parsed_records:
                    app.state.engine.ingest_events(parsed_records)
                if messages:
                    last_id = messages[-1][0]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await asyncio.sleep(1.0)

@asynccontextmanager


async def lifespan(app: FastAPI):
    use_in_memory = os.getenv("INTEGRATION_TESTING", "false").lower() == "true"

    if use_in_memory:
        redis_client = InMemoryRedis()
    else:
        redis_client = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "redis_bus"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )

    app.state.redis = redis_client

    app.state.engine = ProductionStateEngine()
    txn_path = os.getenv("TXN_PATH", "/app/data/POS_transactions.csv")
    app.state.engine.load_transactions(txn_path)

    app.state.stream_task = None
    if not use_in_memory:
        app.state.stream_task = asyncio.create_task(_consume_stream(app))

    yield

    if app.state.stream_task:
        app.state.stream_task.cancel()
        await asyncio.gather(app.state.stream_task, return_exceptions=True)
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")


async def structured_telemetry(request: Request, call_next):
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    request.state.event_count = getattr(request.state, "event_count", 0)
    request.state.store_id = getattr(request.state, "store_id", "global")
    start = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(response.status_code)
        return response
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        print(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "store_id": str(getattr(request.state, "store_id", "global")),
                    "endpoint": request.url.path,
                    "latency_ms": float(latency_ms),
                    "event_count": int(getattr(request.state, "event_count", 0)),
                    "status_code": int(status_code),
                }
            ),
            flush=True,
        )


def _structured_error(status_code: int, code: str, message: str, trace_id: Optional[str] = None):
    payload = {"error": code, "message": message}
    if trace_id:
        payload["trace_id"] = trace_id
    return JSONResponse(status_code=status_code, content=payload)

@app.exception_handler(RequestValidationError)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _structured_error(400, "VALIDATION_ERROR", "Request validation failed.")

@app.exception_handler(redis.exceptions.RedisError)


async def redis_error_handler(request: Request, exc: redis.exceptions.RedisError):
    return _structured_error(503, "REDIS_UNAVAILABLE", "Redis connectivity unavailable.")

@app.exception_handler(ConnectionError)


async def connection_error_handler(request: Request, exc: ConnectionError):
    return _structured_error(503, "SERVICE_UNAVAILABLE", "Connection fault detected.")

@app.exception_handler(Exception)


async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (redis.exceptions.RedisError, OSError)):
        return _structured_error(503, "SERVICE_UNAVAILABLE", "Backend service unavailable.")
    return _structured_error(500, "INTERNAL_ERROR", "Unexpected server error.")

@app.get("/health")


async def health(request: Request):
    try:
        await app.state.redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="redis_unavailable") from exc

    status = app.state.engine.health()
    status["redis_connected"] = True
    return status

@app.post("/events/ingest")


async def ingest_events(request: Request, payload: Any = Body(...)):
    records = _coerce_ingest_records(payload)
    if len(records) > 500:
        raise HTTPException(status_code=400, detail="batch_size_exceeds_500")
    request.state.event_count = len(records)
    if records:
        request.state.store_id = str(records[0].get("store_id", "mixed"))
    accepted_records: List[Dict[str, Any]] = []
    error_log: List[Dict[str, Any]] = []
    partial_success_matrix: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        try:
            event = InboundEventModel.model_validate(record)
            normalized = event.model_dump()
            if normalized["metadata"].get("session_seq") is None:
                normalized["metadata"]["session_seq"] = index + 1
            accepted_records.append(normalized)
            partial_success_matrix.append({"index": index, "event_id": normalized["event_id"], "accepted": True})
        except (ValidationError, ValueError, TypeError) as exc:
            error_log.append({"index": index, "error": str(exc)})
            partial_success_matrix.append({"index": index, "accepted": False, "error": str(exc)})

    summary = app.state.engine.ingest_events(accepted_records)
    await _publish_events(app.state.redis, accepted_records)
    response = {
        "accepted": summary["accepted"],
        "duplicate_count": summary["duplicate_count"],
        "partial_success": partial_success_matrix,
        "error_log": error_log,
    }
    response["status"] = "partial_success" if error_log or summary["duplicate_count"] else "success"
    return response

@app.get("/stores/{store_id}/metrics")


async def store_metrics(request: Request, store_id: str):
    request.state.store_id = store_id
    return app.state.engine.snapshot_metrics(store_id)

@app.get("/stores/{store_id}/funnel")


async def store_funnel(request: Request, store_id: str):
    request.state.store_id = store_id
    return app.state.engine.funnel(store_id)

@app.get("/stores/{store_id}/heatmap")


async def store_heatmap(request: Request, store_id: str):
    request.state.store_id = store_id
    return app.state.engine.heatmap(store_id)

@app.get("/stores/{store_id}/anomalies")


async def store_anomalies(request: Request, store_id: str):
    request.state.store_id = store_id
    return {"store_id": store_id, "active_anomalies": app.state.engine.anomalies(store_id)}
