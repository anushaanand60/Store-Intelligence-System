from fastapi import FastAPI
import time

app = FastAPI()

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
    return {
        "live_occupancy": 0,
        "total_entries": 0,
        "total_exits": 0,
        "hourly_staff_counts": 0,
        "store_conversion_rate_percentage": 0.0
    }