# Store Intelligence System

## Overview

Store Intelligence System is a containerized retail analytics stack that discovers camera videos from `data/Store 1` and `data/Store 2`, processes them through a Redis-backed pipeline, and exposes store-scoped analytics from a FastAPI application.

## Core Components

- `redis_bus`: shared event broker
- `central_api`: FastAPI service under `app/main.py`
- `pipeline_worker`: video discovery and event emission under `pipeline/detect.py`

## Data Layout

The repository now expects all runtime assets to live under `data/`.

- `data/Store 1/CAM 1 - zone.mp4`
- `data/Store 1/CAM 2 - zone.mp4`
- `data/Store 1/CAM 3 - entry.mp4`
- `data/Store 1/CAM 5 - billing.mp4`
- `data/Store 2/billing_area.mp4`
- `data/Store 2/entry 1.mp4`
- `data/Store 2/entry 2.mp4`
- `data/Store 2/zone.mp4`
- `data/POS - sample transactionsb1e826f.csv`
- `data/sample_eventsbe42122.jsonl`

## Start

```bash
docker compose up --build
```

## Live Dashboard Verification

```bash
python dashboard_render.py
```

## API

- `POST /events/ingest`
- `GET /stores/{id}/metrics`
- `GET /stores/{id}/funnel`
- `GET /stores/{id}/heatmap`
- `GET /stores/{id}/anomalies`
- `GET /health`

## Notes

The refactor keeps the spatial hash tracker and 45-frame ghost cache intact, but moves the application into the requested root layout:

- `app/`
- `pipeline/`
- `tests/`
- `docs/`
