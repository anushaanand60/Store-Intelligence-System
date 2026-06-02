# System Design

## Purpose

This platform performs real-time in-store intelligence by processing five camera feeds, publishing track data into Redis Streams, aggregating behavior and anomaly state asynchronously, and serving fused metrics through a REST API.

## Architecture Overview

The system follows a decoupled producer-consumer architecture.

* Producers: `cam1_worker` to `cam5_worker`
* Message bus: Redis Streams
* Processing layer: `behavior_worker` and `anomaly_worker`
* API layer: FastAPI service endpoints

### Runtime Topology

![Runtime Topology](docs/runtime_topology.png)

## Producer Layer

Each camera worker reads frames from its configured video source and performs human detection. Active identities are maintained through `SpatialHashTracker`, and structured tracking payloads are published into camera-specific Redis streams.

Published stream pattern:

* `camera_stream:CAM_1`
* `camera_stream:CAM_2`
* `camera_stream:CAM_3`
* `camera_stream:CAM_4`
* `camera_stream:CAM_5`

Published payload fields:

* `camera_id`
* `timestamp`
* `frame_index`
* `tracks`
* `active_tracks`

## Message Bus Contract

Redis Streams provide ordered event ingestion with bounded stream depth. Camera workers publish events using `xadd`, while downstream consumers read batches using `xread`. This separation prevents API responsiveness from being affected by camera ingestion load.

## Tracking and State Management

Track identities are maintained through `SpatialHashTracker`, which performs localized matching between detections and active tracks.

Temporarily lost tracks are stored in `ghost_tracks` with a `ttl` of 45 frames and velocity-based position carry-forward. This allows short occlusions to be handled without fragmenting shopper journeys while keeping memory usage bounded.

## Async Central Processing Layer

The central service runs two asynchronous lifecycle workers.

### 1. Behavior Worker (`behavior_worker`)

Responsibilities:

* Consume camera streams in batches
* Resolve zone assignments
* Track stage progression
* Maintain occupancy state
* Detect entries and exits
* Update funnel progression
* Write fused counters into Redis

Examples of maintained state:

* `store:live_occupancy`
* `store:total_entries`
* `store:total_exits`

### 2. Anomaly Worker (`anomaly_worker`)

Responsibilities:

* Read fused counters
* Monitor rolling activity patterns
* Detect abnormal spikes or drops
* Emit warning events when thresholds are exceeded

This separation keeps ingestion, analytics, anomaly detection, and API serving independent while sharing a single service lifecycle.

## Data Sources and Startup Indexing

The service loads local datasets during startup:

* Store layout spreadsheet
* Transaction CSV

The layout file is processed through `load_zone_map`, while transaction data is processed through `load_transactions`.

Startup indexing populates:

* `app.state.zone_map`
* `app.state.transactions_by_hour`

Performing these operations once at startup avoids repeated file parsing and transformation during request handling.

## Zone Topology and Camera Mapping

Zone definitions are loaded from the store layout spreadsheet and mapped to camera-specific regions.

Primary camera-stage mapping:

* `CAM_3` → Entrance
* `CAM_1` and `CAM_4` → Skincare and Aisles
* `CAM_2` → Cosmetics
* `CAM_5` → Checkout

Checkout-related zones are additionally used to distinguish persistent checkout-presence tracks from customers where applicable.

### Zone Resolution Flow

![Zone Resolution Flow](docs/zone_resolution_flow.png)

## FastAPI API Layer

The FastAPI service exposes reviewer-facing endpoints.

### `GET /health`

Purpose:

* Verify service liveness
* Verify Redis connectivity
* Report event-loop responsiveness

### `GET /metrics`

Purpose:

* Return live occupancy metrics
* Return entry and exit counts
* Return conversion statistics
* Return hourly transaction context

Primary data sources:

* `store:live_occupancy`
* `store:total_entries`
* `store:total_exits`
* `get_metrics_snapshot()`

### `GET /funnel`

Purpose:

* Return stage-wise funnel progression
* Return drop-off percentages
* Return conversion performance across store stages

Primary data sources:

* `get_funnel_snapshot()`
* Cached transaction aggregates

## Operational Characteristics

* Stream ingestion is asynchronous and batched.
* Track association scales through spatial hashing rather than global matching.
* Lost-track handling uses bounded ghost state with automatic expiration.
* Endpoint responses are served from cached counters and precomputed snapshots.
* Startup indexing removes recurring file parsing overhead.
* The design prioritizes low-latency reads and stable behavior under concurrent multi-camera throughput.