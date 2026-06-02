# Engineering Choices

This document records the key engineering decisions made to keep the system reliable and efficient.

## 1) Unified Async Lifecycle Inside Main Service

* **Choice:**
  Run Redis consumers and analytics workers (`behavior_worker` and `anomaly_worker`) as asynchronous background tasks inside the FastAPI lifespan.

* **Alternative considered:**
  Running separate daemon scripts or independent worker processes.

* **Reasoning:**
  Managing multiple processes increases deployment complexity and introduces coordination overhead. Keeping background tasks in one async lifecycle allows shared access to `app.state.redis`, controlled startup and shutdown, and a single service health path through `GET /health`.

* **Result:**
  Stable container behavior with predictable Redis connectivity and simpler operations.

## 2) Spatial Hashing for Track Association

* **Choice:**
  Use a 2D spatial hash grid in `SpatialHashTracker` to limit candidate matches between detections and existing tracks.

* **Alternative considered:**
  Comparing every detection against every active track in each frame.

* **Reasoning:**
  A naive approach requires O(N × M) comparisons per frame, where N represents active tracks and M represents new detections. As store occupancy grows, this quickly becomes inefficient. Spatial hashing buckets tracks by grid cell and only checks neighboring cells, so matching stays focused on local candidates instead of the full scene.

* **Result:**
  Lower processing overhead and smoother performance as store occupancy increases.

## 3) LRU Cache with 45-Frame Ghost Track Persistence

* **Choice:**
  Keep temporarily lost tracks in `ghost_tracks` with a `ttl` of 45 and velocity-based position carry-forward.

* **Alternative considered:**
  Removing tracks immediately after a missed detection or retaining lost tracks indefinitely.

* **Reasoning:**
  Shoppers can temporarily disappear because of occlusions, crowded aisles, or intersecting paths. Immediate removal increases duplicate counting risk when the same track reappears, while unbounded retention increases stale state growth. Using a 45-frame LRU-backed ghost state strikes a balance between tracking continuity and bounded resource consumption.

* **Result:**
  More accurate tracking continuity, reduced double counting, and bounded memory usage for dropped-track state.

## 4) Boot-Time Data Indexing

* **Choice:**
  Load and preprocess layout and transaction datasets during application startup using `load_zone_map` and `load_transactions`.

* **Alternative considered:**
  Reading and parsing files during API requests.

* **Reasoning:**
  Parsing the Excel layout and transaction CSV at boot converts repeated I/O and transformation cost into a one-time initialization step. Requests then read from `app.state.zone_map` and `app.state.transactions_by_hour`.

* **Result:**
  Consistent low-latency API responses under repeated access.

## 5) Redis Streams as Event Backbone

* **Choice:**
  Use Redis Streams between camera workers and the central API, with per-camera channels like `camera_stream:CAM_1` through `camera_stream:CAM_5`.

* **Alternative considered:**
  Direct synchronous communication between workers and API components.

* **Reasoning:**
  Direct coupling can propagate processing delays upstream. Stream-based producer-consumer separation allows workers to publish with `xadd` and the API to consume in batches with `xread`, reducing contention during burst periods.

* **Result:**
  Better fault tolerance and smoother handling of burst traffic.

## 6) Read-Optimized Metrics APIs

* **Choice:**
  Serve `GET /metrics` and `GET /funnel` from precomputed behavior state and Redis counters such as `store:live_occupancy`, `store:total_entries`, and `store:total_exits`.

* **Alternative considered:**
  Recomputing analytics every time an API request is received.

* **Reasoning:**
  These endpoints are expected to be queried frequently by dashboards and evaluators. Recomputing funnel and occupancy state on each request would add avoidable latency, so responses use `get_metrics_snapshot`, `get_funnel_snapshot`, and cached transaction aggregates.

* **Result:**
  Fast API responses with stable output and minimal request-time computation.