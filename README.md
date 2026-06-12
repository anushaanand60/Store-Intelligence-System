# Store Intelligence System

An in-store retail analytics and customer journey prediction platform. The system ingests concurrent camera feeds, tracks shopper trajectories, stitches multi-camera identities, and predicts customer conversion likelihoods before shoppers approach the checkout area.

---

## System Architecture

The system splits concerns into three layers: a high-throughput multi-process computer vision pipeline, a durable Redis message broker, and a stateful FastAPI application server.

```
       +---------------------------------------------+
       |           Concurrent Camera Feeds           |
       |  (Store 1: 4 Cams | Store 2: 4 Cams | etc)  |
       +---------------------------------------------+
                              |
                              v [YOLOv8 + OpenCV]
       +---------------------------------------------+
       |          Multi-Process Producer             |
       |  - Isolated OS Process per Camera           |
       |  - O(N) Spatial Hash Target Tracker         |
       |  - 45-Frame Ghost Velocity Cache            |
       +---------------------------------------------+
                              |
                              v [Redis Stream Events]
       +---------------------------------------------+
       |         Durable Redis Event Broker          |
       |   - Streams: store:intelligence:events      |
       |   - Consumer Group: api_analytics_consumers |
       +---------------------------------------------+
                              |
                              v [Durable Stream Reading]
       +---------------------------------------------+
       |            FastAPI API Ingest               |
       |   - Stateful StoreAnalyticsEngine           |
       |   - Camera-Boundary Identity Stitching      |
       |   - Live Heatmaps, Metrics & Funnels        |
       +---------------------------------------------+
```

---

## Key Capabilities and Technical Highlights

### 1. Multi-Process Camera Parallelization
To bypass Python's Global Interpreter Lock (GIL), [pipeline/detect.py](pipeline/detect.py) spawns an isolated OS process per camera stream. Workers perform concurrent OpenCV decoding and YOLOv8 inference, synchronizing console logging via a shared multiprocessing lock. 
*   **Sequential Baseline**: 32.92 FPS
*   **Parallel Execution (8 Workers)**: 62.80 FPS
*   **Multiprocessing Speedup**: **1.91x**
*   *Validation Script*: [tests/benchmark_parallelization.py](tests/benchmark_parallelization.py)

### 2. Occlusion-Resistant Tracking
The tracking loop features a **Spatial Hash Tracker** and a **Ghost Velocity Cache** in [pipeline/tracker.py](pipeline/tracker.py):
*   **Spatial Hashing**: Maps bounding boxes to a $10 \times 10$ coordinate grid. IOU matching is restricted to neighboring cells, reducing track association complexity from $O(N^2)$ to $O(N)$.
*   **Ghost Projections**: When a shopper is temporarily occluded, a 45-frame velocity buffer projects their trajectory vector forward to cleanly re-associate their `visitor_id` once they reappear.

### 3. Camera-Boundary Identity Stitching
As customers move through the store (crossing fields of view from `ENTRY -> ZONE -> BILLING`), the state engine in [app/core_logic.py](app/core_logic.py) stitches camera-specific tracking aliases into single canonical visitor sessions. 
*   **Transition Score**:
    $$\text{Score} = 0.40 \cdot S_{\text{temporal}} + 0.25 \cdot S_{\text{scale}} + 0.20 \cdot S_{\text{aspect}} + 0.15 \cdot S_{\text{boundary}}$$
*   **Ambiguity Margin & Uniqueness**: Stitches are only accepted if the best candidate's score exceeds the runner-up by a margin $> 0.15$ (`AMBIGUITY_MARGIN`), preventing identity collapse.
*   **Stitching Telemetry Endpoint**: `GET /stores/{store_id}/stitch-metrics` exposes real-time attempts, acceptances, rejections, and the overall unique visitor inflation reduction.

### 4. Early Conversion Prediction and Billing Ablation
The machine learning pipeline predicts shopper conversion (attributing checkout events to POS transaction records) early in the chronological journey.
*   **Duration-Based Slicing**: Sessions are sliced at elapsed duration horizons ($10\%, 25\%, 50\%, 75\%$).
*   **Target Leakage Audit**: Evaluated models with all billing-camera telemetry completely removed (**Ablation 2**). Even when blind to physical checkout areas, the Random Forest model achieves an **ROC-AUC of 0.7415 at 25% duration** and **0.8709 at 50% duration** based strictly on AISLE dwell-times, exploration footprint, and movement speed.
*   *Audit Reports*: [docs/early_prediction_report.md](docs/early_prediction_report.md) and [docs/reid_feasibility_report.md](docs/reid_feasibility_report.md).

---

## Repository File Structure

```
Store-Intelligence/
├── app/
│   ├── core_logic.py          # Stateful store analytics engine & stitching logic
│   ├── main.py                # FastAPI server, endpoints & Redis consumer groups
│   ├── models.py              # Pydantic schemas (EventEnvelope, EventMetadata)
│   └── requirements.txt       # FastAPI application dependencies
├── pipeline/
│   ├── detect.py              # Multiprocess camera ingestion & YOLO tracking
│   ├── tracker.py             # SpatialHashTracker and GhostVelocityCache
│   ├── emit.py                # Redis Stream emitter utility
│   ├── detect_once.py         # Full run-to-completion ingestion script
│   ├── export_full_dataset_csv.py   # Compiles real session telemetry into CSV
│   ├── reid_feasibility_study.py    # Offline study replaying ambiguity rejections
│   ├── generate_duration_prefix_dataset.py # Slices prefixes on elapsed duration
│   └── train_duration_prefix_models.py     # Cross-validated ablated model evaluation
├── docs/
│   ├── DESIGN.md              # System design decisions and engineering trade-offs
│   ├── CHOICES.md             # Architecture alternatives and technology choices
│   ├── final_architecture.md  # Detailed system blueprints and sequences
│   ├── early_prediction_report.md  # Journey-difficulty ML prediction analysis
│   ├── reid_feasibility_report.md  # OSNet Re-ID embedding feasibility study
│   ├── store_invariant_feature_design.md # Topology-invariant feature designs
│   └── video_assets_report.md # Inventory and crop sanity checklist for video feeds
├── data/                      # Local data assets and model outputs (git-ignored, except key plots)
│   ├── duration_prefix_prediction_curve.png # Multi-horizon early prediction learning curve
│   ├── stability_audit.png    # Validation performance distributions for classifiers
│   ├── ablation_stability.png # Ablation study performance boxplots & feature importances
│   ├── feature_normalization_audit.png # Distributions comparing raw vs. normalized features
│   ├── Store 1/               # [git-ignored due to file size] Raw video camera streams for Store 1
│   │   ├── CAM 1 - zone.mp4
│   │   ├── CAM 2 - zone.mp4
│   │   ├── CAM 3 - entry.mp4
│   │   └── CAM 5 - billing.mp4
│   ├── Store 2/               # [git-ignored due to file size] Raw video camera streams for Store 2
│   │   ├── billing_area.mp4
│   │   ├── entry 1.mp4
│   │   ├── entry 2.mp4
│   │   └── zone.mp4
│   ├── POS - sample transactionsb1e826f.csv # [git-ignored due to POS PII/size] Raw transaction logs
│   ├── sample_eventsbe42122.jsonl # [git-ignored due to event log volume] Multi-camera trajectory events
│   └── best_conversion_model.pkl # [git-ignored binary artifact] Serialized Random Forest model scaler and weights
├── tests/
│   ├── test_pipeline.py       # Unit tests for tracker and ghost cache liveness
│   ├── test_metrics.py        # Integration tests for consumer groups and metrics
│   ├── test_anomalies.py      # Regression tests for anomaly warning states
│   └── benchmark_parallelization.py # Benchmark sequential vs. parallel FPS
├── docker-compose.yml         # Container definitions for app, redis, and pipelines
└── README.md                  # This file
```

---

## Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Python 3.10+ (if running scripts locally)

### Running the System with Docker
1.  Spin up the shared Redis bus, central API backend, and pipeline worker:
    ```bash
    docker compose up --build
    ```
2.  Launch the live operational terminal dashboard:
    ```bash
    python dashboard_render.py
    ```

### Running the Test Suite
Ensure Redis and Pydantic dependencies are satisfied, then run the cross-validated test suites:
```bash
python -m pytest
```

---

## Performance and ML Metrics Summary

### Multiprocessing Stream Execution Benchmarks
| Metric | Sequential Mode | Parallel Mode (8 Workers) |
| :--- | :---: | :---: |
| **Wall-clock Runtime** | 72.90s | 38.22s |
| **Aggregate Throughput** | 32.92 FPS | 62.80 FPS |
| **Data Consistency** | PASS | PASS |
| **Total Events Emitted** | 181 | 181 |
| **Unique Visitors** | 25 | 25 |

### Early Prediction Model Performance (Random Forest ROC-AUC)
| Journey Progress | Reference Config | Ablation 2 (Billing Blocked) |
| :--- | :---: | :---: |
| **10% Duration** | 0.7950 | 0.5751 |
| **25% Duration** | 0.8544 | 0.7415 |
| **50% Duration** | 0.9176 | 0.8709 |
| **100% Duration** | 0.9736 | 0.9764 |

---

*For detailed component specifications and interaction diagrams, refer to [docs/final_architecture.md](docs/final_architecture.md).*
