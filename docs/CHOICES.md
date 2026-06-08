# Engineering Choices - Store Intelligence System

This document outlines the three key architectural decisions made during the system design, the options considered, what the AI suggested, and our final rationale.

---

## Decision 1: Detection Model & Pipeline Selection

### Options Considered
1. **High-Accuracy Heavyweights (YOLOv8x / RT-DETR)**: Offer excellent detection bounding box stability and higher raw confidence, but require substantial GPU compute and memory resources.
2. **Vision-Language Models (VLMs like PaliGemma / Gemini Nano)**: Offer zero-shot zone classification and human/staff detection capability out-of-the-box via natural language prompting.
3. **Lightweight Edge-Focused Model with Heuristics (YOLOv8-nano + Spatial Hash + Background Subtractor Fallback)**: Extremely low memory footprint and high execution speed on CPU, supplemented by spatial hashes and historical shadow caches.

### What AI Suggested
The AI initially recommended a hybrid VLM approach for zone classification and staff detection, suggestion prompts like:
> *"Classify if the person in the bounding box is wearing store staff uniform (red vest) or is a customer. Classify the zone category (Entry, Skincare, Billing) based on the surrounding context."*

### What We Chose and Why
We **overrode the AI's recommendation to use a VLM** and instead chose **YOLOv8-nano with a motion-subtraction fallback and local spatial heuristic tracking**.

**VLM Evaluation:**
* **Prompt Evaluated**: We tested zero-shot prompts for staff identification and zone classification using a VLM.
* **Why it Failed**: The VLM introduced a massive latency bottleneck (over 250ms per frame evaluation) and threatened to exceed container memory limits, risking Out-Of-Memory (OOM) crashes under simultaneous multi-camera video decode pressure.
* **Final Selection**: We deployed YOLOv8-nano. The low raw detection confidence on face-blurred or partially occluded edge objects was mitigated by pairing it with a **45-frame velocity shadow cache** and **Spatial Hash Tracker** in `pipeline/tracker.py`. This ensures high tracking continuity without the computational cost of a VLM.

---

## Decision 2: Event Schema Design Rationale

### Options Considered
1. **Polymorphic / Deeply Nested Payloads**: Separate schemas for `ZoneEntry`, `ZoneExit`, `QueueJoin`, etc., with polymorphic attributes based on event type.
2. **Protobuf/Binary Stream Serialization**: Highly compact binary serialization format for high-throughput stream ingestion.
3. **Flat JSON Envelope with Nested Metadata**: A consistent top-level JSON structure for every event type with a flexible `metadata` bag containing SKU zones, queue depths, and tracking variables.

### What AI Suggested
The AI suggested using a Polymorphic JSON Schema where the structure changes dynamically depending on the event type (e.g. including a sub-object `billing_details` only on queue events).

### What We Chose and Why
We chose the **Flat JSON Envelope with Nested Metadata**.
* **Reasoning**: Changing payload schemas introduce high parsing complexity and serialization cost in python-based asynchronous stream consumers. A flat schema allows the central ingest layer to quickly validate, index, and write events into the registry using standard Pydantic models. It isolates custom operational attributes (like `sku_zone` or `session_seq`) into a single metadata bag, keeping the main stream consumer logic simple, generic, and fast.

---

## Decision 3: API Architecture Choice (Memory Caching & Lifespan Isolation)

### Options Considered
1. **On-Demand Query-Time DB Joins**: Scan raw Redis streams and join with the POS transactions CSV dataset dynamically on every incoming request to `GET /stores/{id}/metrics`.
2. **Pre-computed Session-State Caching with Ingestion Processing**: Aggregate incoming events into a localized, in-memory state engine (`StoreAnalyticsEngine`) in real-time as events are ingested.
3. **External Task Queues (Celery/Redis Queue)**: Offload all metric calculations to background worker processes, writing the output back to a relational database.

### What AI Suggested
The AI suggested doing query-time database lookups or running pandas dataframe scans over the POS CSV files on every telemetry request to ensure absolute real-time accuracy.

### What We Chose and Why
We chose **Pre-computed Session-State Caching inside the FastAPI Lifespan Loop**.
* **Reasoning**: Dashboard applications demand sub-millisecond API response times. Repeatedly reading the POS CSV file or parsing large Redis Streams on every request to compute cumulative metrics (like `unique_visitors` and conversion rate) degrades performance. By processing and indexing the transactions at boot time and aggregating incoming events in memory (`ProductionStateEngine`), lookup requests are resolved in near-instantaneous time. We also isolated the engine instantiation to the FastAPI lifespan context, ensuring that states are clean and isolated across different integration test runs.
