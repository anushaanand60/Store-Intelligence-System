import os
import sys
import json
import csv
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "app"))

from app.core_logic import ProductionStateEngine, SessionRecord, NormalizedEvent

def main():
    events_path = WORKSPACE_DIR / "data" / "full_events.json"
    txn_path = WORKSPACE_DIR / "data" / "POS_transactions.csv"
    output_path = WORKSPACE_DIR / "data" / "ml_features_prefixes.csv"

    if not events_path.exists():
        print(f"Error: Full events not found at {events_path}.")
        sys.exit(1)

    print(f"Loading full events from {events_path}...")
    with open(events_path, "r", encoding="utf-8") as f:
        events_raw = json.load(f)

    events_sorted = sorted(
        events_raw,
        key=lambda e: (e.get("timestamp", ""), e.get("camera_id", ""), e.get("visitor_id", ""), e.get("event_type", ""))
    )

    print("Ingesting events and reconstructing shopper sessions...")
    engine = ProductionStateEngine()
    engine.load_transactions(str(txn_path))
    engine.ingest_events(events_sorted)

    store_ids = engine.store_ids()
    print(f"Stores detected: {store_ids}")

    prefix_data = []
    horizon_event_counts = {10: [], 25: [], 50: [], 75: [], 100: []}

    for store_id in store_ids:
        sessions = engine._non_staff_sessions(store_id)
        engine._purchase_units(store_id, sessions)
        print(f"Store {store_id}: Processing {len(sessions)} sessions...")

        for session in sessions:
            sorted_session_events = sorted(session.events, key=lambda e: e.timestamp)
            N = len(sorted_session_events)
            if N == 0:
                continue

            true_conversion = 1 if session.reached_stage("PURCHASE") else 0

            for observed_pct in [10, 25, 50, 75, 100]:
                k = max(1, round(N * (observed_pct / 100.0)))
                horizon_event_counts[observed_pct].append(k)

                prefix_session = SessionRecord(
                    store_id=session.store_id,
                    visitor_id=session.visitor_id,
                    session_seq=session.session_seq
                )

                for event in sorted_session_events[:k]:
                    prefix_session.ingest(event)

                features = prefix_session.to_ml_features()
                features["conversion_outcome"] = true_conversion
                features["observed_pct"] = observed_pct
                features["event_count"] = k
                features["total_events_in_session"] = N
                features["store_id"] = session.store_id
                prefix_data.append(features)

    total_examples = len(prefix_data)
    print(f"\nGenerated {total_examples} prefix examples from {len(prefix_data)//5} sessions.")
    print("\n")
    print("      OBSERVATION HORIZON EVENT STATISTICS    ")
    print(f"{'Horizon %':<12} | {'Mean Events':<12} | {'Min Events':<10} | {'Max Events':<10}")

    for pct in [10, 25, 50, 75, 100]:
        counts = horizon_event_counts[pct]
        mean_c = sum(counts) / len(counts) if counts else 0
        min_c = min(counts) if counts else 0
        max_c = max(counts) if counts else 0
        print(f"{pct:<11}% | {mean_c:<12.2f} | {min_c:<10} | {max_c:<10}")

    if total_examples > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(prefix_data[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(prefix_data)
        print(f"Saved prefix dataset containing {total_examples} rows to {output_path}")

    stats_path = WORKSPACE_DIR / "data" / "prefix_stats.json"
    summary_stats = {}
    for pct in [10, 25, 50, 75, 100]:
        counts = horizon_event_counts[pct]
        summary_stats[str(pct)] = {
            "mean": sum(counts) / len(counts) if counts else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "count": len(counts)
        }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

if __name__ == "__main__":
    main()