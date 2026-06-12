import os
import sys
import json
import csv
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "app"))

from app.core_logic import ProductionStateEngine

def main():
    events_path = WORKSPACE_DIR / "data" / "benchmark_events.json"
    txn_path = WORKSPACE_DIR / "data" / "POS_transactions.csv"
    output_path = WORKSPACE_DIR / "data" / "ml_features_real.csv"

    if not events_path.exists():
        print(f"Error: Benchmark events not found at {events_path}. Run the benchmark first!")
        sys.exit(1)

    print(f"Loading events from {events_path}...")
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    events_sorted = sorted(
        events,
        key=lambda e: (e.get("timestamp", ""), e.get("camera_id", ""), e.get("visitor_id", ""), e.get("event_type", ""))
    )

    print("Ingesting events and correlating checkouts against POS transactions...")
    engine = ProductionStateEngine()
    engine.load_transactions(str(txn_path))
    engine.ingest_events(events_sorted)

    store_ids = engine.store_ids()
    print(f"Stores detected: {store_ids}")

    combined_dataset = []
    for store_id in store_ids:
        dataset = engine.generate_ml_dataset(store_id)
        combined_dataset.extend(dataset)

    total_sessions = len(combined_dataset)
    total_conversions = sum(1 for row in combined_dataset if row.get("conversion_outcome") == 1)

    if total_sessions > 0:
        conversion_rate = total_conversions / total_sessions
        conversions_pct = (total_conversions / total_sessions) * 100
        non_conversions_pct = 100.0 - conversions_pct
    else:
        conversion_rate = 0.0
        conversions_pct = 0.0
        non_conversions_pct = 0.0

    print("           DATASET SUFFICIENCY GATE           ")
    print(f"Total Sessions       : {total_sessions}")
    print(f"Total Conversions    : {total_conversions}")
    print(f"Conversion Rate      : {conversion_rate:.2%}")
    print(f"Class Balance (Conv) : {conversions_pct:.2f}% ({total_conversions} sessions)")
    print(f"Class Balance (Non)  : {non_conversions_pct:.2f}% ({total_sessions - total_conversions} sessions)")

    if total_sessions < 100:
        print("WARNING: Dataset too small for reliable model evaluation.")
        print("Continuing as an exploratory baseline.")
    else:
        print("SUCCESS: Dataset size is sufficient for standard model evaluation.")

    if total_sessions > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(combined_dataset[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(combined_dataset)
        print(f"Saved real dataset containing {total_sessions} rows to {output_path}")
    else:
        print("Error: No sessions extracted from events.")
        sys.exit(1)

if __name__ == "__main__":
    main()