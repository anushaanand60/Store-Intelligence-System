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
    events_path = WORKSPACE_DIR / "data" / "full_events.json"
    txn_path = WORKSPACE_DIR / "data" / "POS_transactions.csv"
    output_path = WORKSPACE_DIR / "data" / "ml_features_full.csv"

    if not events_path.exists():
        print(f"Error: Full events not found at {events_path}.")
        sys.exit(1)

    print(f"Loading full events from {events_path}...")
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
        print(f"Store {store_id}: generated {len(dataset)} sessions for ML")
        store_convs = sum(1 for row in dataset if row.get("conversion_outcome") == 1)
        store_non_convs = len(dataset) - store_convs
        print(f"  Conversions: {store_convs} ({store_convs/len(dataset):.2%})")
        print(f"  Non-conversions: {store_non_convs} ({store_non_convs/len(dataset):.2%})")
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

    print("      FULL DATASET SUFFICIENCY & CLASS STATISTICS      ")
    print(f"Total Sessions       : {total_sessions}")
    print(f"Total Conversions    : {total_conversions}")
    print(f"Conversion Rate      : {conversion_rate:.2%}")
    print(f"Class Balance (Conv) : {conversions_pct:.2f}% ({total_conversions} sessions)")
    print(f"Class Balance (Non)  : {non_conversions_pct:.2f}% ({total_sessions - total_conversions} sessions)")
    print("\n")

    if total_sessions > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(combined_dataset[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(combined_dataset)
        print(f"Saved full dataset containing {total_sessions} rows to {output_path}")

    stats = {}
    for store_id in store_ids:
        prefix = f"VIS_{store_id[-3:]}"
        store_rows = [row for row in combined_dataset if row.get("visitor_id", "").startswith(prefix)]
        s_conv = sum(1 for r in store_rows if r.get("conversion_outcome") == 1)
        stats[store_id] = {
            "sessions": len(store_rows),
            "conversions": s_conv,
            "non_conversions": len(store_rows) - s_conv
        }

    with open(WORKSPACE_DIR / "data" / "full_dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()