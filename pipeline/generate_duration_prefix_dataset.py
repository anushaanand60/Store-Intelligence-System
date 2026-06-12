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
    output_std_path = WORKSPACE_DIR / "data" / "ml_features_duration_prefixes.csv"
    output_abl_path = WORKSPACE_DIR / "data" / "ml_features_duration_prefixes_no_billing.csv"

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

    std_prefix_data = []
    abl_prefix_data = []
    horizon_event_counts = {10: [], 25: [], 50: [], 75: [], 100: []}
    horizon_billing_visits = {10: 0, 25: 0, 50: 0, 75: 0, 100: 0}
    total_sessions_count = 0

    for store_id in store_ids:
        sessions = engine._non_staff_sessions(store_id)
        engine._purchase_units(store_id, sessions)
        print(f"Store {store_id}: Processing {len(sessions)} sessions...")

        for session in sessions:
            sorted_session_events = sorted(session.events, key=lambda e: e.timestamp)
            N = len(sorted_session_events)
            if N == 0:
                continue

            total_sessions_count += 1
            T_start = sorted_session_events[0].timestamp
            T_end = sorted_session_events[-1].timestamp
            duration_sec = (T_end - T_start).total_seconds()
            true_conversion = 1 if session.reached_stage("PURCHASE") else 0

            for observed_pct in [10, 25, 50, 75, 100]:
                cutoff_sec = duration_sec * (observed_pct / 100.0)
                prefix_events = [e for e in sorted_session_events if (e.timestamp - T_start).total_seconds() <= cutoff_sec]

                if not prefix_events:
                    prefix_events = [sorted_session_events[0]]

                k = len(prefix_events)
                horizon_event_counts[observed_pct].append(k)

                has_visited_billing = any("BILLING" in str(e.camera_id).upper() for e in prefix_events)
                if has_visited_billing:
                    horizon_billing_visits[observed_pct] += 1

                prefix_session_std = SessionRecord(
                    store_id=session.store_id,
                    visitor_id=session.visitor_id,
                    session_seq=session.session_seq
                )
                for event in prefix_events:
                    prefix_session_std.ingest(event)

                features_std = prefix_session_std.to_ml_features()
                features_std["conversion_outcome"] = true_conversion
                features_std["observed_pct"] = observed_pct
                features_std["event_count"] = k
                features_std["total_events_in_session"] = N
                features_std["store_id"] = session.store_id
                features_std["has_visited_billing"] = 1 if has_visited_billing else 0
                std_prefix_data.append(features_std)

                prefix_events_no_bill = [e for e in prefix_events if not "BILLING" in str(e.camera_id).upper()]
                if not prefix_events_no_bill:
                    prefix_events_no_bill = [sorted_session_events[0]]

                prefix_session_abl = SessionRecord(
                    store_id=session.store_id,
                    visitor_id=session.visitor_id,
                    session_seq=session.session_seq
                )
                for event in prefix_events_no_bill:
                    prefix_session_abl.ingest(event)

                features_abl = prefix_session_abl.to_ml_features()
                features_abl["conversion_outcome"] = true_conversion
                features_abl["observed_pct"] = observed_pct
                features_abl["event_count"] = len(prefix_events_no_bill)
                features_abl["total_events_in_session"] = N
                features_abl["store_id"] = session.store_id
                features_abl["billing_reached"] = 0
                abl_prefix_data.append(features_abl)

    print(f"\nGenerated {len(std_prefix_data)} standard and ablated prefix examples from {total_sessions_count} sessions.")
    print("\n")
    print("      HORIZON STATISTICS & BILLING VISITATION RATES     ")
    print(f"{'Horizon %':<12} | {'Mean Events':<12} | {'Billing Visits':<15} | {'Visitation Rate':<15}")

    visitation_rates = {}
    for pct in [10, 25, 50, 75, 100]:
        counts = horizon_event_counts[pct]
        mean_c = sum(counts) / len(counts) if counts else 0
        visits = horizon_billing_visits[pct]
        rate = visits / total_sessions_count if total_sessions_count > 0 else 0
        visitation_rates[pct] = rate
        print(f"{pct:<11}% | {mean_c:<12.2f} | {visits:<15} | {rate:<15.2%}")

    if len(std_prefix_data) > 0:
        for path, data in [(output_std_path, std_prefix_data), (output_abl_path, abl_prefix_data)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            headers = list(data[0].keys())
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            print(f"Saved dataset containing {len(data)} rows to {path}")

        summary_stats = {
            "visitation_rates": {str(k): v for k, v in visitation_rates.items()},
            "event_counts": {
                str(pct): {
                    "mean": sum(horizon_event_counts[pct]) / len(horizon_event_counts[pct]),
                    "min": min(horizon_event_counts[pct]),
                    "max": max(horizon_event_counts[pct])
                }
                for pct in [10, 25, 50, 75, 100]
            }
        }

        with open(WORKSPACE_DIR / "data" / "duration_prefix_stats.json", "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)

if __name__ == "__main__":
    main()