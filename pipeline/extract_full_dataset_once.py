import os
import sys
import json
import subprocess
import threading
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "app"))

from app.core_logic import ProductionStateEngine

def run_pipeline(max_frames: int = 0) -> list[dict]:
    cmd = [sys.executable, "pipeline/detect_once.py"]
    env = os.environ.copy()
    env["SEQUENTIAL"] = "false"
    env["MAX_FRAMES"] = str(max_frames)
    env["DATA_ROOT"] = str(WORKSPACE_DIR / "data")
    env["FRAME_STRIDE"] = "5"

    process = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True
    )

    stdout_lines = []

    def read_stdout(stream):
        for line in stream:
            stdout_lines.append(line)

    t = threading.Thread(target=read_stdout, args=(process.stdout,))
    t.start()
    process.wait()
    t.join()

    events = []
    for line in stdout_lines:
        line_str = line.strip()
        if not line_str:
            continue
        try:
            event = json.loads(line_str)
            if isinstance(event, dict) and "event_id" in event:
                events.append(event)
        except json.JSONDecodeError:
            pass

    return events

def main():
    print("Executing full pipeline processing across all camera videos...")
    print("This will process 100% of the available frames (MAX_FRAMES=0)...")

    events = run_pipeline(max_frames=0)
    print(f"Total events generated: {len(events)}")

    events_sorted = sorted(
        events,
        key=lambda e: (e.get("timestamp", ""), e.get("camera_id", ""), e.get("visitor_id", ""), e.get("event_type", ""))
    )

    txn_path = WORKSPACE_DIR / "data" / "POS_transactions.csv"
    engine = ProductionStateEngine()
    engine.load_transactions(str(txn_path))
    engine.ingest_events(events_sorted)

    store_ids = engine.store_ids()
    print(f"Stores detected: {store_ids}")

    results = {}
    for store_id in store_ids:
        sessions = engine._non_staff_sessions(store_id)
        results[store_id] = len(sessions)
        print(f"Store {store_id}: {len(sessions)} sessions")

    output_path = WORKSPACE_DIR / "data" / "full_events.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print(f"Saved full events to {output_path}")

    summary = {
        "total_events": len(events),
        "sessions_per_store": results,
        "total_sessions": sum(results.values())
    }
    summary_path = WORKSPACE_DIR / "data" / "full_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()