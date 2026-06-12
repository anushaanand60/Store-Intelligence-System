import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "app"))

from app.core_logic import ProductionStateEngine

def get_process_memory(pid):
    if not psutil:
        return 0
    try:
        parent = psutil.Process(pid)
        total_mem = parent.memory_info().rss
        for child in parent.children(recursive=True):
            total_mem += child.memory_info().rss
        return total_mem
    except Exception:
        return 0

def run_pipeline(mode: str, max_frames: int = 150) -> tuple[float, int, list[dict], float]:
    cmd = [sys.executable, "pipeline/detect.py"]
    env = os.environ.copy()
    env["SEQUENTIAL"] = "true" if mode == "sequential" else "false"
    env["MAX_FRAMES"] = str(max_frames)
    env["DATA_ROOT"] = os.path.join(WORKSPACE_DIR, "data")
    env["FRAME_STRIDE"] = "5"

    start_time = time.perf_counter()
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

    peak_memory = 0
    while True:
        ret = process.poll()
        if psutil:
            mem = get_process_memory(process.pid)
            if mem > peak_memory:
                peak_memory = mem
        if ret is not None:
            break
        time.sleep(0.05)

    t.join()
    elapsed = time.perf_counter() - start_time

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

    total_frames = 8 * max_frames
    peak_memory_mb = peak_memory / (1024.0 * 1024.0)
    return elapsed, total_frames, events, peak_memory_mb

def main():
    print("      PORTING PIPELINE BENCHMARK: SEQUENTIAL VS PARALLEL WORKERS      ")
    print("\n")

    max_frames = 300
    print(f"Executing sequential baseline run (MAX_FRAMES={max_frames} per camera)...")
    seq_time, seq_frames, seq_events, seq_mem = run_pipeline("sequential", max_frames)
    seq_fps = seq_frames / seq_time
    print(f"Sequential Completed: {seq_time:.2f}s, Aggregate FPS: {seq_fps:.2f}, Peak Memory: {seq_mem:.2f} MB")
    print(f"Events Emitted: {len(seq_events)}\n")

    print(f"Executing parallel process run (MAX_FRAMES={max_frames} per camera)...")
    par_time, par_frames, par_events, par_mem = run_pipeline("parallel", max_frames)
    par_fps = par_frames / par_time
    print(f"Parallel Completed: {par_time:.2f}s, Aggregate FPS: {par_fps:.2f}, Peak Memory: {par_mem:.2f} MB")
    print(f"Events Emitted: {len(par_events)}\n")

    try:
        events_path = os.path.join(WORKSPACE_DIR, "data", "benchmark_events.json")
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(par_events, f, indent=2)
    except Exception as e:
        print(f"Failed to save benchmark events: {e}")

    print("Analyzing and verifying consistency of outputs...")
    txn_path = os.path.join(WORKSPACE_DIR, "data", "POS_transactions.csv")

    seq_events_sorted = sorted(
        seq_events,
        key=lambda e: (e.get("timestamp", ""), e.get("camera_id", ""), e.get("visitor_id", ""), e.get("event_type", ""))
    )
    par_events_sorted = sorted(
        par_events,
        key=lambda e: (e.get("timestamp", ""), e.get("camera_id", ""), e.get("visitor_id", ""), e.get("event_type", ""))
    )

    engine_seq = ProductionStateEngine()
    engine_seq.load_transactions(txn_path)
    engine_seq.ingest_events(seq_events_sorted)
    seq_metrics = engine_seq.snapshot_metrics("STORE_BLR_002")
    seq_dataset = engine_seq.generate_ml_dataset("STORE_BLR_002")

    engine_par = ProductionStateEngine()
    engine_par.load_transactions(txn_path)
    engine_par.ingest_events(par_events_sorted)
    par_metrics = engine_par.snapshot_metrics("STORE_BLR_002")
    par_dataset = engine_par.generate_ml_dataset("STORE_BLR_002")

    consistency_passed = True
    reasons = []

    try:
        assert seq_metrics["unique_visitors"] == par_metrics["unique_visitors"], f"Unique Visitors mismatch: {seq_metrics['unique_visitors']} vs {par_metrics['unique_visitors']}"
        assert seq_metrics["total_entries"] == par_metrics["total_entries"], f"Total Entries mismatch: {seq_metrics['total_entries']} vs {par_metrics['total_entries']}"
        assert len(seq_dataset) == len(par_dataset), f"ML Dataset size mismatch: {len(seq_dataset)} vs {len(par_dataset)}"

        seq_visitors = {row["visitor_id"] for row in seq_dataset}
        par_visitors = {row["visitor_id"] for row in par_dataset}
        assert seq_visitors == par_visitors, "Visitor ID sets mismatch in ML feature datasets"

        seq_conversions = {row["visitor_id"]: row["conversion_outcome"] for row in seq_dataset}
        par_conversions = {row["visitor_id"]: row["conversion_outcome"] for row in par_dataset}
        for vid in seq_conversions:
            assert seq_conversions[vid] == par_conversions[vid], f"Conversion outcome mismatch for visitor {vid}: {seq_conversions[vid]} vs {par_conversions[vid]}"
    except AssertionError as exc:
        consistency_passed = False
        reasons.append(str(exc))

    speedup = seq_time / par_time

    print("                          BENCHMARK RESULTS                           ")
    print(f"{'Metric':<30} | {'Sequential':<12} | {'Parallel':<12}")
    print(f"{'Wall-clock Runtime(s)':<30} | {seq_time:<12.2f} | {par_time:<12.2f}")
    print(f"{'Aggregate Throughput(FPS)':<30} | {seq_fps:<12.2f} | {par_fps:<12.2f}")
    print(f"{'Peak Memory Usage(MB)':<30} | {seq_mem:<12.2f} | {par_mem:<12.2f}")
    print(f"{'Events Emitted':<30} | {len(seq_events):<12} | {len(par_events):<12}")
    print(f"{'Unique Visitors':<30} | {seq_metrics['unique_visitors']:<12} | {par_metrics['unique_visitors']:<12}")
    print(f"Measured Speedup Factor: {speedup:.2f}x")
    print(f"Data Consistency Verified: {'PASS' if consistency_passed else 'FAIL'}")

    if not consistency_passed:
        print(f"Mismatches detected: {reasons}")

    print("                      STITCHING TELEMETRY (PARALLEL)                 ")
    for key, val in engine_par.stitch_statistics.items():
        print(f"{key:<30}: {val}")

    if not consistency_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()