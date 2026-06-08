from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from collections import defaultdict

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/app")

from emit import StreamEmitter, structured_event_line
from tracker import GhostVelocityCache, SpatialHashTracker

STORE_CAMERA_MAP = {
    "Store 1": {
        "CAM 1 - zone.mp4": "CAM_ZONE_01",
        "CAM 2 - zone.mp4": "CAM_ZONE_02",
        "CAM 3 - entry.mp4": "CAM_ENTRY_01",
        "CAM 5 - billing.mp4": "CAM_BILLING_01",
    },
    "Store 2": {
        "billing_area.mp4": "CAM_BILLING_02",
        "entry 1.mp4": "CAM_ENTRY_02",
        "entry 2.mp4": "CAM_ENTRY_03",
        "zone.mp4": "CAM_ZONE_03",
    },
}


def discover_video_assets(data_root: Path) -> List[Tuple[str, str, Path]]:
    assets: List[Tuple[str, str, Path]] = []
    for store_dir_name, camera_map in STORE_CAMERA_MAP.items():
        store_dir = data_root / store_dir_name
        if not store_dir.exists():
            continue
        for file_name, camera_id in camera_map.items():
            file_path = store_dir / file_name
            if file_path.exists():
                if store_dir_name == "Store 1":
                    store_id = "STORE_BLR_002"
                elif store_dir_name == "Store 2":
                    store_id = "ST1076"
                else:
                    store_id = "STORE_BLR_002"
                assets.append((store_id, camera_id, file_path))
    return assets


def _zone_from_camera(camera_id: str) -> str:
    if "BILLING" in camera_id:
        return "BILLING"
    if "ENTRY" in camera_id:
        return "ENTRY"
    return "ZONE"


def _open_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed_to_open_video:{path}")
    return capture


def _detect_with_fallback(frame, model, bgsub):
    detections = []
    if model is not None:
        try:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)]})
            return detections
        except Exception:
            pass
    fg = bgsub.apply(frame)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= 4000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({"bbox": [float(x), float(y), float(x + w), float(y + h)]})
    return detections


def _create_model():
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        _ = model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        return model
    except Exception:
        return None

@dataclass


class TrackSession:
    visitor_id: str
    session_seq: int
    last_camera_zones: Dict[str, str] = field(default_factory=dict)
    last_zone_seen_at: Optional[datetime] = None
    session_start_at: Optional[datetime] = None
    dwell_accumulator: Dict[str, int] = field(default_factory=dict)
    last_emit_at: Optional[datetime] = None
    present: bool = True
    billing_queue_joined: bool = False


def _build_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: Optional[str],
    dwell_ms: Optional[int],
    is_staff: bool,
    confidence: float,
    session_seq: int,
    queue_depth: Optional[int] = None,
) -> Dict[str, object]:
    return {
        "event_id": str(uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms if dwell_ms is not None else 0,
        "is_staff": is_staff,
        "confidence": round(confidence, 2),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": zone_id,
            "session_seq": session_seq,
        },
    }


def _canonical_visitor_id(visitor_id: str) -> str:
    text = str(visitor_id).strip()
    parts = text.split("_")
    if len(parts) >= 2 and parts[0] == "VIS":
        numeric_parts = [part for part in parts[1:] if part.isdigit()]
        if numeric_parts:
            return f"VIS_{numeric_parts[-1]}"
    return text


def main():
    data_root = Path(os.getenv("DATA_ROOT", "/app/data"))
    redis_host = os.getenv("REDIS_HOST", "redis_bus")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    emitter = StreamEmitter(host=redis_host, port=redis_port)
    emitter.ping()

    model = _create_model()

    assets = discover_video_assets(data_root)
    if not assets:
        raise RuntimeError(f"no_video_assets_found:{data_root}")

    trackers = {camera_id: SpatialHashTracker() for _, camera_id, _ in assets}
    ghosts = {camera_id: GhostVelocityCache(max_frames=45) for _, camera_id, _ in assets}
    bgsubs = {camera_id: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True) for _, camera_id, _ in assets}

    open_captures = {camera_id: _open_video(path) for _, camera_id, path in assets}

    session_state: Dict[Tuple[str, str], TrackSession] = {}
    session_counter: Dict[Tuple[str, str], int] = {}
    frame_stride = int(os.getenv("FRAME_STRIDE", "5"))
    camera_frame_indices: Dict[str, int] = defaultdict(int)

    try:
        while True:
            emitted = []

            for store_id, camera_id, video_path in assets:
                capture = open_captures[camera_id]
                ok, frame = capture.read()

                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                camera_frame_indices[camera_id] += 1
                if camera_frame_indices[camera_id] % frame_stride != 0:
                    continue

                bgsub = bgsubs[camera_id]
                tracker = trackers[camera_id]
                detections = _detect_with_fallback(frame, model, bgsub)
                tracks = tracker.update_and_match(detections)
                now = datetime.now(timezone.utc)

                for track in tracks:
                    raw_visitor_id = f"VIS_{store_id[-3:]}_{track['track_id']:06d}"

                    session_seq_key = (store_id, raw_visitor_id)
                    session = session_state.get(session_seq_key)

                    if session is None:
                        session_counter[session_seq_key] = session_counter.get(session_seq_key, 0) + 1
                        session = TrackSession(
                            visitor_id=raw_visitor_id,
                            session_seq=session_counter[session_seq_key],
                            session_start_at=now
                        )
                        session_state[session_seq_key] = session

                    zone = _zone_from_camera(camera_id)
                    last_camera_zone = session.last_camera_zones.get(camera_id)

                    if last_camera_zone != zone:
                        if last_camera_zone is not None:
                            last_dwell = session.dwell_accumulator.get(last_camera_zone, 0)
                            emitted.append(
                                _build_event(
                                    store_id, camera_id, raw_visitor_id, "ZONE_EXIT", now,
                                    last_camera_zone, last_dwell, False, 0.88, session.session_seq
                                )
                            )

                        event_type = "ZONE_ENTER"
                        if not session.last_camera_zones and zone == "ENTRY":
                            event_type = "ENTRY"
                        elif session.last_camera_zones and zone == "ENTRY":
                            event_type = "REENTRY"

                        emitted.append(
                            _build_event(
                                store_id, camera_id, raw_visitor_id, event_type, now,
                                zone, None, False, 0.92, session.session_seq, queue_depth=len(tracks)
                            )
                        )
                        session.last_camera_zones[camera_id] = zone
                        session.last_zone_seen_at = now
                    else:
                        if session.last_zone_seen_at is None: session.last_zone_seen_at = now
                        dwell_ms = int((now - session.last_zone_seen_at).total_seconds() * 1000)
                        session.dwell_accumulator[zone] = dwell_ms

                        if dwell_ms > 0 and dwell_ms % 30000 < 1000:
                            emitted.append(_build_event(store_id, camera_id, raw_visitor_id, "ZONE_DWELL", now, zone, dwell_ms, False, 0.91, session.session_seq, queue_depth=len(tracks)))

                        if "BILLING" in zone and not session.billing_queue_joined:
                            emitted.append(_build_event(store_id, camera_id, raw_visitor_id, "BILLING_QUEUE_JOIN", now, zone, None, False, 0.9, session.session_seq, queue_depth=len(tracks)))
                            session.billing_queue_joined = True

                    session.last_emit_at = now

                for key in list(session_state.keys()):
                    sess = session_state[key]
                    if key[0] == store_id and sess.last_emit_at and (now - sess.last_emit_at) > timedelta(minutes=1):
                        last_zone_val = list(sess.last_camera_zones.values())[-1] if sess.last_camera_zones else None
                        emitted.append(_build_event(store_id, camera_id, sess.visitor_id, "EXIT", now, last_zone_val, None, False, 0.87, sess.session_seq))
                        session_state.pop(key, None)

                if emitted:
                    emitter.emit_many(emitted)
                    for event in emitted:
                        print(structured_event_line(event), flush=True)

                ghosts[camera_id].step()
                time.sleep(0.01)

    finally:
        for cap in open_captures.values():
            cap.release()

if __name__ == "__main__":
    main()
