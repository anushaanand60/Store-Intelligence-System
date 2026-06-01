import os
import time
import json
import logging
import cv2
import numpy as np
import redis
from tracker import SpatialHashTracker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("store_intelligence.worker")

def detect_with_fallback(frame, model, bgsub):
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
        except Exception as exc:
            logger.info(json.dumps({"event": "yolo_runtime_fallback", "error": str(exc)}))
    fg = bgsub.apply(frame)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) <= 4000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append({"bbox": [float(x), float(y), float(x + w), float(y + h)]})
    return detections

def create_yolo():
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        _ = model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        logger.info(json.dumps({"event": "yolo_ready"}))
        return model
    except Exception as exc:
        logger.info(json.dumps({"event": "yolo_unavailable_fallback_to_mog2", "error": str(exc)}))
        return None

def main():
    camera_id = os.getenv("CAMERA_ID", "CAM_1")
    video_path = os.getenv("VIDEO_PATH", "/shared_data/CAM 1.mp4")
    redis_host = os.getenv("REDIS_HOST", "redis_bus")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    rc = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    for attempt in range(1, 6):
        try:
            rc.ping()
            break
        except Exception:
            time.sleep(2 ** attempt)

    tracker = SpatialHashTracker(grid_cols=10, grid_rows=10)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    
    model = create_yolo()
    bgsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    stream = f"camera_stream:{camera_id}"
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue
        detections = detect_with_fallback(frame, model, bgsub)
        tracks = tracker.update_and_match(detections)
        payload = {
            "camera_id": camera_id,
            "timestamp": f"{time.time():.6f}",
            "frame_index": str(frame_idx),
            "tracks": json.dumps(tracks),
            "active_tracks": str(tracker.get_active_count()),
        }
        rc.xadd(stream, payload, maxlen=1000)
        logger.info(json.dumps({"event": "frame_published", "camera_id": camera_id, "frame_index": frame_idx, "tracks": len(tracks)}))

if __name__ == "__main__":
    main()