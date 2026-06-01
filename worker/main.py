import os
import time
import json
import logging
import cv2
from tracker import SpatialHashTracker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("store_intelligence.worker")

def detect_with_fallback(frame, bgsub):
    detections = []
    fg = bgsub.apply(frame)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) <= 4000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append({"bbox": [float(x), float(y), float(x + w), float(y + h)]})
    return detections

def main():
    camera_id = os.getenv("CAMERA_ID", "CAM_1")
    video_path = os.getenv("VIDEO_PATH", "/shared_data/CAM 1.mp4")
    tracker = SpatialHashTracker(grid_cols=10, grid_rows=10)
    if not os.path.exists(video_path):
        logger.info(json.dumps({"event": "mock_video_source_running", "path": video_path}))
        print("[Worker] Video file path not found yet, running on idle cycle...", flush=True)
        while True:
            time.sleep(5)
            
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")

    bgsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    frame_idx = 0
    
    print(f"[Worker] Local CV feed processing opened for {camera_id}", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
                
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue
        detections = detect_with_fallback(frame, bgsub)

    logger.info(json.dumps({
        "event": "local_frame_processed", 
        "camera_id": camera_id, 
        "frame_index": frame_idx, 
        "detected_blobs": len(detections)
    }))

if __name__ == "__main__":
    main()