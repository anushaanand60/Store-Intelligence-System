import os
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("store_intelligence.worker")

def main():
    camera_id = os.getenv("CAMERA_ID", "CAM_1")
    video_path = os.getenv("VIDEO_PATH", "/shared_data/CAM 1.mp4")
    
    logger.info(json.dumps({
        "event": "worker_initialized_skeleton",
        "camera_id": camera_id,
        "target_source": video_path
    }))
    
    print("[Worker Loop] Waiting for system orchestration layer hooks...", flush=True)
    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()