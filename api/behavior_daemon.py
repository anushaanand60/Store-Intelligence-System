import os
import time
import json
import redis
from behavior_engine import StoreIntelligenceBehaviorEngine

def run_daemon():
    redis_host = os.getenv("REDIS_HOST", "redis_bus")
    r_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    engine = StoreIntelligenceBehaviorEngine(r_client)
    print("[Behavior Daemon] Memory fusion layer online. Syncing behavioral states to Redis Bus...", flush=True)

    while True:
        try:
            streams = {f"camera_stream:CAM_{i}": "$" for i in range(1, 6)}
            response = r_client.xread(streams, block=1000, count=10)

            if not response:
                continue

            for stream_name, messages in response:
                camera_id = stream_name.split(":")[-1]
                active_track_ids_in_frame = []
                
                for msg_id, payload in messages:
                    tracks = json.loads(payload.get("tracks", "[]"))
                    current_time = float(payload.get("timestamp", time.time()))
                    
                    for entity in tracks:
                        track_id = entity["track_id"]
                        bbox = entity["bbox"]
                        active_track_ids_in_frame.append(track_id)

                        classification = engine.process_track_frame(camera_id, track_id, bbox, current_time)

                    engine.handle_dropped_tracks(active_track_ids_in_frame, camera_id)

            r_client.set("store:staff_count", len(engine.staff_registry))
            r_client.set("store:social_groups_count", len(engine.social_groups))

            zone_counts = {"CAM_3": 0, "CAM_1_4": 0, "CAM_2": 0, "CAM_5": 0}
            for t_id, data in engine.track_history.items():
                last_zone = data["zones_visited"][-1]
                if last_zone in ["CAM_1", "CAM_4"]:
                    zone_counts["CAM_1_4"] += 1
                elif last_zone == "CAM_2":
                    zone_counts["CAM_2"] += 1
                elif last_zone == "CAM_5":
                    zone_counts["CAM_5"] += 1
                elif last_zone == "CAM_3":
                    zone_counts["CAM_3"] += 1

            r_client.hset("store:funnel_distribution", mapping=zone_counts)

        except Exception as err:
            print(f"[Behavior Daemon Error] Connection stream drop: {str(err)}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run_daemon()