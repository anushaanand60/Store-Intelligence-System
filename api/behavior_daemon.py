import os
import time
import json
import redis

def run_daemon():
    redis_host = os.getenv("REDIS_HOST", "redis_bus")
    r_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)

    print("[Behavior Daemon] Stream consumer bridge online. Subscribing to Redis channels...", flush=True)

    while True:
        try:
            streams = {f"camera_stream:CAM_{i}": "$" for i in range(1, 6)}
            response = r_client.xread(streams, block=1000, count=10)

            if not response:
                continue

            for stream_name, messages in response:
                camera_id = stream_name.split(":")[-1]
                
                for msg_id, payload in messages:
                    tracks = json.loads(payload.get("tracks", "[]"))
                    current_time = float(payload.get("timestamp", time.time()))
                    
                    print(f"[Daemon Processing] Ingested {len(tracks)} entity frames from channel: {camera_id}", flush=True)

        except Exception as err:
            print(f"[Behavior Daemon Error] Connection stream drop: {str(err)}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run_daemon()