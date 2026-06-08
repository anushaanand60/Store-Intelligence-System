from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import error, request
from typing import Any, Dict, Iterable, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/app")

import redis


class StreamEmitter:
    def __init__(self, host: str = "redis_bus", port: int = 6379, stream_name: str = "store:intelligence:events"):
        self.stream_name = stream_name
        redis_host = os.getenv("REDIS_HOST", host)
        redis_port = int(os.getenv("REDIS_PORT", str(port)))
        self.client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def emit(self, payload: Dict[str, Any], maxlen: int = 5000) -> str:
        sanitized_payload = {}
        for k, v in payload.items():
            if isinstance(v, bool):
                sanitized_payload[k] = "true" if v else "false"
            elif isinstance(v, (dict, list)):
                sanitized_payload[k] = json.dumps(v)
            elif v is None:
                sanitized_payload[k] = ""
            else:
                sanitized_payload[k] = str(v)

        return self.client.xadd(self.stream_name, sanitized_payload, maxlen=maxlen)

    def emit_many(self, payloads: Iterable[Dict[str, Any]], maxlen: int = 5000) -> List[str]:
        ids = []
        for payload in payloads:
            ids.append(self.emit(payload, maxlen=maxlen))
        return ids


def structured_event_line(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _post_event(api_url: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward pipeline event JSON to the API ingest endpoint.")
    parser.add_argument("--api-url", default=os.getenv("API_INGEST_URL", "http://localhost:8000/events/ingest"))
    args = parser.parse_args()

    forwarded = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            _post_event(args.api_url, payload)
            forwarded += 1
        except (json.JSONDecodeError, error.URLError, TimeoutError, ValueError) as exc:
            print(f"[emit] forward_failed: {exc}", file=sys.stderr, flush=True)
    print(f"[emit] forwarded={forwarded}", file=sys.stderr, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
