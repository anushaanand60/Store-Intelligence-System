import json
import logging
from collections import deque

logger = logging.getLogger("store_intelligence.anomaly")

class StoreIntelligenceAnomalyDetector:
    def __init__(self, window_size=30, z_threshold=3.5):
        self.window = deque(maxlen=window_size)
        self.z_threshold = z_threshold

    def check_stream_velocity(self, current_entries):
        self.window.append(float(current_entries))
        if len(self.window) < 10:
            return None
        baseline = list(self.window)[:-1]
        mean = sum(baseline) / len(baseline)
        var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
        std = (var ** 0.5)+ 1e-6
        z_score = abs(float(current_entries) - mean) / std
        if z_score > self.z_threshold:
            event = {
                "event": "traffic_spike",
                "severity": "critical",
                "current_entries": int(current_entries),
                "baseline_mean": round(mean, 4),
                "z_score": round(z_score, 4),
            }
            logger.warning(json.dumps(event))
            return event
        return None