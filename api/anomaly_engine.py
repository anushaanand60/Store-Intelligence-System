import json
import logging
from collections import deque

logger = logging.getLogger("store_intelligence.anomaly")

class StoreIntelligenceAnomalyDetector:
    def __init__(self, window_size=30, z_threshold=3.5):
        self.window = deque(maxlen=window_size)
        self.z_threshold = z_threshold