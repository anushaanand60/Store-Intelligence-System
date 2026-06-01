import logging
from collections import defaultdict, deque

logger = logging.getLogger("store_intelligence.behavior")

class StoreIntelligenceBehaviorEngine:
    def __init__(self, zone_map):
        self.zone_map = zone_map
        self.track_state = {}
        self.total_entries = 0
        self.total_exits = 0
        self.live_occupancy = 0

    def _resolve_zone(self, camera_id, center):
        for zone in self.zone_map.get(camera_id, []):
            x1, y1, x2, y2 = zone["bbox"]
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                return zone["zone"]
        return None