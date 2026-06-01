import logging
from collections import defaultdict, deque

logger = logging.getLogger("store_intelligence.behavior")

class StoreIntelligenceBehaviorEngine:
    def __init__(self, zone_map):
        self.zone_map = zone_map
        self.track_state = {}
        self.staff_registry = set()
        self.zone_occupancy = defaultdict(set)
        self.funnel_stage_sets = {"entrance": set(), "skincare": set(), "cosmetics": set(), "checkout": set()}
        self.total_entries = 0
        self.total_exits = 0
        self.live_occupancy = 0

    def _resolve_zone(self, camera_id, center):
        for zone in self.zone_map.get(camera_id, []):
            x1, y1, x2, y2 = zone["bbox"]
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                return zone["zone"]
        return None
    
    def _resolve_stage(self, camera_id, zone_name):
        if camera_id == "CAM_3":
            return "entrance"
        if camera_id in {"CAM_1", "CAM_4"}:
            return "skincare"
        if camera_id == "CAM_2":
            return "cosmetics"
        if camera_id == "CAM_5":
            return "checkout"
        if not zone_name:
            return None
        z = zone_name.lower()
        if "entrance" in z or "exit" in z:
            return "entrance"
        if "skin" in z or "aisle" in z or "consult" in z:
            return "skincare"
        if "cosmetic" in z:
            return "cosmetics"
        if "checkout" in z or "pos" in z:
            return "checkout"
        return None

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)