import logging
import math
import json
from collections import defaultdict, deque

logger = logging.getLogger("store_intelligence.behavior")

class StoreIntelligenceBehaviorEngine:
    def __init__(self, zone_map):
        self.zone_map = zone_map
        self.track_state = {}
        self.ghost_tracks = {}
        self.staff_registry = set()
        self.social_groups = {}
        self.zone_occupancy = defaultdict(set)
        self.funnel_stage_sets = {"entrance": set(), "skincare": set(), "cosmetics": set(), "checkout": set()}
        self.total_entries = 0
        self.total_exits = 0
        self.live_occupancy = 0

    def process_track_frame(self, camera_id, track_id, bbox, current_time):
        center = self._center(bbox)
        zone_name = self._resolve_zone(camera_id, center)
        stage = self._resolve_stage(camera_id, zone_name)

        state = self.track_state.get(track_id)
        if state is None:
            state = {
                "last_seen_camera": camera_id,
                "last_bbox": bbox,
                "center": center,
                "last_seen_ts": current_time,
                "velocity_window": deque(maxlen=12),
                "stages": set(),
                "zones": set(),
                "in_store": False,
            }
            self.track_state[track_id] = state
        else:
            vx, vy = center[0] - state["center"][0], center[1] - state["center"][1]
            state["velocity_window"].append((vx, vy))
            state["last_bbox"] = bbox
            state["center"] = center
            state["last_seen_ts"] = current_time
            state["last_seen_camera"] = camera_id

        if self._is_staff(track_id, camera_id, center):
            return "STAFF"

        if zone_name:
            state["zones"].add(zone_name)
            self.zone_occupancy[zone_name].add(track_id)

        if stage:
            state["stages"].add(stage)
            self.funnel_stage_sets[stage].add(track_id)

        if stage == "entrance" and not state["in_store"]:
            state["in_store"] = True
            self.total_entries += 1
            self.live_occupancy = max(self.live_occupancy + 1, 0)
        elif stage == "checkout" and state["in_store"]:
            self.total_exits += 1

        self._group_cluster(track_id, center)
        if track_id in self.ghost_tracks:
            del self.ghost_tracks[track_id]
        return "CUSTOMER"

    def handle_dropped_tracks(self, active_track_ids, camera_id):
        active = set(active_track_ids)
        for track_id, state in list(self.track_state.items()):
            if track_id in active:
                continue
            if state["last_seen_camera"] != camera_id:
                continue
            if track_id not in self.ghost_tracks:
                vx, vy = self._rolling_velocity(state)
                self.ghost_tracks[track_id] = {
                    "ttl": 45,
                    "velocity": (vx, vy),
                    "last_bbox": list(state["last_bbox"]),
                    "camera_id": camera_id,
                }
            ghost = self.ghost_tracks[track_id]
            ghost["ttl"] -= 1
            if ghost["ttl"] <= 0:
                if state.get("in_store", False):
                    state["in_store"] = False
                    self.total_exits += 1
                    self.live_occupancy = max(self.live_occupancy - 1, 0)
                del self.ghost_tracks[track_id]
                del self.track_state[track_id]
                continue
            x1, y1, x2, y2 = ghost["last_bbox"]
            vx, vy = ghost["velocity"]
            ghost["last_bbox"] = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def get_metrics_snapshot(self):
        live_occupancy = max(self.live_occupancy, 0)
        total_entries = max(self.total_entries, 0)
        total_exits = max(self.total_exits, 0)
        if total_exits > total_entries:
            total_exits = max(0, total_entries - live_occupancy)
            self.total_exits = total_exits
        active_staff_id_set = self.staff_registry
        hourly_staff_counts = len(active_staff_id_set)
        return {
            "live_occupancy": live_occupancy,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "hourly_staff_counts": hourly_staff_counts,
        }

    def get_funnel_snapshot(self):
        stages = ["entrance", "skincare", "cosmetics", "checkout"]
        labels = {
            "entrance": "Entrance (CAM 3)",
            "skincare": "Skincare/Aisles (CAM 1/4)",
            "cosmetics": "Cosmetics Area (CAM 2)",
            "checkout": "Checkout Terminal (CAM 5)",
        }
        base = max(len(self.funnel_stage_sets["entrance"]), 1)
        payload = []
        prev = base
        for stage in stages:
            count = len(self.funnel_stage_sets[stage]) if stage != "entrance" else base
            count = min(count, prev)
            drop = round((1 - (count / base)) * 100, 2) if base else 0.0
            payload.append({"zone": labels[stage], "reached_milestone_count": count, "drop_off_percentage": drop})
            prev = count
        return payload

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

    def _is_staff(self, track_id, camera_id, center):
        if track_id in self.staff_registry:
            return True
        for zone in self.zone_map.get(camera_id, []):
            if not zone.get("is_pos"):
                continue
            x1, y1, x2, y2 = zone["bbox"]
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                self.staff_registry.add(track_id)
                logger.info(json.dumps({"event": "staff_blacklisted", "track_id": track_id, "camera_id": camera_id}))
                return True
        return False

    def _group_cluster(self, track_id, center):
        for other_id, other in self.track_state.items():
            if other_id == track_id:
                continue
            distance = math.dist(center, other["center"])
            if distance > 140:
                continue
            v1 = self._rolling_velocity(self.track_state[track_id])
            v2 = self._rolling_velocity(other)
            if self._velocity_alignment(v1, v2) < 0.6:
                continue
            gid = f"group_{min(track_id, other_id)}_{max(track_id, other_id)}"
            self.social_groups.setdefault(gid, set()).update({track_id, other_id})

    @staticmethod
    def _velocity_alignment(v1, v2):
        n1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]) + 1e-6
        n2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]) + 1e-6
        return ((v1[0] * v2[0]) + (v1[1] * v2[1])) / (n1 * n2)

    @staticmethod
    def _rolling_velocity(state):
        window = state["velocity_window"]
        if not window:
            return (0.0, 0.0)
        sx = sum(v[0] for v in window)
        sy = sum(v[1] for v in window)
        n = len(window)
        return (sx / n, sy / n)

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)