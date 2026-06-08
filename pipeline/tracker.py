from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


class SpatialHashTracker:
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, grid_cols: int = 10, grid_rows: int = 10, iou_threshold: float = 0.25):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_w = max(frame_width // grid_cols, 1)
        self.cell_h = max(frame_height // grid_rows, 1)
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict[str, object]] = {}
        self.next_track_id = 1
        self.max_stale = 30

    def update_and_match(self, detections):
        bucket = defaultdict(list)
        for track_id, state in self.tracks.items():
            bucket[self._cell(state["bbox"])].append(track_id)

        matched_ids = set()
        output = []
        for det in detections:
            bbox = det["bbox"]
            cell = self._cell(bbox)
            candidates = []
            row, col = cell
            for rr in range(max(0, row - 1), min(self.grid_rows, row + 2)):
                for cc in range(max(0, col - 1), min(self.grid_cols, col + 2)):
                    candidates.extend(bucket.get((rr, cc), []))
            best_tid = None
            best_iou = self.iou_threshold
            for tid in candidates:
                if tid in matched_ids:
                    continue
                iou = self._iou(bbox, self.tracks[tid]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {"bbox": bbox, "stale": 0}
                output.append({"track_id": track_id, "bbox": bbox, "hash": f"{cell[0]}_{cell[1]}"})
            else:
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["stale"] = 0
                matched_ids.add(best_tid)
                output.append({"track_id": best_tid, "bbox": bbox, "hash": f"{cell[0]}_{cell[1]}"})

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids and not any(output_item["track_id"] == track_id for output_item in output):
                self.tracks[track_id]["stale"] += 1
                if self.tracks[track_id]["stale"] > self.max_stale:
                    del self.tracks[track_id]

        return output

    def get_active_count(self):
        return len(self.tracks)

    def _cell(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        row = min(max(cy // self.cell_h, 0), self.grid_rows - 1)
        col = min(max(cx // self.cell_w, 0), self.grid_cols - 1)
        return row, col

    @staticmethod

    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        union = area_a + area_b - inter + 1e-6
        return inter / union

@dataclass


class GhostTrack:
    bbox: List[float]
    velocity: Tuple[float, float]
    ttl: int = 45


class GhostVelocityCache:
    def __init__(self, max_frames: int = 45):
        self.max_frames = max_frames
        self._ghosts: Dict[int, GhostTrack] = {}

    def add(self, track_id: int, bbox: List[float], velocity: Tuple[float, float]) -> None:
        self._ghosts[track_id] = GhostTrack(bbox=list(bbox), velocity=velocity, ttl=self.max_frames)

    def step(self) -> Dict[int, GhostTrack]:
        for track_id in list(self._ghosts.keys()):
            ghost = self._ghosts[track_id]
            ghost.ttl -= 1
            if ghost.ttl <= 0:
                del self._ghosts[track_id]
                continue
            x1, y1, x2, y2 = ghost.bbox
            vx, vy = ghost.velocity
            ghost.bbox = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]
        return self._ghosts

    def clear(self, track_id: int) -> None:
        self._ghosts.pop(track_id, None)

    def active(self) -> Dict[int, GhostTrack]:
        return dict(self._ghosts)
