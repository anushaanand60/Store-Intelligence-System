import math
from collections import defaultdict

class SpatialHashTracker:
    def __init__(self, frame_width=1920, frame_height=1080, grid_cols=10, grid_rows=10, iou_threshold=0.25):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_w = max(frame_width // grid_cols, 1)
        self.cell_h = max(frame_height // grid_rows, 1)
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_track_id = 1
        self.max_stale = 30

    def _cell(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        row = min(max(cy // self.cell_h, 0), self.grid_rows - 1)
        col = min(max(cx // self.cell_w, 0), self.grid_cols - 1)
        return row, col