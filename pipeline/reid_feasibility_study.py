"""
Re-ID feasibility study - looking at ambiguity rejections from stitching

We run the benchmark events through a modified version of the stitching engine
that saves all the candidate details when it gets an ambiguity rejection.

What this does:
- Pulls out bounding box crops from the video frames
- Checks if the crops are actually usable (not full frame, decent size)
- Saves some example contact sheets so we can look at them
- Runs a pretrained Re-ID model to get appearance features
- Computes cosine similarity between source and candidates
- Figures out how many rejections Re-ID would fix
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "app"))

from app.core_logic import (
    NormalizedEvent,
    ProductionStateEngine,
    SessionRecord,
    normalize_store_id,
)

STORE_TOPOLOGY = {
    "STORE_BLR_002": {
        "CAM_ENTRY_01": {"CAM_ZONE_01", "CAM_ZONE_02"},
        "CAM_ZONE_01": {"CAM_BILLING_01"},
        "CAM_ZONE_02": {"CAM_BILLING_01"},
    },
    "ST1076": {
        "CAM_ENTRY_02": {"CAM_ZONE_03"},
        "CAM_ENTRY_03": {"CAM_ZONE_03"},
        "CAM_ZONE_03": {"CAM_BILLING_02"},
    }
}

CAMERA_BOUNDARIES = {
    # Store 1
    ("CAM_ENTRY_01", "CAM_ZONE_01"): ("x", "x", True),
    ("CAM_ENTRY_01", "CAM_ZONE_02"): ("x", "x", True),
    ("CAM_ZONE_01", "CAM_BILLING_01"): ("y", "y", True),
    ("CAM_ZONE_02", "CAM_BILLING_01"): ("y", "y", True),
    # Store 2
    ("CAM_ENTRY_02", "CAM_ZONE_03"): ("x", "x", True),
    ("CAM_ENTRY_03", "CAM_ZONE_03"): ("x", "x", True),
    ("CAM_ZONE_03", "CAM_BILLING_02"): ("y", "y", True),
}

TEMPORAL_WEIGHT = 0.40
SCALE_WEIGHT = 0.25
ASPECT_WEIGHT = 0.20
BOUNDARY_WEIGHT = 0.15
STITCH_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.15

from pipeline.detect import STORE_CAMERA_MAP


@dataclass
class CandidateRecord:
    session_visitor_id: str
    session_seq: int
    last_camera_id: str
    last_timestamp: datetime
    last_bbox: Optional[List[float]]
    heuristic_score: float
    rank: int


@dataclass
class AmbiguityRejection:
    rejection_index: int
    source_event: NormalizedEvent
    source_camera_id: str
    source_timestamp: datetime
    source_bbox: Optional[List[float]]
    candidates: List[CandidateRecord]
    score_margin: float
    store_id: str


@dataclass
class CropRecord:
    label: str
    camera_id: str
    timestamp: datetime
    bbox: List[float]
    frame_idx: int
    crop_image: Optional[np.ndarray]
    width: int = 0
    height: int = 0
    area: int = 0
    is_full_frame: bool = False
    is_usable: bool = False


class InstrumentedStitchingEngine(ProductionStateEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ambiguity_rejections: List[AmbiguityRejection] = []
        self._rejection_counter = 0
        self._identity_aliases: Dict[str, str] = {}
        self.stitch_statistics = {
            "attempts": 0,
            "accepted": 0,
            "rejections_ambiguity": 0,
            "rejections_threshold": 0,
            "average_score": 0.0,
            "total_score_sum": 0.0
        }

    def _resolve_visitor_id(self, visitor_id: str) -> str:
        canonical = self._canonical_visitor_id(visitor_id)
        visited = set()
        while canonical in self._identity_aliases:
            if canonical in visited:
                break
            visited.add(canonical)
            canonical = self._identity_aliases[canonical]
        return canonical

    def _find_matching_session(self, event: NormalizedEvent) -> Optional[SessionRecord]:
        now = event.timestamp
        canonical_id = self._canonical_visitor_id(event.visitor_id)
        resolved_id = self._resolve_visitor_id(canonical_id)

        for session in self._stores.get(event.store_id, {}).values():
            session_resolved_id = self._resolve_visitor_id(session.visitor_id)
            if resolved_id == session_resolved_id and session.is_recent_active(now, timedelta(minutes=15)):
                return session

        store_topo = STORE_TOPOLOGY.get(event.store_id, {})
        incoming_cams = {src for src, dests in store_topo.items() if event.camera_id in dests}
        if not incoming_cams:
            return None

        candidates = []
        for session in self._stores.get(event.store_id, {}).values():
            if self._resolve_visitor_id(session.visitor_id) == resolved_id:
                continue
            if not session.events:
                continue
            last_event = session.events[-1]
            if last_event.camera_id not in incoming_cams:
                continue
            dt = (event.timestamp - session.last_timestamp).total_seconds()
            if not (0.5 <= dt <= 10.0):
                continue
            session_avg_conf = sum(e.confidence for e in session.events) / len(session.events)
            if session_avg_conf < 0.40 or event.confidence < 0.40:
                continue

            S_temporal = 1.0 - (dt - 0.5) / 9.5
            prev_bbox = last_event.metadata.get("bbox")
            curr_bbox = event.metadata.get("bbox")

            if prev_bbox and curr_bbox and len(prev_bbox) == 4 and len(curr_bbox) == 4:
                area_prev = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                area_curr = (curr_bbox[2] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[1])
                S_scale = min(area_prev, area_curr) / max(area_prev, area_curr) if area_prev > 0 and area_curr > 0 else 1.0
            else:
                S_scale = 1.0

            if prev_bbox and curr_bbox and len(prev_bbox) == 4 and len(curr_bbox) == 4:
                w_prev = prev_bbox[2] - prev_bbox[0]
                h_prev = prev_bbox[3] - prev_bbox[1]
                w_curr = curr_bbox[2] - curr_bbox[0]
                h_curr = curr_bbox[3] - curr_bbox[1]
                if h_prev > 0 and h_curr > 0 and w_prev > 0 and w_curr > 0:
                    ar_prev = w_prev / h_prev
                    ar_curr = w_curr / h_curr
                    S_aspect = min(ar_prev, ar_curr) / max(ar_prev, ar_curr)
                else:
                    S_aspect = 1.0
            else:
                S_aspect = 1.0

            S_boundary = 1.0
            if (last_event.camera_id, event.camera_id) in CAMERA_BOUNDARIES:
                src_axis, dest_axis, polarity = CAMERA_BOUNDARIES[(last_event.camera_id, event.camera_id)]
                src_val = last_event.metadata.get("x_pct" if src_axis == "x" else "y_pct")
                dest_val = event.metadata.get("x_pct" if dest_axis == "x" else "y_pct")
                if src_val is not None and dest_val is not None:
                    try:
                        src_val_f = float(src_val)
                        dest_val_f = float(dest_val)
                        if polarity:
                            S_boundary_src = src_val_f / 100.0
                            S_boundary_dest = 1.0 - (dest_val_f / 100.0)
                        else:
                            S_boundary_src = 1.0 - (src_val_f / 100.0)
                            S_boundary_dest = dest_val_f / 100.0
                        S_boundary = (S_boundary_src + S_boundary_dest) / 2.0
                    except (ValueError, TypeError):
                        pass

            score = (
                TEMPORAL_WEIGHT * S_temporal +
                SCALE_WEIGHT * S_scale +
                ASPECT_WEIGHT * S_aspect +
                BOUNDARY_WEIGHT * S_boundary
            )
            candidates.append((score, session, last_event))

        if candidates:
            self.stitch_statistics["attempts"] += 1
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_session, best_last_event = candidates[0]

            if best_score < STITCH_THRESHOLD:
                self.stitch_statistics["rejections_threshold"] += 1
                return None

            if len(candidates) > 1:
                second_best_score = candidates[1][0]
                margin = best_score - second_best_score
                if margin <= AMBIGUITY_MARGIN:
                    self.stitch_statistics["rejections_ambiguity"] += 1
                    self._rejection_counter += 1
                    candidate_records = []
                    for rank, (score, session, last_evt) in enumerate(candidates, 1):
                        candidate_records.append(CandidateRecord(
                            session_visitor_id=session.visitor_id,
                            session_seq=session.session_seq,
                            last_camera_id=last_evt.camera_id,
                            last_timestamp=last_evt.timestamp,
                            last_bbox=last_evt.metadata.get("bbox"),
                            heuristic_score=score,
                            rank=rank,
                        ))
                    self.ambiguity_rejections.append(AmbiguityRejection(
                        rejection_index=self._rejection_counter,
                        source_event=event,
                        source_camera_id=event.camera_id,
                        source_timestamp=event.timestamp,
                        source_bbox=event.metadata.get("bbox"),
                        candidates=candidate_records,
                        score_margin=margin,
                        store_id=event.store_id,
                    ))
                    return None

            self._identity_aliases[canonical_id] = best_session.visitor_id
            self.stitch_statistics["accepted"] += 1
            self.stitch_statistics["total_score_sum"] += best_score
            self.stitch_statistics["average_score"] = (
                self.stitch_statistics["total_score_sum"] / self.stitch_statistics["accepted"]
            )
            return best_session

        return None


def build_camera_video_map(data_root: Path) -> Dict[str, Tuple[Path, str]]:
    cam_map = {}
    for store_dir_name, camera_map in STORE_CAMERA_MAP.items():
        store_dir = data_root / store_dir_name
        if not store_dir.exists():
            continue
        store_id = "STORE_BLR_002" if store_dir_name == "Store 1" else "ST1076"
        for file_name, camera_id in camera_map.items():
            file_path = store_dir / file_name
            if file_path.exists():
                cam_map[camera_id] = (file_path, store_id)
    return cam_map


BASE_TIME = datetime(2026, 4, 10, 14, 0, 0, tzinfo=timezone.utc)


def timestamp_to_frame_idx(timestamp: datetime, fps: float) -> int:
    dt_seconds = (timestamp - BASE_TIME).total_seconds()
    return max(0, int(round(dt_seconds * fps)))


def extract_crop_from_video(video_path: Path, frame_idx: int, bbox: List[float]) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
    finally:
        cap.release()


def extract_all_crops(rejections: List[AmbiguityRejection], cam_map: Dict[str, Tuple[Path, str]]) -> Dict[int, Dict[str, CropRecord]]:
    cam_fps: Dict[str, float] = {}
    for cam_id, (vpath, _) in cam_map.items():
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cam_fps[cam_id] = fps if fps and fps > 0 else 25.0
        cap.release()

    all_crops: Dict[int, Dict[str, CropRecord]] = {}
    for rej in rejections:
        crops: Dict[str, CropRecord] = {}
        if rej.source_camera_id in cam_map and rej.source_bbox:
            vpath, _ = cam_map[rej.source_camera_id]
            fps = cam_fps[rej.source_camera_id]
            fidx = timestamp_to_frame_idx(rej.source_timestamp, fps)
            img = extract_crop_from_video(vpath, fidx, rej.source_bbox)
            bbox = rej.source_bbox
            bw = int(bbox[2] - bbox[0])
            bh = int(bbox[3] - bbox[1])
            is_full = bw >= 1900 and bh >= 1060
            crops["source"] = CropRecord(
                label=f"rej_{rej.rejection_index:02d}_source",
                camera_id=rej.source_camera_id,
                timestamp=rej.source_timestamp,
                bbox=rej.source_bbox,
                frame_idx=fidx,
                crop_image=img,
                width=bw,
                height=bh,
                area=bw * bh,
                is_full_frame=is_full,
                is_usable=img is not None and not is_full and bw >= 20 and bh >= 40,
            )

        for cand in rej.candidates:
            key = f"candidate_{cand.rank}"
            if cand.last_camera_id in cam_map and cand.last_bbox:
                vpath, _ = cam_map[cand.last_camera_id]
                fps = cam_fps[cand.last_camera_id]
                fidx = timestamp_to_frame_idx(cand.last_timestamp, fps)
                img = extract_crop_from_video(vpath, fidx, cand.last_bbox)
                bbox = cand.last_bbox
                bw = int(bbox[2] - bbox[0])
                bh = int(bbox[3] - bbox[1])
                is_full = bw >= 1900 and bh >= 1060
                crops[key] = CropRecord(
                    label=f"rej_{rej.rejection_index:02d}_{key}",
                    camera_id=cand.last_camera_id,
                    timestamp=cand.last_timestamp,
                    bbox=bbox,
                    frame_idx=fidx,
                    crop_image=img,
                    width=bw,
                    height=bh,
                    area=bw * bh,
                    is_full_frame=is_full,
                    is_usable=img is not None and not is_full and bw >= 20 and bh >= 40,
                )

        all_crops[rej.rejection_index] = crops

    return all_crops


def audit_crop_quality(all_crops: Dict[int, Dict[str, CropRecord]]) -> Dict[str, Any]:
    total_crops = 0
    usable_crops = 0
    full_frame_crops = 0
    null_crops = 0
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    fully_usable_rejections = 0
    partially_usable_rejections = 0
    unusable_rejections = 0
    rejection_details = []

    for rej_idx, crops in sorted(all_crops.items()):
        source = crops.get("source")
        cand_keys = sorted([k for k in crops if k.startswith("candidate_")])
        rej_usable_count = 0
        rej_total = 0

        for key in ["source"] + cand_keys:
            cr = crops.get(key)
            if cr is None:
                continue
            total_crops += 1
            rej_total += 1
            if cr.crop_image is None:
                null_crops += 1
                continue
            if cr.is_full_frame:
                full_frame_crops += 1
                continue
            if cr.is_usable:
                usable_crops += 1
                rej_usable_count += 1
                widths.append(cr.width)
                heights.append(cr.height)
                areas.append(cr.area)
                if cr.height > 0:
                    aspect_ratios.append(cr.width / cr.height)

        source_usable = source is not None and source.is_usable
        cand_usable = sum(1 for k in cand_keys if crops.get(k) and crops[k].is_usable)

        if source_usable and cand_usable >= 2:
            fully_usable_rejections += 1
            status = "FULLY_USABLE"
        elif source_usable and cand_usable >= 1:
            partially_usable_rejections += 1
            status = "PARTIALLY_USABLE"
        else:
            unusable_rejections += 1
            status = "UNUSABLE"

        rejection_details.append({
            "rejection_index": rej_idx,
            "status": status,
            "source_usable": source_usable,
            "candidates_usable": cand_usable,
            "total_candidates": len(cand_keys),
        })

    return {
        "total_crops": total_crops,
        "usable_crops": usable_crops,
        "full_frame_crops": full_frame_crops,
        "null_crops": null_crops,
        "usable_rate": usable_crops / max(total_crops, 1),
        "width_stats": {
            "min": min(widths) if widths else 0,
            "max": max(widths) if widths else 0,
            "mean": sum(widths) / len(widths) if widths else 0,
        },
        "height_stats": {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "mean": sum(heights) / len(heights) if heights else 0,
        },
        "area_stats": {
            "min": min(areas) if areas else 0,
            "max": max(areas) if areas else 0,
            "mean": sum(areas) / len(areas) if areas else 0,
        },
        "aspect_ratio_stats": {
            "min": round(min(aspect_ratios), 3) if aspect_ratios else 0,
            "max": round(max(aspect_ratios), 3) if aspect_ratios else 0,
            "mean": round(sum(aspect_ratios) / len(aspect_ratios), 3) if aspect_ratios else 0,
        },
        "fully_usable_rejections": fully_usable_rejections,
        "partially_usable_rejections": partially_usable_rejections,
        "unusable_rejections": unusable_rejections,
        "rejection_details": rejection_details,
    }


def save_contact_sheets(all_crops: Dict[int, Dict[str, CropRecord]], output_dir: Path, max_rejections: int = 22) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    cell_h = 128
    cell_w = 64
    pad = 4

    for rej_idx, crops in sorted(all_crops.items()):
        if rej_idx > max_rejections:
            break
        source = crops.get("source")
        cand_keys = sorted([k for k in crops if k.startswith("candidate_")])
        n_cols = 1 + len(cand_keys)
        sheet_w = n_cols * (cell_w + pad) + pad
        sheet_h = cell_h + 2 * pad + 30
        sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)
        col = 0
        items = [("source", source)] + [(k, crops.get(k)) for k in cand_keys]

        for label, cr in items:
            x_offset = pad + col * (cell_w + pad)
            y_offset = pad
            if cr is not None and cr.crop_image is not None and cr.is_usable:
                resized = cv2.resize(cr.crop_image, (cell_w, cell_h))
                sheet[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = resized
                cv2.rectangle(sheet, (x_offset - 1, y_offset - 1), (x_offset + cell_w, y_offset + cell_h), (0, 200, 0), 1)
            elif cr is not None and cr.crop_image is not None:
                resized = cv2.resize(cr.crop_image, (cell_w, cell_h))
                sheet[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = resized
                cv2.rectangle(sheet, (x_offset - 1, y_offset - 1), (x_offset + cell_w, y_offset + cell_h), (0, 0, 200), 1)
            else:
                cv2.rectangle(sheet, (x_offset, y_offset), (x_offset + cell_w, y_offset + cell_h), (80, 80, 80), -1)
                cv2.putText(sheet, "N/A", (x_offset + 10, y_offset + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            label_y = y_offset + cell_h + 15
            short_label = label.replace("candidate_", "C")
            if label == "source":
                short_label = "SRC"
            cv2.putText(sheet, short_label, (x_offset + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

            if label.startswith("candidate_"):
                rank = int(label.split("_")[1])
                if cr:
                    for rej_idx_inner, crops_inner in all_crops.items():
                        if rej_idx_inner == rej_idx:
                            break
                cv2.putText(sheet, f"R{rank}", (x_offset + 5, label_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 255), 1)
            col += 1

        path = output_dir / f"contact_sheet_rej_{rej_idx:02d}.png"
        cv2.imwrite(str(path), sheet)
        saved_paths.append(path)

    if saved_paths:
        overview_rows = []
        for p in saved_paths:
            img = cv2.imread(str(p))
            if img is not None:
                overview_rows.append(img)
        if overview_rows:
            max_w = max(r.shape[1] for r in overview_rows)
            padded = []
            for r in overview_rows:
                if r.shape[1] < max_w:
                    extra = np.full((r.shape[0], max_w - r.shape[1], 3), 40, dtype=np.uint8)
                    r = np.hstack([r, extra])
                padded.append(r)
            overview = np.vstack(padded)
            overview_path = output_dir / "contact_sheet_overview.png"
            cv2.imwrite(str(overview_path), overview)
            saved_paths.append(overview_path)

    return saved_paths


def load_reid_model():
    import torch
    try:
        from torchreid.utils import FeatureExtractor
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device='cpu',
        )
        print("[Re-ID] Loaded OSNet via torchreid")
        return extractor, "torchreid", 512
    except ImportError:
        pass

    print("[Re-ID] torchreid not available, using torchvision ResNet50 feature extractor")
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T

    weights = models.ResNet50_Weights.DEFAULT
    base = models.resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(base.children())[:-1])
    feature_extractor.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return feature_extractor, transform, 2048


def extract_embedding(model, preprocess, model_type: str, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    import torch
    if model_type == "torchreid":
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = model([crop_rgb])
        return features[0].cpu().numpy()

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(crop_rgb).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class ReidAnalysisResult:
    rejection_index: int
    heuristic_margin: float
    heuristic_best_score: float
    heuristic_second_score: float
    num_candidates: int
    reid_best_similarity: float = 0.0
    reid_second_similarity: float = 0.0
    reid_margin: float = 0.0
    reid_top1_candidate_rank: int = 0
    reid_would_resolve: bool = False
    reid_top1_matches_heuristic: bool = False
    analyzable: bool = False


def run_reid_analysis(rejections: List[AmbiguityRejection], all_crops: Dict[int, Dict[str, CropRecord]], audit: Dict[str, Any]) -> List[ReidAnalysisResult]:
    usable_count = audit["fully_usable_rejections"] + audit["partially_usable_rejections"]
    if usable_count == 0:
        print("[Re-ID] No usable rejections for Re-ID analysis. Skipping embedding generation.")
        return []

    print(f"\n[Re-ID] Loading pretrained model...")
    model, preprocess, embed_dim = load_reid_model()
    model_type = "torchreid" if isinstance(preprocess, str) and preprocess == "torchreid" else "torchvision"
    print(f"[Re-ID] Model loaded. Embedding dimension: {embed_dim}")

    results = []
    for rej in rejections:
        crops = all_crops.get(rej.rejection_index, {})
        source_crop = crops.get("source")
        result = ReidAnalysisResult(
            rejection_index=rej.rejection_index,
            heuristic_margin=rej.score_margin,
            heuristic_best_score=rej.candidates[0].heuristic_score if rej.candidates else 0,
            heuristic_second_score=rej.candidates[1].heuristic_score if len(rej.candidates) > 1 else 0,
            num_candidates=len(rej.candidates),
        )

        if source_crop is None or not source_crop.is_usable:
            results.append(result)
            continue

        source_emb = extract_embedding(model, preprocess, model_type, source_crop.crop_image)
        if source_emb is None:
            results.append(result)
            continue

        cand_similarities = []
        for cand in rej.candidates:
            key = f"candidate_{cand.rank}"
            cand_crop = crops.get(key)
            if cand_crop is not None and cand_crop.is_usable:
                cand_emb = extract_embedding(model, preprocess, model_type, cand_crop.crop_image)
                if cand_emb is not None:
                    sim = cosine_similarity(source_emb, cand_emb)
                    cand_similarities.append((cand.rank, sim, cand.heuristic_score))
                    continue
            cand_similarities.append((cand.rank, None, cand.heuristic_score))

        valid_sims = [(rank, sim, hscore) for rank, sim, hscore in cand_similarities if sim is not None]
        if len(valid_sims) < 2:
            results.append(result)
            continue

        result.analyzable = True
        valid_sims.sort(key=lambda x: x[1], reverse=True)
        result.reid_best_similarity = valid_sims[0][1]
        result.reid_second_similarity = valid_sims[1][1]
        result.reid_margin = valid_sims[0][1] - valid_sims[1][1]
        result.reid_top1_candidate_rank = valid_sims[0][0]
        result.reid_top1_matches_heuristic = (valid_sims[0][0] == 1)
        result.reid_would_resolve = result.reid_margin > AMBIGUITY_MARGIN
        results.append(result)

        print(f"  Rejection {rej.rejection_index:2d}: heuristic_margin={rej.score_margin:.4f} | reid_margin={result.reid_margin:.4f} | resolve={'YES' if result.reid_would_resolve else 'NO'} | top1_match={'YES' if result.reid_top1_matches_heuristic else 'NO'}")

    return results


def generate_report(rejections: List[AmbiguityRejection], audit: Dict[str, Any], reid_results: List[ReidAnalysisResult], contact_sheets: List[Path], model_type: str, output_path: Path):
    lines = []
    lines.append("# Re-ID Feasibility Study Report")
    lines.append("")
    lines.append("This report evaluates whether appearance-based Re-Identification (Re-ID) embeddings")
    lines.append("can resolve the ambiguity-rejected stitching transitions that the current spatial-temporal")
    lines.append("heuristic cannot disambiguate.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Ambiguity Rejection Summary")
    lines.append("")
    lines.append(f"The instrumented stitching replay of {len(rejections)} benchmark events identified")
    lines.append(f"**{len(rejections)} ambiguity rejections** out of the total stitching attempts.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :---: |")
    lines.append(f"| Total Ambiguity Rejections | {len(rejections)} |")
    avg_margin = sum(r.score_margin for r in rejections) / len(rejections) if rejections else 0
    lines.append(f"| Average Heuristic Score Margin | {avg_margin:.4f} |")
    avg_cands = sum(len(r.candidates) for r in rejections) / len(rejections) if rejections else 0
    lines.append(f"| Average Candidates Per Rejection | {avg_cands:.1f} |")
    lines.append("")
    lines.append("## 2. Bounding-Box Crop Quality Audit")
    lines.append("")
    lines.append("Before generating Re-ID embeddings, we audited the bounding-box crops from the benchmark")
    lines.append("events to determine if they contain sufficient visual information for appearance matching.")
    lines.append("")
    lines.append("### 2.1 Overall Crop Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :---: |")
    lines.append(f"| Total Crops Extracted | {audit['total_crops']} |")
    lines.append(f"| Usable Crops | {audit['usable_crops']} ({audit['usable_rate']*100:.1f}%) |")
    lines.append(f"| Full-Frame (Unusable) | {audit['full_frame_crops']} |")
    lines.append(f"| Null/Failed Extraction | {audit['null_crops']} |")
    lines.append("")
    lines.append("### 2.2 Usable Crop Dimensions")
    lines.append("")
    lines.append("| Dimension | Min | Max | Mean |")
    lines.append("| :--- | :---: | :---: | :---: |")
    ws = audit["width_stats"]
    lines.append(f"| Width (px) | {ws['min']} | {ws['max']} | {ws['mean']:.0f} |")
    hs = audit["height_stats"]
    lines.append(f"| Height (px) | {hs['min']} | {hs['max']} | {hs['mean']:.0f} |")
    ar = audit["aspect_ratio_stats"]
    lines.append(f"| Aspect Ratio (W/H) | {ar['min']:.3f} | {ar['max']:.3f} | {ar['mean']:.3f} |")
    aa = audit["area_stats"]
    lines.append(f"| Area (px²) | {aa['min']:,} | {aa['max']:,} | {aa['mean']:,.0f} |")
    lines.append("")
    lines.append("### 2.3 Per-Rejection Usability")
    lines.append("")
    lines.append("| Usability Status | Count | Description |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Fully Usable** | {audit['fully_usable_rejections']} | Source + ≥2 candidate crops are person-level |")
    lines.append(f"| **Partially Usable** | {audit['partially_usable_rejections']} | Source + 1 candidate usable |")
    lines.append(f"| **Unusable** | {audit['unusable_rejections']} | Source or all candidates are full-frame/failed |")
    lines.append("")
    lines.append("### 2.4 Rejection-Level Crop Audit Detail")
    lines.append("")
    lines.append("| Rejection | Status | Source Usable | Candidates Usable / Total |")
    lines.append("| :---: | :--- | :---: | :---: |")
    for d in audit["rejection_details"]:
        lines.append(f"| {d['rejection_index']} | {d['status']} | {'YES' if d['source_usable'] else 'NO'} | {d['candidates_usable']} / {d['total_candidates']} |")
    lines.append("")
    if contact_sheets:
        lines.append("### 2.5 Representative Contact Sheets")
        lines.append("")
        lines.append("Contact sheet images have been saved to `data/reid_crops/` for visual inspection.")
        lines.append("Each sheet shows `[Source | Candidate 1 | Candidate 2 | ...]` with green borders")
        lines.append("for usable crops and red borders for full-frame/unusable crops.")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Re-ID Embedding Analysis")
    lines.append("")
    analyzable = [r for r in reid_results if r.analyzable]
    not_analyzable = [r for r in reid_results if not r.analyzable]

    if not reid_results:
        lines.append("> [!WARNING]")
        lines.append("> No rejections had sufficient usable crops for Re-ID analysis.")
        lines.append("> The crop quality gate blocked embedding generation.")
    else:
        lines.append(f"**Model**: {model_type} (pretrained, CPU inference)")
        lines.append("")
        lines.append(f"Out of {len(rejections)} ambiguity rejections:")
        lines.append(f"- **{len(analyzable)}** had sufficient usable crops for Re-ID analysis")
        lines.append(f"- **{len(not_analyzable)}** were not analyzable (missing source or candidate crops)")
        lines.append("")

        if analyzable:
            lines.append("### 3.1 Cosine Similarity Margin Distribution")
            lines.append("")
            lines.append("| Rejection | Heuristic Margin | Re-ID Margin | Re-ID Best Sim | Re-ID 2nd Sim | Would Resolve? |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
            for r in analyzable:
                resolve = "YES" if r.reid_would_resolve else "NO"
                lines.append(f"| {r.rejection_index} | {r.heuristic_margin:.4f} | {r.reid_margin:.4f} | {r.reid_best_similarity:.4f} | {r.reid_second_similarity:.4f} | {resolve} |")
            lines.append("")

            reid_margins = [r.reid_margin for r in analyzable]
            heur_margins = [r.heuristic_margin for r in analyzable]

            lines.append("### 3.2 Margin Comparison Statistics")
            lines.append("")
            lines.append("| Metric | Heuristic | Re-ID (Cosine) |")
            lines.append("| :--- | :---: | :---: |")
            lines.append(f"| Mean Margin | {sum(heur_margins)/len(heur_margins):.4f} | {sum(reid_margins)/len(reid_margins):.4f} |")
            lines.append(f"| Min Margin | {min(heur_margins):.4f} | {min(reid_margins):.4f} |")
            lines.append(f"| Max Margin | {max(heur_margins):.4f} | {max(reid_margins):.4f} |")
            lines.append(f"| Std Dev | {np.std(heur_margins):.4f} | {np.std(reid_margins):.4f} |")
            lines.append("")

            lines.append("### 3.3 Top-1 Candidate Selection Analysis")
            lines.append("")
            top1_matches = sum(1 for r in analyzable if r.reid_top1_matches_heuristic)
            top1_disagrees = len(analyzable) - top1_matches
            lines.append("| Metric | Value |")
            lines.append("| :--- | :---: |")
            lines.append(f"| Total Analyzable Rejections | {len(analyzable)} |")
            lines.append(f"| Re-ID Top-1 Agrees with Heuristic Top-1 | {top1_matches} ({top1_matches/len(analyzable)*100:.1f}%) |")
            lines.append(f"| Re-ID Top-1 Disagrees with Heuristic Top-1 | {top1_disagrees} ({top1_disagrees/len(analyzable)*100:.1f}%) |")
            lines.append("")
            lines.append("| Rejection | Re-ID Top-1 Rank | Heuristic Top-1 Rank | Agreement |")
            lines.append("| :---: | :---: | :---: | :---: |")
            for r in analyzable:
                agree = "YES" if r.reid_top1_matches_heuristic else "NO"
                lines.append(f"| {r.rejection_index} | {r.reid_top1_candidate_rank} | 1 | {agree} |")
            lines.append("")

            lines.append("### 3.4 Estimated Ambiguity Resolution Rate")
            lines.append("")
            resolved = sum(1 for r in analyzable if r.reid_would_resolve)
            unresolved = len(analyzable) - resolved
            lines.append("| Metric | Value |")
            lines.append("| :--- | :---: |")
            lines.append(f"| Rejections Where Re-ID Margin > {AMBIGUITY_MARGIN} | **{resolved}** / {len(analyzable)} ({resolved/len(analyzable)*100:.1f}%) |")
            lines.append(f"| Rejections Still Ambiguous After Re-ID | {unresolved} / {len(analyzable)} ({unresolved/len(analyzable)*100:.1f}%) |")
            lines.append(f"| Estimated Resolution Rate (of all {len(rejections)} rejections) | **{resolved}** / {len(rejections)} ({resolved/len(rejections)*100:.1f}%) |")
            lines.append("")

            if resolved > 0:
                lines.append("> [!TIP]")
                lines.append(f"> Re-ID embeddings could theoretically resolve **{resolved}** of the **{len(rejections)}** ambiguity rejections ({resolved/len(rejections)*100:.1f}%), increasing the stitching acceptance rate from the current baseline.")
            else:
                lines.append("> [!WARNING]")
                lines.append("> Re-ID embeddings did not provide sufficient margin to resolve any ambiguity rejections.")
                lines.append("> This may indicate that the visual similarity between competing candidates is genuinely high,")
                lines.append("> or that the crop quality (resolution, occlusion) limits appearance discrimination.")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Conclusions & Recommendations")
    lines.append("")

    if analyzable:
        resolved = sum(1 for r in analyzable if r.reid_would_resolve)
        total_rej = len(rejections)
        lines.append("### Key Findings")
        lines.append("")
        lines.append(f"1. **Crop Quality**: {audit['usable_rate']*100:.1f}% of extracted bounding-box crops are person-level and suitable for appearance embedding. {audit['fully_usable_rejections']} of {total_rej} rejections are fully analyzable with Re-ID.")
        lines.append(f"2. **Candidate Separation**: Re-ID cosine similarity margins (mean={sum(reid_margins)/len(reid_margins):.4f}) compared to heuristic score margins (mean={sum(heur_margins)/len(heur_margins):.4f}).")
        lines.append(f"3. **Resolution Rate**: {resolved}/{len(analyzable)} analyzable rejections ({resolved/max(len(analyzable),1)*100:.1f}%) would be resolved by Re-ID, representing {resolved}/{total_rej} ({resolved/total_rej*100:.1f}%) of all ambiguity rejections.")
        lines.append(f"4. **Top-1 Agreement**: Re-ID agrees with the heuristic's top-1 candidate in {top1_matches}/{len(analyzable)} ({top1_matches/len(analyzable)*100:.1f}%) of cases.")
        lines.append("")
        lines.append("### Recommendations")
        lines.append("")
        if resolved / max(len(analyzable), 1) >= 0.3:
            lines.append("- **Re-ID integration is justified**: A meaningful fraction of ambiguity rejections would be resolved.")
            lines.append("- Recommended model: OSNet-AIN (lightweight, ~2.2M parameters, designed for person Re-ID).")
            lines.append("- Integration point: Add Re-ID scoring as a tiebreaker component in `_find_matching_session` when the heuristic margin falls below the ambiguity threshold.")
        else:
            lines.append("- **Re-ID integration has limited value at current scale**: The resolution rate is low.")
            lines.append("- This may improve with better detection crops (YOLO instead of MOG2 fallback) or higher resolution video.")
            lines.append("- Consider prioritizing camera calibration and topology-aware features over Re-ID.")
    else:
        lines.append("The crop quality audit determined that insufficient crops were available for Re-ID analysis.")
        lines.append("This is likely due to the MOG2 background subtraction fallback producing full-frame bounding boxes.")
        lines.append("Re-ID feasibility should be re-evaluated once real YOLO detections with proper bounding boxes are available.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This report was generated as an offline feasibility study. No production pipeline modifications were made.*")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Report] Saved to: {output_path}")


def main():
    print("   RE-ID FEASIBILITY STUDY: AMBIGUITY-REJECTED TRANSITIONS")
    data_root = WORKSPACE / "data"
    events_path = data_root / "benchmark_events.json"
    txn_path = data_root / "POS_transactions.csv"
    crops_dir = data_root / "reid_crops"
    report_path = WORKSPACE / "docs" / "reid_feasibility_report.md"

    print("\n[Step 1] Loading benchmark events...")
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"  Loaded {len(events)} events")

    events_sorted = sorted(
        events,
        key=lambda e: (
            e.get("timestamp", ""),
            e.get("camera_id", ""),
            e.get("visitor_id", ""),
            e.get("event_type", ""),
        ),
    )

    print("\n[Step 2] Replaying events through instrumented stitching engine...")
    engine = InstrumentedStitchingEngine()
    engine.load_transactions(str(txn_path))
    engine.ingest_events(events_sorted)

    print(f"  Stitching attempts: {engine.stitch_statistics['attempts']}")
    print(f"  Accepted: {engine.stitch_statistics['accepted']}")
    print(f"  Ambiguity rejections: {engine.stitch_statistics['rejections_ambiguity']}")
    print(f"  Threshold rejections: {engine.stitch_statistics['rejections_threshold']}")
    print(f"  Captured rejection details: {len(engine.ambiguity_rejections)}")

    rejections = engine.ambiguity_rejections
    if not rejections:
        print("\n[WARN] No ambiguity rejections captured. Nothing to analyze.")
        return

    print(f"\n[Step 3] Extracting bounding-box crops from video frames...")
    cam_map = build_camera_video_map(data_root)
    print(f"  Camera-video map: {list(cam_map.keys())}")
    all_crops = extract_all_crops(rejections, cam_map)
    print(f"  Extracted crops for {len(all_crops)} rejections")

    print(f"\n[Step 4] Auditing crop quality...")
    audit = audit_crop_quality(all_crops)
    print(f"  Total crops: {audit['total_crops']}")
    print(f"  Usable crops: {audit['usable_crops']} ({audit['usable_rate']*100:.1f}%)")
    print(f"  Full-frame: {audit['full_frame_crops']}")
    print(f"  Null/failed: {audit['null_crops']}")
    print(f"  Fully usable rejections: {audit['fully_usable_rejections']}")
    print(f"  Partially usable: {audit['partially_usable_rejections']}")
    print(f"  Unusable: {audit['unusable_rejections']}")

    print(f"\n[Step 5] Saving contact sheets to {crops_dir}...")
    contact_sheets = save_contact_sheets(all_crops, crops_dir)
    print(f"  Saved {len(contact_sheets)} contact sheet images")

    usable_for_reid = audit["fully_usable_rejections"] + audit["partially_usable_rejections"]
    print(f"\n[Step 6] Re-ID embedding analysis...")
    print(f"  Rejections with usable crops: {usable_for_reid}")
    reid_results = []
    model_type = "none"

    if usable_for_reid > 0:
        print(f"  Crop quality gate: PASSED ({usable_for_reid} usable rejections)")
        reid_results = run_reid_analysis(rejections, all_crops, audit)
        try:
            import torchreid
            model_type = "OSNet x1.0 (torchreid, pretrained on Market-1501 + MSMT17)"
        except ImportError:
            model_type = "ResNet-50 (torchvision, pretrained on ImageNet)"
    else:
        print(f"  Crop quality gate: FAILED (0 usable rejections)")
        print(f"  Skipping Re-ID embedding generation.")

    print(f"\n[Step 7] Generating feasibility report...")
    generate_report(rejections, audit, reid_results, contact_sheets, model_type, report_path)

    print("\n" + "=" * 70)
    print("   FEASIBILITY STUDY COMPLETE")
    analyzable = [r for r in reid_results if r.analyzable]
    if analyzable:
        resolved = sum(1 for r in analyzable if r.reid_would_resolve)
        print(f"  Analyzable rejections: {len(analyzable)} / {len(rejections)}")
        print(f"  Would resolve with Re-ID: {resolved} / {len(analyzable)}")
        print(f"  Estimated resolution rate: {resolved}/{len(rejections)} ({resolved/len(rejections)*100:.1f}%)")
    else:
        print(f"  No rejections were analyzable with Re-ID.")
    print(f"\n  Report: {report_path}")
    print(f"  Contact sheets: {crops_dir}")


if __name__ == "__main__":
    main()