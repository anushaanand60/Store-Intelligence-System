import os
from pathlib import Path

from pipeline.detect import discover_video_assets
from pipeline.tracker import GhostVelocityCache, SpatialHashTracker


def test_tracker_preserves_spatial_hash_matching():
    tracker = SpatialHashTracker()
    first = tracker.update_and_match([{"bbox": [10.0, 10.0, 50.0, 60.0]}])
    second = tracker.update_and_match([{"bbox": [12.0, 12.0, 52.0, 62.0]}])
    assert first[0]["track_id"] == second[0]["track_id"]
    assert tracker.get_active_count() == 1


def test_ghost_cache_expires_after_45_frames():
    cache = GhostVelocityCache(max_frames=45)
    cache.add(1, [0.0, 0.0, 10.0, 10.0], (1.0, 1.0))
    for _ in range(44):
        assert 1 in cache.step()
    assert 1 not in cache.step()


def test_video_discovery_finds_local_store_assets():
    data_root = Path(os.getcwd()) / "data"
    assets = discover_video_assets(data_root)
    assert len(assets) >= 8
    assert any("Store 1" in str(path) for _, _, path in assets)
    assert any("Store 2" in str(path) for _, _, path in assets)
