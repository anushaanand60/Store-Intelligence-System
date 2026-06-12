# Purplle Video Assets - Inventory and Dataset Scale Analysis

We went through all 8 video files (4 per store) in `data/Store 1` and `data/Store 2` frame by frame. Here's the analysis.

---

## 1. All the Videos

### Store 1 (`STORE_BLR_002`)

| File Name | Camera ID | Resolution | FPS | Frames | Duration | Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CAM 1 - zone.mp4` | `CAM_ZONE_01` | 1920x1080 | 29.97 | 4,193 | 139.91s (2.33 min) | 171.92 MB |
| `CAM 2 - zone.mp4` | `CAM_ZONE_02` | 1920x1080 | 29.97 | 3,774 | 125.93s (2.10 min) | 154.71 MB |
| `CAM 3 - entry.mp4` | `CAM_ENTRY_01` | 1920x1080 | 29.97 | 4,436 | 148.01s (2.47 min) | 182.00 MB |
| `CAM 5 - billing.mp4` | `CAM_BILLING_01` | 1920x1080 | 25.00 | 3,465 | 138.60s (2.31 min) | 69.87 MB |
| **Total Store 1** | - | - | - | **15,868** | **552.45s (9.21 min)** | **578.50 MB** |

### Store 2 (`ST1076`)

| File Name | Camera ID | Resolution | FPS | Frames | Duration | Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `billing_area.mp4` | `CAM_BILLING_02` | 960x1080 | 25.00 | 3,126 | 125.04s (2.08 min) | 47.10 MB |
| `entry 1.mp4` | `CAM_ENTRY_02` | 960x1080 | 25.00 | 2,636 | 105.44s (1.76 min) | 26.12 MB |
| `entry 2.mp4` | `CAM_ENTRY_03` | 960x1080 | 25.00 | 2,129 | 85.16s (1.42 min) | 39.44 MB |
| `zone.mp4` | `CAM_ZONE_03` | 960x1080 | 25.00 | 2,898 | 115.92s (1.93 min) | 48.71 MB |
| **Total Store 2** | - | - | - | **10,789** | **431.56s (7.19 min)** | **161.37 MB** |

### Totals Across Both Stores

- **Video files**: 8
- **Total frames**: 26,657
- **Total duration**: 984.01 seconds (16.40 minutes)
- **Total size**: 739.88 MB

---

## 2. Benchmark vs Full Footage

The 58-session dataset we've been using (`ml_features_real.csv`) came from the **benchmark run** where we only processed the first 300 frames of each camera. That was just to test parallelization and get early models running.

Here's how much of the full footage that benchmark actually used:

| Metric | Benchmark (`MAX_FRAMES=300`) | Full Run (`MAX_FRAMES=0`) | Benchmark % of Full |
| :--- | :---: | :---: | :---: |
| **Total frames** | 2,400 | 26,657 | **9.00%** |
| **Total duration** | 90.03s (1.50 min) | 984.01s (16.40 min) | **9.15%** |
| **Frames processed (stride=5)** | 480 | 5,331 | **9.00%** |

### How we got 90 seconds for the benchmark

- Store 1: 3 cameras at 29.97 fps + 1 camera at 25.00 fps. 300 frames = about 10 seconds for the 29.97 cams, 12 seconds for the 25 fps cam.
- Store 2: 4 cameras at 25.00 fps. 300 frames = 12 seconds each.

Added up: (3 × 10.01) + (5 × 12.00) = 90.03 seconds.

---

## 3. Sessions We Got From Running Everything

We ran the full pipeline on all frames and matched tracking sessions against POS data.

### Session Counts

| Store | Benchmark Sessions | Full Run Sessions | How many times bigger |
| :--- | :---: | :---: | :---: |
| **Store 1** | 37 | 252 | 6.81x |
| **Store 2** | 21 | 134 | 6.38x |
| **Total** | **58** | **386** | **6.66x** |

### Class Balance in the Full Dataset

**Store 1:**
- 252 total sessions
- 188 conversions (74.60%)
- 64 non-conversions (25.40%)

**Store 2:**
- 134 total sessions
- 0 conversions (0.00%)
- 134 non-conversions (100.00%)

**Combined:**
- 386 total sessions
- 188 conversions (48.70%)
- 198 non-conversions (51.30%)

### Why Linear Extrapolation Would Have Been Wrong

If we just scaled up by frames (58 × (26657/2400)), we'd expect about 644 sessions. But we only got 386. Here's why:

1. **Benchmark had fragmented tracks** - In a 10-second slice, someone moving across zones might get split into multiple tracking IDs because their full trajectory is incomplete. That artificially inflates session counts.

2. **Full videos let the engine stitch correctly** - When we process everything, the state engine can connect trajectories across cameras properly. Multiple fragmented tracks from the benchmark become one real session.

3. **Full dwell times get captured** - Shoppers arrive, browse, and leave over several minutes. The full video catches the whole thing, not just a slice.

---

## 4. What We Can Actually Do With This

### Maximum dataset size right now

From the videos we have, the absolute maximum is **386 sessions** (252 from Store 1, 134 from Store 2).

> **Big problem: Store 2 has zero conversions**
> 
> Across all 7.19 minutes of Store 2 footage, not a single person bought anything. So any model trained on Store 1 (74.6% conversion) will not transfer to Store 2. Before training cross-store models, we need invariant features like `revisit_ratio` and `dwell_percentile` that don't depend on absolute numbers.

### How to get more data

To train models that actually generalize, we need over 1,000 sessions. At the current shopper density:

- **Store 1**: Need about 40 minutes of video
- **Store 2**: Need about 35 minutes of video

Also Store 2 really needs footage from times when people are actually buying. Right now with 0% conversion, we can't train anything useful for that store.

**Short term fix**: Use the invariant features we talked about (normalized entropy, revisit ratio, dwell percentiles) instead of raw dwell times and absolute coordinates. That way store layout differences don't break the model.