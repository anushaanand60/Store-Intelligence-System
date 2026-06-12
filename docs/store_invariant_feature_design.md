# Store-Invariant Behavioral Features - What Actually Works

When we tried running our ML model on a different store, it failed pretty badly. The model was trained on `STORE_BLR_002` (where 71% of sessions ended in a purchase) but when we tested it on `ST1076` (where zero people bought anything), it was predicting conversion probabilities as high as 68.65% for shoppers who clearly didn't buy.

We thought normalizing by store averages would help, but it actually made things worse. Logistic regression false positives jumped from 41.8% to 55.6%. The reason: stores with higher conversion rates naturally have longer average dwell times because the people who buy tend to stick around longer. When you divide by that average, you're accidentally feeding the model information about conversion rates through the back door.

Beyond that, stores just aren't built the same. Camera heights vary, grid sizes differ, some stores have way more zones than others. Raw numbers like "walked 15% across the frame" or "visited 8 grid cells" don't mean the same thing from one store to the next.

This document explains how to transform our features so they work across different store layouts without needing to retrain the model.

---

## 1. What's Wrong With Our Current Features

We have 11 features in `SessionRecord.to_ml_features()`. Some are fine across stores, some aren't.

| Feature | Problem | Can we fix? |
| :--- | :--- | :--- |
| `billing_reached` | None, binary flag is fine | Yes, keep as is |
| `path_entropy` | Depends on how many grid nodes the store has | Probably, need to normalize |
| `revisit_count` | Long paths naturally have more revisits | Probably |
| `path_length` | Store with more cameras = longer path length | Probably |
| `total_dwell_time_ms` | Bigger stores take longer to walk through. Also conversion rate skews averages | Definitely |
| `unique_grids_visited` | Store with 100 grids vs 10 grids - same number means different things | Definitely |
| `average_movement_distance` | Camera height and FOV completely change what 10% movement means | Definitely |
| `average_detection_confidence` | Lighting and camera quality vary by store | Maybe |
| `minimum_detection_confidence` | Same as above | Maybe |
| `maximum_detection_confidence` | Same as above | Maybe |
| `low_confidence_event_ratio` | Same as above | Maybe |

---

## 2. What To Do Instead

### 2.1 Total Dwell Time

**Why it breaks**: A supermarket takes minutes to walk through. A small kiosk takes seconds. Worse, if you normalize by the store's average dwell time, you're basically dividing by a number that already includes conversion signal (buyers dwell longer, so their store average is higher).

**Fix A: Percentile rank within store**
```python
dwell_percentile = rank_of_session_in_store / total_sessions_in_store
```
Maps everything to [0,1]. 90th percentile in small store = 90th percentile in big store.

*Needs*: Historical dwell times for the store.

**Fix B: Normalize by non-converters only**
```python
dwell_ratio = session_dwell / avg_dwell_of_non_buyers_in_this_store
```
Divides by people who just walked through and didn't buy. This removes the conversion rate bias.

*Needs*: Historical sessions with conversion labels from POS data.

---

### 2.2 Unique Grids Visited

**Why it breaks**: Store A has 10 total grids. Visiting 8 means you saw almost everything. Store B has 100 grids. Visiting 8 means you barely walked anywhere.

**Fix: Exploration ratio**
```python
exploration_ratio = unique_grids_visited / total_grids_in_store
```

*Needs*: Total grid count from `STORE_TOPOLOGY`.

---

### 2.3 Average Movement Distance

**Why it breaks**: A 10% movement in pixels means different physical distances depending on camera height, lens, and how far the person is from the camera. Two cameras mounted at different heights will give completely different numbers for the same walking speed.

**Fix A: Physical velocity (m/s) using homography**
Project the percentage coordinates onto a real ground plane using a homography matrix for each camera, then compute meters per second.

*Needs*: Camera calibration homography matrices. This is a pain to set up but would fix the problem completely.

**Fix B: Percentile rank within store**
Rank the raw movement distance compared to other sessions in the same store.

*Needs*: Historical movement distances for the store. Less accurate than physical velocity but easier to implement.

---

### 2.4 Path Length

**Why it breaks**: More cameras = more grid cells = longer path sequences. A shopper in a store with dense camera coverage will have a longer path length even if they take the same physical route.

**Fix: Path length ratio**
```python
path_length_ratio = path_length / total_grids_in_store
```

*Needs*: Total grid count.

---

### 2.5 Revisit Count

**Why it breaks**: Stores with corridors force backtracking. Open floor plans don't. Also longer paths naturally have more revisits.

**Fix: Revisit ratio**
```python
revisit_ratio = revisit_count / path_length
```
Measures how often you backtrack per step. Removes path length from the equation.

*Needs*: Nothing extra.

---

### 2.6 Path Entropy

**Why it breaks**: The maximum possible entropy depends on how many unique grids you could possibly visit. Larger stores have higher theoretical max entropy.

**Fix: Normalized entropy**
```python
if unique_grids_visited > 1:
    normalized_entropy = shannon_entropy / log2(unique_grids_visited)
else:
    normalized_entropy = 0
```
This scales entropy to [0,1] based on how many unique grids you actually saw, not how many exist in the whole store.

*Needs*: Nothing extra.

---

### 2.7 Detection Confidence

**Why it breaks**: Different lighting, different cameras. A model trained on well-lit store expects high confidence. Darker store gives lower confidence and model might think something is wrong.

**Fix: Z-score per camera**
```python
zscore_confidence = (confidence - mean_confidence_for_this_camera) / std_confidence_for_this_camera
```
Standardizes each camera to mean 0, variance 1.

*Needs*: Rolling mean and std for confidence scores per camera, updated regularly.

---

## 3. What The New Pipeline Looks Like

```
Raw events (x_pct, y_pct, timestamps, camera_id, confidence)
    |
    v
[Calibration data and stats]
- Store grid counts
- Non-converter dwell averages
- Camera homography (optional)
- Per-camera confidence stats
    |
    v
Compute relative features:
- exploration_ratio
- dwell_ratio_baseline
- revisit_ratio
- normalized_path_entropy
- path_length_ratio
- physical_velocity_m_s (if calibrated)
- zscore_confidence_avg
- billing_reached (unchanged)
    |
    v
Train models on these instead of absolute numbers
```

### New feature set

| Feature | Formula | Range |
| :--- | :--- | :---: |
| `billing_reached` | Binary | {0, 1} |
| `exploration_ratio` | unique_grids / G_total | [0, 1] |
| `dwell_ratio_baseline` | dwell_time / avg_dwell_non_converters | [0, ∞) |
| `revisit_ratio` | revisits / path_length | [0, 1) |
| `normalized_path_entropy` | H / log2(unique_grids) | [0, 1] |
| `path_length_ratio` | path_length / G_total | [0, ∞) |
| `physical_velocity_m_s` | meters per second from homography | [0, ~5] |
| `zscore_confidence_avg` | (confidence - mu)/sigma per camera | (-∞, ∞) |

---

## 4. What To Implement First

Ranked by how easy vs how much benefit:

| Rank | Feature | Effort | Benefit | What you need |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `revisit_ratio` | Trivial | High | Nothing |
| 2 | `normalized_path_entropy` | Trivial | Medium | Nothing |
| 3 | `exploration_ratio` | Trivial | High | Total grid count per store |
| 4 | `path_length_ratio` | Trivial | Medium | Total grid count per store |
| 5 | `dwell_ratio_baseline` | Medium | Very high | Non-converter averages |
| 6 | `zscore_confidence_avg` | Medium | High | Rolling per-camera stats |
| 7 | `physical_velocity_m_s` | Hard | Very high | Homography calibration |

Start with the trivial ones first (ranks 1-4). They'll get us most of the way with almost no work.

---

## 5. Extra Data We'll Need

To actually implement all of this, the system needs to know:

**Store metadata** - A config file per store:
```json
{
  "STORE_BLR_002": {
    "total_grids": 16,
    "baseline_non_converter_dwell_ms": 1950.0
  },
  "ST1076": {
    "total_grids": 12,
    "baseline_non_converter_dwell_ms": 1030.0
  }
}
```

**Camera calibration (if we do homography)** - A 3x3 matrix per camera to map pixels to ground plane meters:
```json
{
  "CAM_ZONE_01": {
    "homography_matrix": [
      [0.124, -0.0451, 12.4],
      [0.00311, 0.0988, -4.2],
      [-0.0000112, 0.000254, 1.0]
    ]
  }
}
```

**Rolling stats database** - Keep running means and stds for:
- Confidence scores per camera (updated daily)
- Non-converter dwell times per store (computed from POS-matched sessions)