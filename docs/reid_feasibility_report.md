# Re-ID Feasibility Study Report

This report evaluates whether appearance-based Re-Identification (Re-ID) embeddings
can resolve the ambiguity-rejected stitching transitions that the current spatial-temporal
heuristic cannot disambiguate.

---

## 1. Ambiguity Rejection Summary

The instrumented stitching replay of 22 benchmark events identified
**22 ambiguity rejections** out of the total stitching attempts.

| Metric | Value |
| :--- | :---: |
| Total Ambiguity Rejections | 22 |
| Average Heuristic Score Margin | 0.0383 |
| Average Candidates Per Rejection | 10.8 |

## 2. Bounding-Box Crop Quality Audit

Before generating Re-ID embeddings, we audited the bounding-box crops from the benchmark
events to determine if they contain sufficient visual information for appearance matching.

### 2.1 Overall Crop Statistics

| Metric | Value |
| :--- | :---: |
| Total Crops Extracted | 260 |
| Usable Crops | 259 (99.6%) |
| Full-Frame (Unusable) | 0 |
| Null/Failed Extraction | 0 |

### 2.2 Usable Crop Dimensions

| Dimension | Min | Max | Mean |
| :--- | :---: | :---: | :---: |
| Width (px) | 44 | 400 | 146 |
| Height (px) | 41 | 956 | 189 |
| Aspect Ratio (W/H) | 0.158 | 7.300 | 1.211 |
| Area (px²) | 8,432 | 283,932 | 27,444 |

### 2.3 Per-Rejection Usability

| Usability Status | Count | Description |
| :--- | :---: | :--- |
| **Fully Usable** | 21 | Source + ≥2 candidate crops are person-level |
| **Partially Usable** | 0 | Source + 1 candidate usable |
| **Unusable** | 1 | Source or all candidates are full-frame/failed |

### 2.4 Rejection-Level Crop Audit Detail

| Rejection | Status | Source Usable | Candidates Usable / Total |
| :---: | :--- | :---: | :---: |
| 1 | FULLY_USABLE | YES | 2 / 2 |
| 2 | FULLY_USABLE | YES | 4 / 4 |
| 3 | FULLY_USABLE | YES | 5 / 5 |
| 4 | FULLY_USABLE | YES | 4 / 4 |
| 5 | FULLY_USABLE | YES | 4 / 4 |
| 6 | FULLY_USABLE | YES | 3 / 3 |
| 7 | FULLY_USABLE | YES | 3 / 3 |
| 8 | FULLY_USABLE | YES | 3 / 3 |
| 9 | FULLY_USABLE | YES | 3 / 3 |
| 10 | FULLY_USABLE | YES | 2 / 2 |
| 11 | FULLY_USABLE | YES | 19 / 19 |
| 12 | UNUSABLE | NO | 19 / 19 |
| 13 | FULLY_USABLE | YES | 18 / 18 |
| 14 | FULLY_USABLE | YES | 19 / 19 |
| 15 | FULLY_USABLE | YES | 19 / 19 |
| 16 | FULLY_USABLE | YES | 18 / 18 |
| 17 | FULLY_USABLE | YES | 18 / 18 |
| 18 | FULLY_USABLE | YES | 17 / 17 |
| 19 | FULLY_USABLE | YES | 16 / 16 |
| 20 | FULLY_USABLE | YES | 15 / 15 |
| 21 | FULLY_USABLE | YES | 14 / 14 |
| 22 | FULLY_USABLE | YES | 13 / 13 |

### 2.5 Representative Contact Sheets

Contact sheet images have been saved to `data/reid_crops/` for visual inspection.
Each sheet shows `[Source | Candidate 1 | Candidate 2 | ...]` with green borders
for usable crops and red borders for full-frame/unusable crops.

---

## 3. Re-ID Embedding Analysis

**Model**: ResNet-50 (torchvision, pretrained on ImageNet) (pretrained, CPU inference)

Out of 22 ambiguity rejections:
- **21** had sufficient usable crops for Re-ID analysis
- **1** were not analyzable (missing source or candidate crops)

### 3.1 Cosine Similarity Margin Distribution

| Rejection | Heuristic Margin | Re-ID Margin | Re-ID Best Sim | Re-ID 2nd Sim | Would Resolve? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0072 | 0.0175 | 0.0811 | 0.0636 | NO |
| 2 | 0.0348 | 0.1841 | 0.3072 | 0.1231 | YES |
| 3 | 0.0807 | 0.0538 | 0.2933 | 0.2396 | NO |
| 4 | 0.0324 | 0.2740 | 0.3859 | 0.1119 | YES |
| 5 | 0.0495 | 0.2973 | 0.4686 | 0.1714 | YES |
| 6 | 0.0357 | 0.1688 | 0.2892 | 0.1204 | YES |
| 7 | 0.1074 | 0.0575 | 0.4385 | 0.3810 | NO |
| 8 | 0.0323 | 0.0972 | 0.2673 | 0.1701 | NO |
| 9 | 0.1119 | 0.0809 | 0.3241 | 0.2432 | NO |
| 10 | 0.0342 | 0.0428 | 0.1455 | 0.1027 | NO |
| 11 | 0.0215 | 0.0045 | 0.3061 | 0.3016 | NO |
| 13 | 0.0181 | 0.0182 | 0.2979 | 0.2797 | NO |
| 14 | 0.0429 | 0.0257 | 0.3723 | 0.3466 | NO |
| 15 | 0.0312 | 0.0365 | 0.3157 | 0.2792 | NO |
| 16 | 0.0083 | 0.0130 | 0.4102 | 0.3972 | NO |
| 17 | 0.0019 | 0.0417 | 0.4399 | 0.3982 | NO |
| 18 | 0.0380 | 0.0492 | 0.3548 | 0.3056 | NO |
| 19 | 0.0005 | 0.0038 | 0.3310 | 0.3271 | NO |
| 20 | 0.0232 | 0.0855 | 0.4596 | 0.3741 | NO |
| 21 | 0.0661 | 0.0074 | 0.2925 | 0.2851 | NO |
| 22 | 0.0389 | 0.0844 | 0.3404 | 0.2560 | NO |

### 3.2 Margin Comparison Statistics

| Metric | Heuristic | Re-ID (Cosine) |
| :--- | :---: | :---: |
| Mean Margin | 0.0389 | 0.0783 |
| Min Margin | 0.0005 | 0.0038 |
| Max Margin | 0.1119 | 0.2973 |
| Std Dev | 0.0297 | 0.0824 |

### 3.3 Top-1 Candidate Selection Analysis

| Metric | Value |
| :--- | :---: |
| Total Analyzable Rejections | 21 |
| Re-ID Top-1 Agrees with Heuristic Top-1 | 1 (4.8%) |
| Re-ID Top-1 Disagrees with Heuristic Top-1 | 20 (95.2%) |

| Rejection | Re-ID Top-1 Rank | Heuristic Top-1 Rank | Agreement |
| :---: | :---: | :---: | :---: |
| 1 | 2 | 1 | NO |
| 2 | 4 | 1 | NO |
| 3 | 4 | 1 | NO |
| 4 | 1 | 1 | YES |
| 5 | 2 | 1 | NO |
| 6 | 2 | 1 | NO |
| 7 | 3 | 1 | NO |
| 8 | 2 | 1 | NO |
| 9 | 2 | 1 | NO |
| 10 | 2 | 1 | NO |
| 11 | 11 | 1 | NO |
| 13 | 15 | 1 | NO |
| 14 | 5 | 1 | NO |
| 15 | 4 | 1 | NO |
| 16 | 3 | 1 | NO |
| 17 | 11 | 1 | NO |
| 18 | 6 | 1 | NO |
| 19 | 3 | 1 | NO |
| 20 | 8 | 1 | NO |
| 21 | 10 | 1 | NO |
| 22 | 3 | 1 | NO |

### 3.4 Estimated Ambiguity Resolution Rate

| Metric | Value |
| :--- | :---: |
| Rejections Where Re-ID Margin > 0.15 | **4** / 21 (19.0%) |
| Rejections Still Ambiguous After Re-ID | 17 / 21 (81.0%) |
| Estimated Resolution Rate (of all 22 rejections) | **4** / 22 (18.2%) |

> [!TIP]
> Re-ID embeddings could theoretically resolve **4** of the **22** ambiguity rejections (18.2%), increasing the stitching acceptance rate from the current baseline.

---

## 4. Conclusions & Recommendations

### Key Findings

1. **Crop Quality**: 99.6% of extracted bounding-box crops are person-level and suitable for appearance embedding. 21 of 22 rejections are fully analyzable with Re-ID.
2. **Candidate Separation**: Re-ID cosine similarity margins (mean=0.0783) compared to heuristic score margins (mean=0.0389).
3. **Resolution Rate**: 4/21 analyzable rejections (19.0%) would be resolved by Re-ID, representing 4/22 (18.2%) of all ambiguity rejections.
4. **Top-1 Agreement**: Re-ID agrees with the heuristic's top-1 candidate in 1/21 (4.8%) of cases.

### Recommendations

- **Re-ID integration has limited value at current scale**: The resolution rate is low.
- This may improve with better detection crops (YOLO instead of MOG2 fallback) or higher resolution video.
- Consider prioritizing camera calibration and topology-aware features over Re-ID.

---

*This report is just an offline feasibility study. No production pipeline modifications were made.*
