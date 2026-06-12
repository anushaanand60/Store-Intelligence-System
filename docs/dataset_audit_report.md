# Phase 3A: ML Baseline Training & Real Dataset Audit

Runs the real video visitor session data through training to see if it's actually usable.

---

## 1. Is This Dataset Any Good?

Before building prediction APIs, need to check if we have enough data and if it's stable enough for ML.

### What we got

- **Total Sessions**: 58
- **Total Conversions**: 27
- **Conversion Rate**: 46.55%
- **Class Balance**: 
    - **Conversions**: 46.55% (27 sessions)
    - **Non-conversions**: 53.45% (31 sessions)
- **Sufficiency Gate**: **FAILED** (need 100+ sessions). Got this warning:
    > `WARNING: Dataset too small for reliable model evaluation.`
    > Trained anyway to get a baseline.

### Problems to watch for

1.  **Can't generalize yet**: Bootstrap shows stability within these 58 sessions, but that doesn't mean it'll work on new stores or different days. Need multi-day data.
2.  **Overfitting risk**: With only 58 sessions, a few weird visitor tracks could mess up feature importance.
3.  **Test set is tiny**: On 75/25 split, testing on 15 sessions means one wrong prediction swings metrics a lot.
4.  **Confidence features are broken**:
    > **Confidence Feature Problem**: Avg, min, and max confidence are constant (`0.50` or `1.0`) across all events in `benchmark_events.json`. 
    >
    > *Why*: The CV pipeline falls back to MOG2 background subtraction with hardcoded 0.50 confidence when YOLO is bypassed (which happens during benchmark parallelization). Zero variance means these features give NaN correlations, so we dropped them from training.

### Bottom line

> **Not ready for production.**
> Treat these models as **exploratory baselines only**. Bootstrap shows the signal is real (not just lucky splits), but cross-store results show the model is sensitive to different store layouts.
> 
> **Main finding**: Works well within one store, but doesn't transfer to different store geometries. Need normalization (baseline-relative dwell time, coordinate scaling) before any online deployment.



---

## 2. What the Data Looks Like

Quick summary of the 58 sessions:

- **Dwell Time**: 0 ms to 11,466 ms. Converted visitors stayed way longer (mean > 5,000 ms) vs non-converters who often left immediately (many at 0 ms).
- **Path Length**: 1 step to 8 steps. Converted visitors had min 2 steps, averaged 3.8 steps (moving around). Non-converters mostly stuck to 1 or 2 grids.
- **Movement Distance**: Converted visitors moved a lot ($12.5$ to $77.8$ units). Non-converters had many zero or low-movement sessions (walk in, stand at entrance, leave).

---

## 3. Correlation & Mutual Information Rankings

Used Pearson (linear relationships) and Mutual Information (non-linear patterns) against conversion labels.

| Feature | Pearson | MI | What it means |
| :--- | :---: | :---: | :--- |
| **`billing_reached`** | 0.902 | 0.515 | Strong signal. Getting to checkout basically predicts conversion. |
| **`total_dwell_time_ms`** | 0.779 | 0.429 | Strong correlation. Longer stay = more likely to buy. |
| **`average_movement_distance`** | 0.503 | 0.244 | Moderate. Active shoppers explore more before checking out. |
| **`path_entropy`** | 0.573 | 0.229 | Moderate. Messy paths mean engaged shopping. |
| **`path_length`** | 0.460 | 0.219 | Moderate. More steps means more store traversal. |
| **`unique_grids_visited`** | 0.454 | 0.209 | Moderate. Broader zone coverage. |
| **`maximum_detection_confidence`** | NaN | 0.044 | Weak. Mostly constant in this dataset. |
| **`revisit_count`** | 0.142 | 0.000 | Not useful. |
| **`average_detection_confidence`** | NaN | 0.000 | Constant (0.5). Useless. |
| **`minimum_detection_confidence`** | NaN | 0.000 | Constant (0.5). Useless. |
| **`low_confidence_event_ratio`** | NaN | 0.000 | Constant (1.0). Useless. |

### Quick take

- `billing_reached` and `total_dwell_time_ms` dominate both metrics. Conversion really is about getting to checkout and how long you stay.
- Spatial features (`average_movement_distance`, `path_entropy`) have decent mutual information (> 0.22), so path complexity matters in non-linear ways that linear models might miss.

---

## 4. Model Performance

Trained **Logistic Regression** (linear) and **Random Forest** (non-linear) on 75/25 stratified split.

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **5-Fold CV ROC-AUC (mean)** | **0.988** | 0.968 |
| **Test Accuracy** | 0.933 | 0.933 |
| **Test Precision** | 0.875 | 0.875 |
| **Test Recall** | 1.000 | 1.000 |
| **Test F1** | 0.933 | 0.933 |
| **Test ROC-AUC** | 1.000 | 1.000 |

*That perfect 1.000 ROC-AUC is suspicious - test set is only 15 samples and `billing_reached` makes separation too easy.*

---

## 5. Feature Importance (Random Forest)

Gini importance ranking:

1.  **`billing_reached` (32.79%)**: Strongest by far. Hit the billing zone = probably buying.
2.  **`total_dwell_time_ms` (30.14%)**: Crucial. More time in store = buying intent.
3.  **`average_movement_distance` (17.50%)**: Spatial engagement. Moving across sections = more active shopper.
4.  **`path_length` (7.51%)** & **`path_entropy` (6.44%)**: Path complexity signals.
5.  **`unique_grids_visited` (5.63%)**: How much of the store they covered.

### What to fix

- **Confidence features**: Need to fix YOLO tracking so it reports real confidence scores (0.0-1.0) instead of hardcoded dummy values. Then we can see if occlusion/track quality affects predictions.
- **Not enough data**: Don't deploy online until we have more sessions. The saved model (`data/best_conversion_model.pkl`) is fine for dry runs and validation only.

---

## 6. Bootstrapped Stability Check (500 iterations)

To see if our small sample size (58 sessions) is a real problem, we ran 500 bootstrap resamples. Each iteration: resample with replacement, 75/25 split, train, record metrics and feature importance.

### ROC-AUC Stability

| Model | Mean ROC-AUC | 95% Range |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.988 | [0.893, 1.000] |
| **Random Forest** | 0.993 | [0.932, 1.000] |

> **Interpretation**: Ranges are tight and high (> 0.89). Even with only 58 sessions, the conversion signals are **actually pretty stable** - not just random split luck.

---

### Feature Importances Over 500 Bootstraps (Random Forest)

| Feature | Mean Importance | 95% Range | Tier |
| :--- | :---: | :---: | :--- |
| **`billing_reached`** | 0.3153 | [0.1750, 0.4577] | Tier 1 (Solid) |
| **`total_dwell_time_ms`** | 0.3077 | [0.1621, 0.4704] | Tier 1 (Solid) |
| **`average_movement_distance`** | 0.1557 | [0.0611, 0.2650] | Tier 2 (Moderate) |
| **`path_entropy`** | 0.0736 | [0.0293, 0.1462] | Tier 3 (Weak but non-zero) |
| **`unique_grids_visited`** | 0.0730 | [0.0274, 0.1384] | Tier 3 (Weak but non-zero) |
| **`path_length`** | 0.0729 | [0.0258, 0.1512] | Tier 3 (Weak but non-zero) |
| **`revisit_count`** | 0.0017 | [0.0000, 0.0100] | Noise |
| *Confidence features* | 0.0000 | [0.0000, 0.0000] | Dead |

### Mutual Information Over 500 Bootstraps

| Feature | Mean MI | 95% Range |
| :--- | :---: | :---: |
| **`billing_reached`** | 0.5315 | [0.3783, 0.6980] |
| **`total_dwell_time_ms`** | 0.4629 | [0.3111, 0.6394] |
| **`average_movement_distance`** | 0.3156 | [0.1515, 0.4938] |
| **`unique_grids_visited`** | 0.2681 | [0.1276, 0.4296] |
| **`path_length`** | 0.2642 | [0.1234, 0.4246] |
| **`path_entropy`** | 0.2609 | [0.1229, 0.4071] |
| *Confidence features* | ~0.024 | [0.0000, ~0.134] |
| **`revisit_count`** | 0.0206 | [0.0000, 0.1296] |

### What bootstrapping tells us

1. **Clear tiers**: `billing_reached` and `total_dwell_time_ms` never overlap with lower-tier features. Lower bounds (0.16-0.17) are above upper bounds of Tier 3 features. Hierarchy is stable.
2. **Non-linear signal is real**: MI scores for path features average ~0.26, so spatial complexity genuinely correlates with intent.
3. **Statistically valid**: The stability plot (`data/stability_audit.png`) shows both classifiers have narrow performance spread, median ROC-AUC near 0.99. Our feature engineering is capturing real behavior patterns.

---

## 7. Ablation: Removing `billing_reached`

To test prediction *before* checkout (and check for target leakage), we reran everything without `billing_reached`. 500 bootstrap iterations.

### ROC-AUC With vs Without billing_reached

| Model Setup | Mean ROC-AUC | 95% Range |
| :--- | :---: | :---: |
| **Logistic Regression (full)** | 0.988 | [0.893, 1.000] |
| **Logistic Regression (no billing)** | **0.937** | **[0.759, 1.000]** |
| **Random Forest (full)** | 0.993 | [0.932, 1.000] |
| **Random Forest (no billing)** | **0.976** | **[0.857, 1.000]** |

### What this shows

- **More variance**: Removing `billing_reached` widens confidence intervals. LR lower bound drops from 0.893 to 0.759. RF from 0.932 to 0.857. Without that strong explicit signal, the model is more sensitive to which sessions end up in train vs test.
- **But still strong**: Mean ROC-AUC is still high (RF = 0.976, LR = 0.937). So there are real intent signals *before* checkout - dwell time and movement patterns tell you something even before someone reaches the register.

---

### Feature Importances After Removing billing_reached (Random Forest)

| Feature | Mean Importance | 95% Range | Tier |
| :--- | :---: | :---: | :--- |
| **`total_dwell_time_ms`** | 0.4434 | [0.2793, 0.6016] | Tier 1 |
| **`average_movement_distance`** | 0.2380 | [0.1157, 0.3691] | Tier 2 |
| **`path_entropy`** | 0.1071 | [0.0484, 0.1978] | Tier 3 |
| **`unique_grids_visited`** | 0.1044 | [0.0481, 0.1913] | Tier 3 |
| **`path_length`** | 0.1042 | [0.0457, 0.1897] | Tier 3 |
| **`revisit_count`** | 0.0029 | [0.0000, 0.0152] | Noise |

### Mutual Information After Removing billing_reached

| Feature | Mean MI | 95% Range |
| :--- | :---: | :---: |
| **`total_dwell_time_ms`** | 0.4629 | [0.3111, 0.6394] |
| **`average_movement_distance`** | 0.3147 | [0.1405, 0.4914] |
| **`unique_grids_visited`** | 0.2659 | [0.1254, 0.4400] |
| **`path_length`** | 0.2615 | [0.1271, 0.4275] |
| **`path_entropy`** | 0.2591 | [0.1177, 0.4016] |

### Takeaways from ablation

1.  **Dwell time takes over**: Without `billing_reached`, `total_dwell_time_ms` jumps to 44.34% importance (from 30.77%).
2.  **Movement distance becomes #2**: Rises to 23.80% - clear second most important signal.
3.  **Spatial features double**: All three path features go to ~10.5% each. Pre-checkout trajectory complexity genuinely matters.
4.  **Plots saved**: Ablation results (boxplot + horizontal feature importance with CIs) at `data/ablation_stability.png`.

---

## 8. Known Problems

Three main validity threats:

1.  **Small sample (58 sessions)**: Bootstrap shows internal stability, but that doesn't mean it'll work on new stores, different days, or other shopper types.
2.  **Confidence features are dummy values**: Because the CV pipeline falls back to MOG2 when YOLO is bypassed, all confidence scores are hardcoded to 0.50. Zero variance = useless for modeling in this phase.
3.  **Complete separation in store ST1076**: That store had 20 sessions, 0 conversions (100% drop-off). Meanwhile STORE_BLR_002 had 27 conversions out of 38. This extreme separation breaks cross-store ROC-AUC calculation for models trained on ST1076.

---

## 9. Cross-Store Test (Training on One Store, Predicting on Another)

To test real-world generalization: trained on `STORE_BLR_002` (38 sessions, 27 conversions), predicted on `ST1076` (20 sessions, 0 conversions - all non-buyers).

### Predicted Probabilities on ST1076

| Model | Sub-cohort | Mean Prediction |
| :--- | :--- | :---: |
| **Full model** | All shoppers ($n=20$) | 26.48% (LR) / 46.90% (RF) |
| **Full model** | Reached billing ($n=3$) | **96.97% (LR) / 92.33% (RF)** |
| **Full model** | No billing ($n=17$) | **14.04% (LR) / 38.88% (RF)** |
| **Ablated (no billing)** | All shoppers ($n=20$) | 49.27% (LR) / 72.00% (RF) |
| **Ablated (no billing)** | Reached billing ($n=3$) | 91.46% (LR) / 91.00% (RF) |
| **Ablated (no billing)** | No billing ($n=17$) | 41.83% (LR) / 68.65% (RF) |

### What this means

1.  **Full model works okay for non-checkout**: LR correctly predicts low probability (14.04%) for the 17 who didn't reach checkout. For the 3 who did reach billing, it predicts 96.97% - technically a false positive (since no one bought), but logically consistent with the rule it learned.
2.  **Ablated model fails on new layout**: Without `billing_reached`, predictions for non-checkout shoppers jump to 41.83% (LR) and 68.65% (RF). 
    *Why?* The model learned spatial thresholds (dwell time, movement distance) that fit `STORE_BLR_002`'s layout. `ST1076` has a different physical layout, so the same absolute numbers don't mean the same thing. A shopper who dwells "long" in ST1076 gets flagged as likely to convert, even though they didn't.
3.  **Fix needed**: To make features portable across stores, we need to normalize spatial features. Raw coordinates and milliseconds don't transfer.

---

## 10. Normalization Experiment

Tried to fix the cross-store problem by normalizing features (dividing by store averages) instead of using raw values.

### Store-level averages (raw)

| Metric | STORE_BLR_002 | ST1076 | Ratio |
| :--- | :---: | :---: | :---: |
| **Dwell time (ms)** | 4,174.7 | 1,030.0 | **4.05x** |
| **Movement distance** | 28.76 | 20.26 | **1.42x** |
| **Path length (steps)** | 2.87 | 2.40 | **1.20x** |
| **Unique grids** | 2.84 | 2.40 | **1.18x** |

### Predictions on ST1076 after normalization (ablated model, no billing)

| Cohort | Metric | Raw | Normalized | Change |
| :--- | :--- | :---: | :---: | :--- |
| **Non-billing** ($n=17$) | LR Mean | 41.83% | **55.63%** | Up 13.80% |
| **Non-billing** ($n=17$) | RF Mean | 68.65% | **68.65%** | No change |
| **Billing** ($n=3$) | LR Mean | 91.46% | **94.94%** | Up 3.48% |
| **Billing** ($n=3$) | RF Mean | 91.00% | **91.33%** | No change |

### So normalization made it worse

Counter-intuitive: normalizing by store averages actually increased false positives (non-converters in ST1076 went from 41.83% to 55.63% predicted probability).

**Why this happens (target shift)**: Store averages are skewed by conversion rate. `STORE_BLR_002` converted at 71% (27/38), so its average dwell time is high (4,174 ms). `ST1076` converted at 0% (0/20), so its average dwell time is low (1,030 ms). Dividing ST1076 sessions by their low average inflates them. A 2.0 second dwell in ST1076 becomes ratio 2.0x, but on the BLR scale that's equivalent to 8.3 seconds - which the model strongly associates with buying.

### Better approaches for different store layouts

1.  **Geometric normalization**: Divide spatial features by store-specific physical constants (store diagonal length, camera boundary coordinates).
2.  **Baseline traffic speed (target-independent)**: Normalize dwell times by the average of **non-converting shoppers only** in each store. This aligns baseline walking speed without leaking conversion rate into the scaling factor.
3.  **Plot saved**: Before/after distributions at `data/feature_normalization_audit.png`.