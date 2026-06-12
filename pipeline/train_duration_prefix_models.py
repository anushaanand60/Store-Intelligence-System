import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

WORKSPACE_DIR = Path(__file__).resolve().parent.parent

def evaluate_horizon_config(df_horizon, feature_cols):
    X = df_horizon[feature_cols].copy()
    y = df_horizon["conversion_outcome"].copy()
    groups = df_horizon["session_id"].copy()

    selector = X.std() > 1e-6
    active_features = X.columns[selector].tolist()

    sgkf = StratifiedGroupKFold(n_splits=5)
    lr_metrics = {"roc_auc": [], "precision": [], "recall": [], "f1": []}
    rf_metrics = {"roc_auc": [], "precision": [], "recall": [], "f1": []}
    rf_importances = []

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx][active_features], X.iloc[test_idx][active_features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        y_pred_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        y_pred_lr = lr.predict(X_test_scaled)

        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_train_scaled, y_train)
        y_pred_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
        y_pred_rf = rf.predict(X_test_scaled)

        if len(np.unique(y_test)) < 2:
            continue

        lr_metrics["roc_auc"].append(roc_auc_score(y_test, y_pred_prob_lr))
        lr_metrics["precision"].append(precision_score(y_test, y_pred_lr, zero_division=0))
        lr_metrics["recall"].append(recall_score(y_test, y_pred_lr, zero_division=0))
        lr_metrics["f1"].append(f1_score(y_test, y_pred_lr, zero_division=0))

        rf_metrics["roc_auc"].append(roc_auc_score(y_test, y_pred_prob_rf))
        rf_metrics["precision"].append(precision_score(y_test, y_pred_rf, zero_division=0))
        rf_metrics["recall"].append(recall_score(y_test, y_pred_rf, zero_division=0))
        rf_metrics["f1"].append(f1_score(y_test, y_pred_rf, zero_division=0))

        rf_importances.append(rf.feature_importances_)

    avg_importances = np.mean(rf_importances, axis=0) if rf_importances else []
    importance_dict = dict(zip(active_features, avg_importances)) if len(avg_importances) > 0 else {}

    return {
        "lr": {k: np.mean(v) for k, v in lr_metrics.items()},
        "rf": {k: np.mean(v) for k, v in rf_metrics.items()},
        "importances": importance_dict,
        "feature_count": len(active_features)
    }

def main():
    csv_std_path = WORKSPACE_DIR / "data" / "ml_features_duration_prefixes.csv"
    csv_abl_path = WORKSPACE_DIR / "data" / "ml_features_duration_prefixes_no_billing.csv"

    if not csv_std_path.exists() or not csv_abl_path.exists():
        print("Error: Prefix datasets not found.")
        sys.exit(1)

    df_std = pd.read_csv(csv_std_path)
    df_abl = pd.read_csv(csv_abl_path)

    feature_cols_all = [
        "path_length", "unique_grids_visited", "total_dwell_time_ms", "billing_reached",
        "revisit_count", "path_entropy", "average_movement_distance", "average_detection_confidence",
        "minimum_detection_confidence", "maximum_detection_confidence", "low_confidence_event_ratio"
    ]
    feature_cols_no_reached = [col for col in feature_cols_all if col != "billing_reached"]

    horizons = [10, 25, 50, 75, 100]
    results_ref = {}
    results_abl1 = {}
    results_abl2 = {}

    print("      TRAINING DURATION-BASED PREFIX MODELS               ")
    for pct in horizons:
        df_h_std = df_std[df_std["observed_pct"] == pct]
        results_ref[pct] = evaluate_horizon_config(df_h_std, feature_cols_all)
        results_abl1[pct] = evaluate_horizon_config(df_h_std, feature_cols_no_reached)

        df_h_abl = df_abl[df_abl["observed_pct"] == pct]
        results_abl2[pct] = evaluate_horizon_config(df_h_abl, feature_cols_no_reached)

        print(f"\n Horizon: {pct}% Observed ")
        print(f"Ref (Full):      LR ROC-AUC={results_ref[pct]['lr']['roc_auc']:.4f}, RF ROC-AUC={results_ref[pct]['rf']['roc_auc']:.4f}")
        print(f"Abl1 (No reach): LR ROC-AUC={results_abl1[pct]['lr']['roc_auc']:.4f}, RF ROC-AUC={results_abl1[pct]['rf']['roc_auc']:.4f}")
        print(f"Abl2 (No bill):  LR ROC-AUC={results_abl2[pct]['lr']['roc_auc']:.4f}, RF ROC-AUC={results_abl2[pct]['rf']['roc_auc']:.4f}")

    print("        TOP 5 FEATURE IMPORTANCES (RF GINI)   ")
    for pct in [25, 50]:
        for name, res in [("Ablation 1 (billing_reached Removed)", results_abl1[pct]),
                          ("Ablation 2 (All billing-derived Removed)", results_abl2[pct])]:
            print(f"  * {name}:")
            sorted_imp = sorted(res["importances"].items(), key=lambda x: x[1], reverse=True)[:5]
            for idx, (f_name, val) in enumerate(sorted_imp):
                print(f"    {idx+1}. {f_name:<28} : {val:.4f}")

    print("     INCREMENTAL RF ROC-AUC GAINS BETWEEN HORIZONS     ")
    print(f"{'Horizon Transition':<25} | {'Ref RF':<8} | {'Abl1 RF':<8} | {'Abl2 RF':<8}")
    for i in range(len(horizons) - 1):
        h_prev, h_next = horizons[i], horizons[i+1]
        gain_ref = results_ref[h_next]["rf"]["roc_auc"] - results_ref[h_prev]["rf"]["roc_auc"]
        gain_abl1 = results_abl1[h_next]["rf"]["roc_auc"] - results_abl1[h_prev]["rf"]["roc_auc"]
        gain_abl2 = results_abl2[h_next]["rf"]["roc_auc"] - results_abl2[h_prev]["rf"]["roc_auc"]
        print(f"{h_prev}% observed -> {h_next}% observed | {gain_ref:+.4f}  | {gain_abl1:+.4f}  | {gain_abl2:+.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(horizons, [results_ref[pct]["rf"]["roc_auc"] for pct in horizons], marker='o', color='#e74c3c', linewidth=2.5, label='RF - Full Features (Ref)')
    plt.plot(horizons, [results_abl1[pct]["rf"]["roc_auc"] for pct in horizons], marker='s', color='#3498db', linewidth=2.5, label='RF - billing_reached Removed (Ablation 1)')
    plt.plot(horizons, [results_abl2[pct]["rf"]["roc_auc"] for pct in horizons], marker='^', color='#2ecc71', linewidth=2.5, label='RF - No Billing-derived Info (Ablation 2)')
    plt.axvline(x=100, color='#95a5a6', linestyle='--', alpha=0.7, label='100% Horizon Baseline')
    plt.title('Ablated Early Conversion Prediction Curve: Observed Journey % vs. RF ROC-AUC', fontsize=12, pad=15)
    plt.xlabel('Observed Journey % (Chronological Progress)', fontsize=10, labelpad=10)
    plt.ylabel('Stratified Group 5-Fold ROC-AUC', fontsize=10, labelpad=10)
    plt.xticks(horizons, [f"{h}%" for h in horizons])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.ylim(0.45, 1.05)

    for x, y in zip(horizons, [results_abl2[pct]["rf"]["roc_auc"] for pct in horizons]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#27ae60')

    plt.tight_layout()
    plot_path = WORKSPACE_DIR / "data" / "duration_prefix_prediction_curve.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved ablated learning curve plot to {plot_path}")

    results_json_path = WORKSPACE_DIR / "data" / "duration_prefix_results.json"
    summary_results = {
        "ref": {str(k): v for k, v in results_ref.items()},
        "abl1": {str(k): v for k, v in results_abl1.items()},
        "abl2": {str(k): v for k, v in results_abl2.items()}
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

if __name__ == "__main__":
    main()