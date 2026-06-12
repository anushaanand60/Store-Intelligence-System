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

def evaluate_horizon(df_horizon, feature_cols):
    X = df_horizon[feature_cols].copy()
    y = df_horizon["conversion_outcome"].copy()
    groups = df_horizon["session_id"].copy()

    selector = X.std() > 1e-6
    active_features = X.columns[selector].tolist()

    sgkf = StratifiedGroupKFold(n_splits=5)
    lr_metrics = {"roc_auc": [], "precision": [], "recall": [], "f1": []}
    rf_metrics = {"roc_auc": [], "precision": [], "recall": [], "f1": []}

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

    return {
        "lr": {k: np.mean(v) for k, v in lr_metrics.items()},
        "rf": {k: np.mean(v) for k, v in rf_metrics.items()},
        "feature_count": len(active_features)
    }

def main():
    csv_path = WORKSPACE_DIR / "data" / "ml_features_prefixes.csv"
    if not csv_path.exists():
        print(f"Error: Prefix dataset not found at {csv_path}. Run generate_prefix_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    feature_cols = [
        "path_length",
        "unique_grids_visited",
        "total_dwell_time_ms",
        "billing_reached",
        "revisit_count",
        "path_entropy",
        "average_movement_distance",
        "average_detection_confidence",
        "minimum_detection_confidence",
        "maximum_detection_confidence",
        "low_confidence_event_ratio"
    ]

    horizons = [10, 25, 50, 75, 100]
    results = {}

    print("      TRAINING MODELS PER OBSERVATION HORIZON             ")
    for pct in horizons:
        df_horizon = df[df["observed_pct"] == pct]
        horizon_results = evaluate_horizon(df_horizon, feature_cols)
        results[pct] = horizon_results

        print(f"\n Horizon: {pct}% Observed ")
        print(f"Logistic Regression: ROC-AUC={horizon_results['lr']['roc_auc']:.4f}, F1={horizon_results['lr']['f1']:.4f}, Prec={horizon_results['lr']['precision']:.4f}, Rec={horizon_results['lr']['recall']:.4f}")
        print(f"Random Forest:       ROC-AUC={horizon_results['rf']['roc_auc']:.4f}, F1={horizon_results['rf']['f1']:.4f}, Prec={horizon_results['rf']['precision']:.4f}, Rec={horizon_results['rf']['recall']:.4f}")

    print("        INCREMENTAL ROC-AUC GAINS             ")
    print(f"{'Horizon transition':<22} | {'LR Gain':<10} | {'RF Gain':<10}")
    for i in range(len(horizons) - 1):
        h_prev, h_next = horizons[i], horizons[i+1]
        lr_gain = results[h_next]["lr"]["roc_auc"] - results[h_prev]["lr"]["roc_auc"]
        rf_gain = results[h_next]["rf"]["roc_auc"] - results[h_prev]["rf"]["roc_auc"]
        print(f"{h_prev}% observed -> {h_next}% observed | {lr_gain:+.4f}    | {rf_gain:+.4f}")

    plt.figure(figsize=(10, 6))
    lr_aucs = [results[pct]["lr"]["roc_auc"] for pct in horizons]
    rf_aucs = [results[pct]["rf"]["roc_auc"] for pct in horizons]

    plt.plot(horizons, lr_aucs, marker='o', color='#3498db', linewidth=2.5, label='Logistic Regression')
    plt.plot(horizons, rf_aucs, marker='s', color='#2ecc71', linewidth=2.5, label='Random Forest')
    plt.axvline(x=100, color='#e74c3c', linestyle='--', alpha=0.7, label='100% Horizon Baseline')
    plt.title('Early Conversion Prediction Curve: Observed Journey % vs. Cross-Validated ROC-AUC', fontsize=12, pad=15)
    plt.xlabel('Observed Journey % (Chronological Progress)', fontsize=10, labelpad=10)
    plt.ylabel('Stratified Group 5-Fold ROC-AUC', fontsize=10, labelpad=10)
    plt.xticks(horizons, [f"{h}%" for h in horizons])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.ylim(0.45, 1.05)

    for x, y_lr, y_rf in zip(horizons, lr_aucs, rf_aucs):
        plt.annotate(f"{y_lr:.3f}", (x, y_lr), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#2c3e50')
        plt.annotate(f"{y_rf:.3f}", (x, y_rf), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='#27ae60')

    plt.tight_layout()
    plot_path = WORKSPACE_DIR / "data" / "prefix_prediction_curve.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved learning curve plot to {plot_path}")

    results_json_path = WORKSPACE_DIR / "data" / "prefix_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

if __name__ == "__main__":
    main()