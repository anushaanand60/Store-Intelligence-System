import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

def main():
    dataset_path = WORKSPACE_DIR / "data" / "ml_features_real.csv"
    roc_plot_path = WORKSPACE_DIR / "data" / "roc_curves.png"
    fi_plot_path = WORKSPACE_DIR / "data" / "feature_importance.png"
    model_path = WORKSPACE_DIR / "data" / "best_conversion_model.pkl"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}. Run pipeline/export_real_dataset.py first!")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    y = df["conversion_outcome"]
    X = df.drop(columns=["session_id", "visitor_id", "session_seq", "conversion_outcome"])
    feature_names = X.columns.tolist()

    print(f"Dataset shape: {df.shape}")
    print(f"Features list ({len(feature_names)}): {feature_names}\n")

    correlations = X.corrwith(y)
    np.random.seed(42)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    relation_df = pd.DataFrame({
        "Pearson Correlation": correlations,
        "Mutual Information": mi_scores
    }, index=feature_names)
    relation_df = relation_df.sort_values(by="Mutual Information", ascending=False)

    print("                 PREDICTIVE RELATIONSHIPS WITH TARGET                 ")
    print(relation_df.to_string())
    print("\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set size : {X_train.shape[0]} rows")
    print(f"Test set size     : {X_test.shape[0]} rows\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Training Logistic Regression baseline...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg_cv_scores = cross_val_score(
        log_reg, X_train_scaled, y_train, cv=cv_strategy, scoring="roc_auc"
    )
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    }

    print("Training Random Forest baseline...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_cv_scores = cross_val_score(
        rf_clf, X_train, y_train, cv=cv_strategy, scoring="roc_auc"
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    }

    print("\n")
    print("                         MODEL PERFORMANCE SUMMARY                     ")
    print(f"{'Metric':<25} | {'Logistic Regression':<20} | {'Random Forest':<15}")
    print(f"{'5-Fold Mean CV ROC-AUC':<25} | {np.mean(log_reg_cv_scores):<20.3f} | {np.mean(rf_cv_scores):<15.3f}")

    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        print(f"{metric:<25} | {lr_metrics[metric]:<20.3f} | {rf_metrics[metric]:<15.3f}")

    print("                LOGISTIC REGRESSION COEFFICIENTS                      ")
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": log_reg.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    print("\n")

    print("Analyzing Random Forest feature importances...")
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("                  RANDOM FOREST FEATURE IMPORTANCE                    ")
    for i in range(len(feature_names)):
        print(f"{i+1:<2}. {feature_names[indices[i]]:<30}: {importances[indices[i]]:.4f}")

    print("Generating and saving ROC curves plot...")
    plt.figure(figsize=(8, 6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['ROC-AUC']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"ROC Curves plot saved to {roc_plot_path}")

    print("Generating and saving Feature Importance plot...")
    plt.figure(figsize=(10, 6))
    sorted_features = [feature_names[idx] for idx in indices[::-1]]
    sorted_importances = importances[indices[::-1]]
    plt.barh(sorted_features, sorted_importances, color="skyblue", edgecolor="gray")
    plt.xlabel("Gini Importance Score")
    plt.title("Random Forest - Feature Importance Analysis")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Feature Importance plot saved to {fi_plot_path}")

    best_model_name = "Random Forest" if rf_metrics["ROC-AUC"] >= lr_metrics["ROC-AUC"] else "Logistic Regression"
    best_clf = rf_clf if best_model_name == "Random Forest" else log_reg

    model_payload = {
        "best_model_name": best_model_name,
        "classifier": best_clf,
        "scaler": scaler if best_model_name == "Logistic Regression" else None,
        "features": feature_names,
        "metrics": {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics
        }
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    print(f"Successfully serialized the best model ({best_model_name}) to {model_path}\n")

if __name__ == "__main__":
    main()