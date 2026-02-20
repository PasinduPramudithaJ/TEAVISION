# =============================================================
# COMPLETE ML TRAINING PIPELINE (REGION + GROUP) WITH METRICS SUMMARY
# =============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, jaccard_score
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------- Config ----------------
CSV_PATH = "data/handcrafted_features_opencv_balanced.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Load Data ----------------
df = pd.read_csv(CSV_PATH)

# Features
feature_columns = (
    ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"] +
    ["Texture_mean","Texture_std","Texture_skew","Texture_kurtosis"] +
    ["Edge_mean"] +
    [f"LBP_{i}" for i in range(256)]
)

X = df[feature_columns].values

# ---------------- Models ----------------
models = {
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        class_weight='balanced'
    )
}

# ---------------- Standardize Features ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ---------------- Train for Both Targets ----------------
label_columns = ["region", "group"]

for label_column in label_columns:
    print(f"\n=== Training for {label_column} ===")
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[label_column].values)
    class_names = le.classes_
    num_classes = len(class_names)
    joblib.dump(le, os.path.join(MODEL_DIR, f"{label_column}_encoder.pkl"))
    
    # Store all models metrics here
    metrics_summary = []

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        fold_acc, fold_precision, fold_recall, fold_f1, fold_iou = [], [], [], [], []
        all_y_true, all_y_pred = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), start=1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if len(np.unique(y_train)) < 2:
                print(f"Fold {fold} skipped: training set contains only 1 class")
                continue
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_precision.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            fold_recall.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            fold_f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            fold_iou.append(jaccard_score(y_test, y_pred, average='macro', zero_division=0))

        # Compute mean metrics for this model
        mean_acc = np.mean(fold_acc)
        mean_precision = np.mean(fold_precision)
        mean_recall = np.mean(fold_recall)
        mean_f1 = np.mean(fold_f1)
        mean_iou = np.mean(fold_iou)

        metrics_summary.append({
            "Model": model_name,
            "Accuracy": mean_acc,
            "Precision": mean_precision,
            "Recall": mean_recall,
            "F1-score": mean_f1,
            "IoU": mean_iou
        })

        print(f"\n--- {model_name} ---")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Precision: {mean_precision:.4f}")
        print(f"Recall: {mean_recall:.4f}")
        print(f"F1-score: {mean_f1:.4f}")
        print(f"IoU: {mean_iou:.4f}")

        # Train on full data
        model.fit(X_scaled, y)
        model_filename = os.path.join(MODEL_DIR, f"{model_name}_{label_column}.joblib")
        joblib.dump(model, model_filename)
        print(f"Saved model -> {model_filename}")

        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{model_name} Confusion Matrix ({label_column})")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_{label_column}_confusion_matrix.png"))
        plt.close()
    
    # Summary table per target
    summary_df = pd.DataFrame(metrics_summary)
    summary_csv_path = os.path.join(RESULTS_DIR, f"{label_column}_metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅ Metrics summary for {label_column} saved -> {summary_csv_path}")
    print(summary_df)

print("\n✅ Training completed for both region and group targets.")