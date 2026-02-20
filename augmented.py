import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ==============================
# 1️⃣ Load Dataset
# ==============================
df = pd.read_csv("data/handcrafted_features_opencv.csv")

print("Original Region Distribution:")
print(df["region"].value_counts())

# Encode region
region_encoder = LabelEncoder()
df["region_label"] = region_encoder.fit_transform(df["region"])

# Keep only numeric feature columns
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols.remove("region_label")

X = df[feature_cols]
y = df["region_label"]

target_count = 100
balanced_parts = []

for region_id in sorted(y.unique()):
    region_mask = y == region_id
    region_data = df[region_mask]

    if len(region_data) > target_count:
        # Downsample
        region_sampled = resample(
            region_data,
            replace=False,
            n_samples=target_count,
            random_state=42
        )

    elif len(region_data) < target_count:

        # Apply SMOTE only for region label
        smote = SMOTE(
            sampling_strategy={region_id: target_count},
            random_state=42,
            k_neighbors=min(5, len(region_data) - 1)
        )

        X_res, y_res = smote.fit_resample(X, y)

        temp_df = pd.DataFrame(X_res, columns=feature_cols)
        temp_df["region_label"] = y_res

        region_sampled = temp_df[temp_df["region_label"] == region_id].copy()

        # 🔹 Assign group values proportionally
        original_groups = region_data["group"].values
        region_sampled["group"] = np.random.choice(
            original_groups,
            size=len(region_sampled),
            replace=True
        )

        # Restore region name
        region_name = region_encoder.inverse_transform([region_id])[0]
        region_sampled["region"] = region_name

    else:
        region_sampled = region_data.copy()

    balanced_parts.append(region_sampled)

# ==============================
# Combine all regions
# ==============================
final_df = pd.concat(balanced_parts).reset_index(drop=True)

print("\nBalanced Region Distribution:")
print(final_df["region"].value_counts())
print("\nTotal Samples:", len(final_df))

# ==============================
# Save
# ==============================
final_df.to_csv("data/handcrafted_features_opencv_balanced.csv", index=False)

print("\n✅ Balanced dataset saved with separate region & group columns.")