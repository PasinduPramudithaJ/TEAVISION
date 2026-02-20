# =============================================================
#      HANDCRAFTED FEATURE EXTRACTION (ONLY OPENCV)
#      FOR TRADITIONAL ML (SVM, RF, KNN, Logistic Regression)
# =============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ---------------- Config ----------------
DATA_DIR = "data/preprocessed"   # root folder containing region subfolders
OUTPUT_CSV = "data/handcrafted_features_opencv.csv"

# ---------------- Feature extraction functions ----------------

def extract_color_features(img):
    # RGB mean
    r_mean = np.mean(img[:,:,2])
    g_mean = np.mean(img[:,:,1])
    b_mean = np.mean(img[:,:,0])

    # HSV mean
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])

    return [r_mean, g_mean, b_mean, h_mean, s_mean, v_mean]


def extract_texture_features(gray):
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean((gray - mean)**3) / (std**3 + 1e-8)
    kurtosis = np.mean((gray - mean)**4) / (std**4 + 1e-8)

    return [mean, std, skewness, kurtosis]


def extract_lbp_like(gray):
    lbp = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i,j]
            binary = 0
            binary |= (gray[i-1,j-1] > center) << 7
            binary |= (gray[i-1,j  ] > center) << 6
            binary |= (gray[i-1,j+1] > center) << 5
            binary |= (gray[i,j+1]   > center) << 4
            binary |= (gray[i+1,j+1] > center) << 3
            binary |= (gray[i+1,j]   > center) << 2
            binary |= (gray[i+1,j-1] > center) << 1
            binary |= (gray[i,j-1]   > center) << 0
            lbp[i,j] = binary

    hist = cv2.calcHist([lbp],[0],None,[256],[0,256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    return hist.tolist()


def extract_edge_features(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)

    return [np.mean(mag)]

# ---------------- Process dataset ----------------

features = []
labels = []
paths = []
regions = []
groups = []
group_labels = []

# Define possible tea groups
group_classes = ["OP", "BOP", "BOPF"]

# Only include folders (skip any files in root)
class_names = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

for class_id, cname in enumerate(class_names):

    folder = os.path.join(DATA_DIR, cname)

    for root, _, files in os.walk(folder):
        for fname in files:

            path = os.path.join(root, fname)

            try:
                img = cv2.imread(path)
                if img is None:
                    continue

                img = cv2.resize(img, (224,224))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            except:
                continue

            # =========================
            # Extract GROUP from filename
            # Example: NU_OP_001.jpg
            # =========================
            name = os.path.splitext(fname)[0]
            parts = name.split("_")

            if len(parts) < 2:
                continue

            group = parts[1]  # OP / BOP / BOPF

            if group not in group_classes:
                continue

            group_id = group_classes.index(group)

            # =========================
            # Feature extraction
            # =========================
            f_color = extract_color_features(img)
            f_texture = extract_texture_features(gray)
            f_lbp = extract_lbp_like(gray)
            f_edge = extract_edge_features(gray)

            feature_vector = f_color + f_texture + f_edge + f_lbp

            features.append(feature_vector)
            labels.append(class_id)     # region numeric label
            regions.append(cname)      # region folder name
            groups.append(group)       # OP/BOP/BOPF
            group_labels.append(group_id)
            paths.append(path)

            print(f"Processed: {path}")

# ---------------- Save CSV ----------------

columns = (
    ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"] +
    ["Texture_mean","Texture_std","Texture_skew","Texture_kurtosis"] +
    ["Edge_mean"] +
    [f"LBP_{i}" for i in range(256)]
)

df = pd.DataFrame(features, columns=columns)

df["region_label"] = labels
df["region"] = regions
df["group"] = groups
df["group_label"] = group_labels
df["path"] = paths

df.to_csv(OUTPUT_CSV, index=False)

print("\n=================================================")
print(f"   ✔ Handcrafted Features Saved → {OUTPUT_CSV}")
print("   ✔ Columns added: region, region_label")
print("   ✔ Columns added: group, group_label")
print("=================================================")