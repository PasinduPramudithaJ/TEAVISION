import os
import socket
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

MODEL_DIR = "models"
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- Load All Models ----------------
# Assuming you may have multiple models for regions and groups, named conventionally
available_models = {
    "svm": {
        "region": joblib.load(os.path.join(MODEL_DIR, "SVM_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "SVM_group.joblib"))
    },
    "randomforest": {  # example random forest
        "region": joblib.load(os.path.join(MODEL_DIR, "RandomForest_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "RandomForest_group.joblib"))
    },
    "knn": {  # example KNN
        "region": joblib.load(os.path.join(MODEL_DIR, "KNN_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "KNN_group.joblib"))
    },
    "logisticregression": {  # example logistic regression
        "region": joblib.load(os.path.join(MODEL_DIR, "LogisticRegression_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "LogisticRegression_group.joblib"))
    }
}

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

feature_columns = (
    ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"] +
    ["Texture_mean","Texture_std","Texture_skew","Texture_kurtosis"] +
    ["Edge_mean"] +
    [f"LBP_{i}" for i in range(256)]
)

@app.route("/predict_region_group", methods=["POST"])
def predict():
    # Get model name from header, default to SVM if not provided
    model_name = request.headers.get("X-Model-Name", "svm").lower()
    if model_name not in available_models:
        return jsonify({"error": f"Model '{model_name}' not available"}), 400

    model_region = available_models[model_name]["region"]
    model_group = available_models[model_name]["group"]

    data = request.get_json()
    if not data or "rows" not in data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data["rows"])

    # Fill missing features with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    try:
        X = df[feature_columns].astype(float).values
    except Exception as e:
        return jsonify({"error": f"Feature conversion error: {str(e)}"}), 400

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return jsonify({"error": f"Scaler transform error: {str(e)}"}), 500

    # Predict
    try:
        predicted_region = model_region.predict(X_scaled)
        predicted_group = model_group.predict(X_scaled)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    df["predicted_region"] = predicted_region
    df["predicted_group"] = predicted_group

    # Optional: fill NaNs in original fields with 0
    for col in ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return jsonify({"results": df.to_dict(orient="records")})

# ================= Server Setup ======================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"✅ Server running on:")
    print(f"  Localhost: http://127.0.0.1:5001")
    print(f"  Network:   http://{local_ip}:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)