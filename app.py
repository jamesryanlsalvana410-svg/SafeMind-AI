from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import numpy as np
import json
import threading
import hashlib
import redis
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM

# -----------------------------------------------------
# APP SETUP
# -----------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -----------------------------------------------------
# CLINICAL RECOMMENDATIONS
# -----------------------------------------------------
RECOMMENDATIONS = {
    "low": "Mild symptoms. Maintain healthy habits and monitor.",
    "moderate": "Moderate symptoms. Consider professional support.",
    "high": "Severe symptoms. Seek professional help immediately."
}

def get_recommendation(severity):
    return RECOMMENDATIONS.get(severity, "No recommendation available.")

# -----------------------------------------------------
# FIREBASE SETUP
# -----------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase-key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------------------------------
# REDIS CACHE
# -----------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL")
cache = redis.from_url(REDIS_URL) if REDIS_URL else None
CACHE_TTL = 3600  # 1 hour

# -----------------------------------------------------
# MODEL + METADATA PATHS
# -----------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model_v2.keras")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")

# -----------------------------------------------------
# LOAD METADATA + TOKENIZER ONCE
# -----------------------------------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

TEXT_COLS = meta["text_cols"]
NUM_COLS = meta["num_cols"]
LABEL_CLASSES = meta["label_classes"]
MAX_LEN = meta["tokenizer_params"]["max_len"]

tokenizer = joblib.load(TOKENIZER_PATH)
print("✅ Tokenizer loaded.")

# -----------------------------------------------------
# LOAD MODEL ONCE
# -----------------------------------------------------
def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(MODEL_PATH, compile=False, custom_objects={"LSTM": lstm_no_time_major})
print("✅ Model loaded.")

# -----------------------------------------------------
# FIRESTORE SAVE ASYNC
# -----------------------------------------------------
def save_prediction_async(text_data, numeric_data, severity, confidence):
    try:
        db.collection("api_predictions").add({
            "text_data": text_data,
            "numeric_data": numeric_data,
            "severity": severity,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print("❌ Firestore Async Error:", e)

# -----------------------------------------------------
# PREDICTION PIPELINE
# -----------------------------------------------------
def preprocess_and_predict(input_dict):
    # ---- TEXT PROCESSING ----
    text_string = " ".join([str(input_dict.get(col, "")) for col in TEXT_COLS])
    seq = tokenizer.texts_to_sequences([text_string])
    seq = pad_sequences(seq, maxlen=MAX_LEN)

    # ---- NUMERIC PROCESSING ----
    numeric_list = [float(input_dict.get(col, 0)) for col in NUM_COLS]
    X_num = np.array(numeric_list).reshape(1, -1)

    # ---- PREDICT ----
    pred = model.predict([seq, X_num], verbose=0)
    idx = int(np.argmax(pred))
    severity = LABEL_CLASSES[idx]
    confidence = float(np.max(pred))

    return severity, confidence

# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    cache_key = "dashboard_assessments"
    if cache and cache.get(cache_key):
        assessments = json.loads(cache.get(cache_key))
    else:
        docs = db.collection("assessments").stream()
        assessments = [{**d.to_dict(), "id": d.id} for d in docs]
        if cache:
            cache.setex(cache_key, CACHE_TTL, json.dumps(assessments))
    return render_template("phq9.html", assessments=assessments)

@app.route("/analyze", methods=["POST"])
def analyze():
    # ---- Collect form data ----
    input_dict = {col: request.form.get(col, "") for col in TEXT_COLS}
    for col in NUM_COLS:
        input_dict[col] = request.form.get(col, 0)

    # ---- Generate Redis cache key ----
    cache_key = hashlib.sha256(json.dumps(input_dict, sort_keys=True).encode()).hexdigest()

    # ---- Check cache first ----
    if cache and cache.get(cache_key):
        cached_response = json.loads(cache.get(cache_key))
        severity = cached_response["severity"]
        recommendation = cached_response["recommendation"]
        confidence = cached_response["confidence"]
    else:
        # ---- Predict using model ----
        severity, confidence = preprocess_and_predict(input_dict)
        recommendation = get_recommendation(severity)

        # ---- Async Firestore save ----
        threading.Thread(
            target=save_prediction_async,
            args=(input_dict, NUM_COLS, severity, confidence)
        ).start()

        # ---- Store in Redis ----
        if cache:
            cache.setex(
                cache_key,
                CACHE_TTL,
                json.dumps({
                    "severity": severity,
                    "confidence": confidence,
                    "recommendation": recommendation
                })
            )

    # ---- Save to assessments collection ----
    db.collection("assessments").add({
        **input_dict,
        "severity": severity,
        "confidence": confidence,
        "recommendation": recommendation,
        "created_at": datetime.utcnow().isoformat()
    })

    flash(f"AI Prediction: {severity} — Recommendation: {recommendation}")
    return redirect(url_for("dashboard"))

@app.route("/predict", methods=["POST"])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    data = request.get_json()
    cache_key = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if cache and cache.get(cache_key):
        return jsonify(json.loads(cache.get(cache_key)))

    severity, confidence = preprocess_and_predict(data)
    recommendation = get_recommendation(severity)

    threading.Thread(
        target=save_prediction_async,
        args=(data, NUM_COLS, severity, confidence)
    ).start()

    response = {
        "severity": severity,
        "confidence": confidence,
        "recommendation": recommendation
    }

    if cache:
        cache.setex(cache_key, CACHE_TTL, json.dumps(response))

    return jsonify(response)

# -----------------------------------------------------
# RUN
# -----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
