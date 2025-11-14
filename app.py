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
from tensorflow.keras import Model

# -----------------------------------------------------
# APP SETUP
# -----------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -----------------------------------------------------
# CLINICAL RECOMMENDATIONS
# -----------------------------------------------------
RECOMMENDATIONS = {
    "low": (
        "Your symptoms appear to be mild. Continue practicing healthy habits such as "
        "sleep hygiene, physical activity, and connecting with supportive people. "
        "Monitor your symptoms and seek help if they worsen or persist."
    ),
    "moderate": (
        "Your symptoms appear to be moderate. It is recommended to speak with a "
        "mental health professional or counselor for further evaluation. Early support "
        "can help prevent symptoms from getting worse."
    ),
    "high": (
        "Your symptoms appear to be severe. Please seek professional help as soon as "
        "possible. If you are experiencing thoughts of self-harm or harming others, "
        "contact emergency services or your local crisis hotline immediately."
    )
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
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model_v2.keras")   # Use Keras 3 native format
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")

# -----------------------------------------------------
# LOAD METADATA
# -----------------------------------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

TEXT_COLS = meta["text_cols"]
NUM_COLS = meta["num_cols"]
LABEL_CLASSES = meta["label_classes"]
MAX_LEN = meta["tokenizer_params"]["max_len"]

# -----------------------------------------------------
# LOAD TOKENIZER
# -----------------------------------------------------
tokenizer = joblib.load(TOKENIZER_PATH)
print("✅ Tokenizer loaded successfully.")

# -----------------------------------------------------
# FIX LSTM TIME_MAJOR
# -----------------------------------------------------
def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
model = load_model(MODEL_PATH, compile=False, custom_objects={"LSTM": lstm_no_time_major})
print("✅ Model loaded successfully.")

# -----------------------------------------------------
# BACKGROUND FIRESTORE SAVE
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
        print("❌ Async Firestore Error:", e)

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
    input_dict = {col: request.form.get(col, "") for col in TEXT_COLS}
    for col in NUM_COLS:
        input_dict[col] = request.form.get(col, 0)

    severity, confidence = preprocess_and_predict(input_dict)
    recommendation = get_recommendation(severity)

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
    cache_key = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    if cache and cache.get(cache_key):
        return jsonify(json.loads(cache.get(cache_key)))

    severity, confidence = preprocess_and_predict(data)
    recommendation = get_recommendation(severity)

    threading.Thread(target=save_prediction_async, args=(data, NUM_COLS, severity, confidence)).start()

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
