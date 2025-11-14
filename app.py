import os
import json
import threading
import hashlib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import redis
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
import tensorflow as tf  # Added for optimizations

# -----------------------------
# FORCE CPU (if no GPU available) - COMMENTED OUT TO ALLOW GPU IF AVAILABLE
# -----------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment if you want to force CPU

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -----------------------------
# CLINICAL RECOMMENDATIONS
# -----------------------------
RECOMMENDATIONS = {
    "low": "Mild symptoms. Maintain healthy habits and monitor.",
    "moderate": "Moderate symptoms. Consider professional support.",
    "high": "Severe symptoms. Seek professional help immediately."
}

def get_recommendation(severity):
    return RECOMMENDATIONS.get(severity, "No recommendation available.")

# -----------------------------
# FIREBASE SETUP
# -----------------------------
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase-key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# REDIS CACHE
# -----------------------------
REDIS_URL = os.getenv("REDIS_URL")
cache = redis.from_url(REDIS_URL) if REDIS_URL else None
CACHE_TTL = 3600  # 1 hour

# -----------------------------
# MODEL + METADATA PATHS
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model_v2.keras")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")

# -----------------------------
# LOAD METADATA + TOKENIZER
# -----------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

TEXT_COLS = meta["text_cols"]
NUM_COLS = meta["num_cols"]
LABEL_CLASSES = meta["label_classes"]
MAX_LEN = meta["tokenizer_params"]["max_len"]

tokenizer = joblib.load(TOKENIZER_PATH)
print("✅ Tokenizer loaded.")

# -----------------------------
# LOAD MODEL
# -----------------------------
def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(MODEL_PATH, compile=False, custom_objects={"LSTM": lstm_no_time_major})
# OPTIMIZATION: Wrap in tf.function for faster inference (graph mode)
model = tf.function(model)
print("✅ Model loaded and optimized.")

# -----------------------------
# PREDICTION FUNCTION (ENHANCED CACHING)
# -----------------------------
def preprocess_and_predict(input_dict):
    # Text processing with caching
    text_string = " ".join([str(input_dict.get(col, "")) for col in TEXT_COLS])
    text_hash = hashlib.sha256(text_string.encode()).hexdigest()
    seq_cache_key = f"seq_{text_hash}"
    
    if cache and cache.get(seq_cache_key):
        seq = np.array(json.loads(cache.get(seq_cache_key)))
    else:
        seq = tokenizer.texts_to_sequences([text_string])
        seq = pad_sequences(seq, maxlen=MAX_LEN)
        if cache:
            cache.setex(seq_cache_key, CACHE_TTL, json.dumps(seq.tolist()))
    
    # Numeric processing (can add caching here if inputs vary little)
    numeric_list = [float(input_dict.get(col, 0)) for col in NUM_COLS]
    X_num = np.array(numeric_list).reshape(1, -1)
    
    # Predict
    pred = model.predict([seq, X_num], verbose=0)
    idx = int(np.argmax(pred))
    severity = LABEL_CLASSES[idx]
    confidence = float(np.max(pred))
    return severity, confidence

# -----------------------------
# FIRESTORE ASYNC SAVE
# -----------------------------
def save_prediction_async(input_data, result):
    try:
        db.collection("api_predictions").add({
            "input_data": input_data,
            **result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print("❌ Firestore Async Error:", e)

# -----------------------------
# ROUTES
# -----------------------------
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

# -----------------------------
# API: PREDICT (Instant Response)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    start_total = time.time()
    
    # Log initial resource usage
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().percent
    
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400
    
    # Add input size limit to prevent large payloads (adjust as needed)
    if request.content_length and request.content_length > 10 * 1024 * 1024:  # 10MB limit
        return jsonify({"error": "Input too large"}), 413
    
    input_data = request.get_json()
    cache_key = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    
    # Return cached result if exists
    if cache and cache.get(cache_key):
        cached_result = json.loads(cache.get(cache_key))
        print(f"Cache hit for key {cache_key[:8]}... Total time: {time.time() - start_total:.2f}s")
        return jsonify(cached_result)
    
    try:
        # Time preprocessing and prediction
        start_predict = time.time()
        severity, confidence = preprocess_and_predict(input_data)  # Pass model if needed: preprocess_and_predict(input_data, model)
        predict_time = time.time() - start_predict
        print(f"Preprocess and predict took: {predict_time:.2f}s")
        
        # Time recommendation
        start_rec = time.time()
        recommendation = get_recommendation(severity)
        rec_time = time.time() - start_rec
        print(f"Recommendation took: {rec_time:.2f}s")
        
        result = {
            "severity": severity,
            "confidence": confidence,
            "recommendation": recommendation
        }
        
        # Cache the result (your original code was missing this!)
        if cache:
            cache.set(cache_key, json.dumps(result), timeout=3600)  # Cache for 1 hour
        
        # Log total time and resource usage
        total_time = time.time() - start_total
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent
        print(f"Total response time: {total_time:.2f}s | CPU: {cpu_before:.1f}% -> {cpu_after:.1f}% | Mem: {mem_before:.1f}% -> {mem_after:.1f}%")
        
        return jsonify(result)
    
    except Exception as e:
        # Log errors to prevent silent failures
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Enable Flask threaded mode for multiple simultaneous requests
    app.run(host="0.0.0.0", port=port, threaded=True)
