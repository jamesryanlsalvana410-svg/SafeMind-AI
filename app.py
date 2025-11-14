import os
import json
import threading
import hashlib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
import redis
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import firebase_admin
from firebase_admin import credentials, firestore

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
# MODEL + TOKENIZER
# -----------------------------
MODEL_DIR = "model"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model_v2.tflite")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")

# Load metadata
with open(META_PATH, "r") as f:
    meta = json.load(f)

TEXT_COLS = meta["text_cols"]
NUM_COLS = meta["num_cols"]
LABEL_CLASSES = meta["label_classes"]
MAX_LEN = meta["tokenizer_params"]["max_len"]

# Load tokenizer
tokenizer = joblib.load(TOKENIZER_PATH)
print("✅ Tokenizer loaded.")

# -----------------------------
# LOAD TFLITE MODEL WITH FLEX DELEGATE
# -----------------------------
interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL_PATH,
    experimental_delegates=[tf.lite.experimental.load_delegate("libtensorflowlite_flex_delegate.so")]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded with Flex delegate.")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def preprocess_and_predict(input_dict):
    # Text preprocessing
    text_string = " ".join([str(input_dict.get(col, "")) for col in TEXT_COLS])
    seq = tokenizer.texts_to_sequences([text_string])
    seq = pad_sequences(seq, maxlen=MAX_LEN, dtype=np.int32)

    # Numeric preprocessing
    numeric_list = [float(input_dict.get(col, 0)) for col in NUM_COLS]
    X_num = np.array(numeric_list, dtype=np.float32).reshape(1, -1)

    # Set tensors and run inference
    interpreter.set_tensor(input_details[0]['index'], seq)
    interpreter.set_tensor(input_details[1]['index'], X_num)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    idx = int(pred.argmax())
    severity = LABEL_CLASSES[idx]
    confidence = float(pred.max())

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

@app.route("/predict", methods=["POST"])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    input_data = request.get_json()
    cache_key = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()

    # Return cached result if exists
    if cache and cache.get(cache_key):
        return jsonify(json.loads(cache.get(cache_key)))

    try:
        severity, confidence = preprocess_and_predict(input_data)
        recommendation = get_recommendation(severity)

        result = {
            "severity": severity,
            "confidence": confidence,
            "recommendation": recommendation
        }

        # Cache result
        if cache:
            cache.setex(cache_key, CACHE_TTL, json.dumps(result))

        # Save asynchronously
        threading.Thread(target=save_prediction_async, args=(input_data, result), daemon=True).start()

        return jsonify(result)

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
