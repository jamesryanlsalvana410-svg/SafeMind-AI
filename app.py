from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import json
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences, tokenizer_from_json
from tensorflow.keras.layers import LSTM

# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -------------------- FIREBASE INIT --------------------
try:
    # Use environment variable for Render safety (avoid storing firebase-key.json in repo)
    firebase_json = os.getenv("FIREBASE_KEY_JSON")
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("firebase-key.json")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print("⚠️ Firebase initialization failed:", e)
    db = None

# -------------------- MODEL CONFIG --------------------
MODEL_PATH = "model/safemind_model.h5"
TOKENIZER_PATH = "model/tokenizer.json"
SCALER_PATH = "model/scaler.pkl"
LE_PATH = "model/label_encoder.pkl"
MAX_LEN = 100

# -------------------- LOAD MODEL ON STARTUP --------------------
print("⏳ Loading AI model and preprocessing components...")
try:
    # Fix for LSTM time_major bug
    def lstm_no_time_major(*args, **kwargs):
        kwargs.pop("time_major", None)
        return LSTM(*args, **kwargs)

    model = load_model(MODEL_PATH, custom_objects={"LSTM": lstm_no_time_major})
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LE_PATH)
    print("✅ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model, tokenizer, scaler, le = None, None, None, None


# -------------------- FIRESTORE ASYNC HELPER --------------------
def save_prediction_async(text, severity, confidence):
    if not db:
        return
    try:
        db.collection("api_predictions").add({
            "text": text,
            "severity": severity,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print("⚠️ Error saving prediction:", e)


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    if not db:
        return "⚠️ Database unavailable.", 500
    assessments_ref = db.collection("assessments").stream()
    assessments = [{**a.to_dict(), "id": a.id} for a in assessments_ref]
    return render_template("phq9.html", assessments=assessments)


@app.route("/analyze", methods=["POST"])
def analyze():
    if not model or not db:
        flash("Model or database not available.")
        return redirect(url_for("dashboard"))

    try:
        text = request.form.get("text")
        answers = [int(request.form.get(f"q{i}")) for i in range(1, 10)]
        total = sum(answers)

        scaled = scaler.transform(np.array([answers]))
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

        db.collection("assessments").add({
            "text": text,
            "severity": severity,
            "score": float(total),
            "created_at": datetime.utcnow().isoformat()
        })

        flash(f"AI Prediction: {severity} (PHQ-9 total: {total})")
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("dashboard"))


@app.route("/predict", methods=["POST"])
def predict_api():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or "text" not in data or "answers" not in data:
            return jsonify({"error": 'Missing "text" or "answers" in JSON body'}), 400

        text = data["text"]
        answers = data["answers"]

        while len(answers) < 9:
            answers.append(0)
        answers = answers[:9]

        scaled = scaler.transform(np.array([answers]))
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]
        confidence = float(np.max(pred))

        threading.Thread(target=save_prediction_async, args=(text, severity, confidence)).start()

        return jsonify({
            "severity": severity,
            "confidence": confidence,
            "message": "Prediction stored asynchronously"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)