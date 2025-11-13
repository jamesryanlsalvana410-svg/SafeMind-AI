from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # Fixed import
from tensorflow.keras.layers import LSTM
import joblib
import threading

# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -------------------- FIREBASE INIT --------------------
try:
    cred = credentials.Certificate("firebase-key.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase connected successfully.")
except Exception as e:
    db = None
    print("⚠️ Firebase connection failed:", e)

# -------------------- MODEL LOADING --------------------
MODEL_DIR = "model"
model_path = os.path.join(MODEL_DIR, "safemind_model.h5")
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

for path in [model_path, tokenizer_path, scaler_path, le_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚨 Missing required model file: {path}")

def lstm_no_time_major(*args, **kwargs):
    """Compatibility fix for Keras LSTM layers."""
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(model_path, custom_objects={"LSTM": lstm_no_time_major})

with open(tokenizer_path, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

MAX_LEN = 100
SCALER_FEATURES = 9
MODEL_FEATURES = 10

print("✅ Model and preprocessing objects loaded successfully!")

# -------------------- FIRESTORE ASYNC HELPER --------------------
def save_prediction_async(text, severity, confidence):
    """Save prediction to Firestore asynchronously."""
    if not db:
        print("⚠️ Firestore not initialized — skipping save.")
        return
    try:
        db.collection("api_predictions").add({
            "text": text,
            "severity": severity,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"📝 Saved prediction for text: {text[:30]}...")
    except Exception as e:
        print("❌ Error saving prediction:", e)

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    """Displays PHQ-9 assessment results stored in Firestore."""
    if not db:
        flash("Firestore not connected.", "error")
        return render_template("phq9.html", assessments=[])
    
    assessments_ref = db.collection("assessments").stream()
    assessments = [{**a.to_dict(), "id": a.id} for a in assessments_ref]
    return render_template("phq9.html", assessments=assessments)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles PHQ-9 form submissions from frontend."""
    text = request.form.get("text", "")
    answers = [int(request.form.get(f"q{i}", 0)) for i in range(1, 10)]
    total = sum(answers)

    scaled = np.array([answers]).reshape(1, -1)
    scaled = scaler.transform(scaled)
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict([seq, scaled])
    severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

    if db:
        db.collection("assessments").add({
            "text": text,
            "severity": severity,
            "score": float(total),
            "created_at": datetime.utcnow().isoformat()
        })

    flash(f"AI Prediction: {severity} (PHQ-9 total: {total})")
    return redirect(url_for("dashboard"))

@app.route("/predict", methods=["POST"])
def predict_api():
    """API endpoint for external predictions."""
    try:
        data = request.get_json(force=True)
        if not data or "text" not in data or "answers" not in data:
            return jsonify({"error": 'Missing "text" or "answers" in JSON body'}), 400

        text = data["text"]
        answers = data["answers"]

        # Pad or truncate to match scaler features
        answers = (answers + [0] * SCALER_FEATURES)[:SCALER_FEATURES]
        scaled = scaler.transform(np.array([answers]))

        # Pad for model if necessary
        if scaled.shape[1] < MODEL_FEATURES:
            scaled = np.append(scaled, np.zeros((1, MODEL_FEATURES - scaled.shape[1])), axis=1)

        # Tokenize text input
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        # Prediction
        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]
        confidence = float(np.max(pred))

        # Save asynchronously
        threading.Thread(target=save_prediction_async, args=(text, severity, confidence)).start()

        return jsonify({
            "severity": severity,
            "confidence": confidence,
            "message": "Prediction stored asynchronously"
        })

    except Exception as e:
        print("❌ API Error:", e)
        return jsonify({"error": str(e)}), 500

# -------------------- APP RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)