from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import json
import threading  # for async Firestore writes

# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -------------------- FIREBASE INIT --------------------
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- MODEL CONFIG --------------------
model_path = 'model/safemind_model.h5'
tokenizer_path = 'model/tokenizer.json'
scaler_path = 'model/scaler.pkl'
le_path = 'model/label_encoder.pkl'
MAX_LEN = 100

# 🔹 Lazy load cache
_model = None
_tokenizer = None
_scaler = None
_le = None

# -------------------- HELPER: Load model lazily --------------------
def get_model_components():
    """Load model and preprocessing objects only when needed."""
    global _model, _tokenizer, _scaler, _le

    if _model is None or _tokenizer is None or _scaler is None or _le is None:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        from tensorflow.keras.layers import LSTM

        # Fix for 'time_major' argument in custom LSTM
        def lstm_no_time_major(*args, **kwargs):
            kwargs.pop('time_major', None)
            return LSTM(*args, **kwargs)

        print("⏳ Loading AI model and preprocessing components...")
        _model = load_model(model_path, custom_objects={'LSTM': lstm_no_time_major})
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            _tokenizer = tokenizer_from_json(f.read())
        _scaler = joblib.load(scaler_path)
        _le = joblib.load(le_path)
        print("✅ Model and preprocessing objects loaded successfully!")

    return _model, _tokenizer, _scaler, _le


# -------------------- FIRESTORE ASYNC HELPER --------------------
def save_prediction_async(text, severity, confidence):
    """Save prediction to Firestore asynchronously."""
    try:
        db.collection('api_predictions').add({
            'text': text,
            'severity': severity,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        print("⚠️ Error saving prediction:", e)


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    assessments_ref = db.collection('assessments').stream()
    assessments = [{**a.to_dict(), 'id': a.id} for a in assessments_ref]
    return render_template('phq9.html', assessments=assessments)


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    answers = [int(request.form.get(f'q{i}')) for i in range(1, 10)]
    total = sum(answers)

    # Lazy load model components
    model, tokenizer, scaler, le = get_model_components()

    # Preprocess inputs
    scaled = np.array([answers]).reshape(1, -1)
    scaled = scaler.transform(scaled)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict severity
    pred = model.predict([seq, scaled])
    severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

    # Save to Firestore
    db.collection('assessments').add({
        'text': text,
        'severity': severity,
        'score': float(total),
        'created_at': datetime.utcnow().isoformat()
    })

    flash(f'AI Prediction: {severity} (PHQ-9 total: {total})')
    return redirect(url_for('dashboard'))


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'answers' not in data:
            return jsonify({'error': 'Missing "text" or "answers" in JSON body'}), 400

        text = data['text']
        answers = data['answers']

        # Lazy load model components
        model, tokenizer, scaler, le = get_model_components()
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Normalize answers length
        SCALER_FEATURES = 9
        while len(answers) < SCALER_FEATURES:
            answers.append(0)
        answers = answers[:SCALER_FEATURES]

        # Scale numeric input
        scaled = scaler.transform(np.array([answers]))

        # Ensure correct input shape for model
        MODEL_FEATURES = 10
        if scaled.shape[1] < MODEL_FEATURES:
            scaled = np.append(scaled, np.zeros((1, MODEL_FEATURES - scaled.shape[1])), axis=1)

        # Process text
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]
        confidence = float(np.max(pred))

        # Save prediction asynchronously
        threading.Thread(target=save_prediction_async, args=(text, severity, confidence)).start()

        return jsonify({
            'severity': severity,
            'confidence': confidence,
            'message': 'Prediction stored asynchronously'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------- APP RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
