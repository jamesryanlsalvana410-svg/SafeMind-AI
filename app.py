from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import LSTM
import joblib
import json

# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')

# -------------------- FIREBASE INIT --------------------
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- MODEL LOADING --------------------
model_path = 'model/safemind_model.h5'
tokenizer_path = 'model/tokenizer.json'
scaler_path = 'model/scaler.pkl'
le_path = 'model/label_encoder.pkl'

for path in [model_path, tokenizer_path, scaler_path, le_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚨 Required file not found: {path}")

def lstm_no_time_major(*args, **kwargs):
    kwargs.pop('time_major', None)
    return LSTM(*args, **kwargs)

model = load_model(model_path, custom_objects={'LSTM': lstm_no_time_major})
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

MAX_LEN = 100
print("✅ Model and preprocessing objects loaded successfully!")

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    # Retrieve assessments from Firestore
    assessments_ref = db.collection('assessments').stream()
    assessments = [
        {**a.to_dict(), 'id': a.id}
        for a in assessments_ref
    ]
    return render_template('phq9.html', assessments=assessments)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    answers = [int(request.form.get(f'q{i}')) for i in range(1, 10)]
    total = sum(answers)

    # Preprocess inputs
    scaled = np.array([answers]).reshape(1, -1)
    scaled = scaler.transform(scaled)
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

        scaled = np.array([answers]).reshape(1, -1)
        scaled = scaler.transform(scaled)
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

        # Store prediction in Firestore
        db.collection('api_predictions').add({
            'text': text,
            'severity': severity,
            'confidence': float(np.max(pred)),
            'timestamp': datetime.utcnow().isoformat()
        })

        return jsonify({
            'severity': severity,
            'confidence': float(np.max(pred))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- APP RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
