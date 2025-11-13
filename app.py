from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import LSTM
import joblib
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///safemind.db'

db = SQLAlchemy(app)

# -------------------- MODEL LOADING --------------------
model_path = 'model/safemind_model.h5'
tokenizer_path = 'model/tokenizer.json'
scaler_path = 'model/scaler.pkl'
le_path = 'model/label_encoder.pkl'

# Check files exist
for path in [model_path, tokenizer_path, scaler_path, le_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚨 Required file not found: {path}")

# Custom LSTM to ignore 'time_major' argument
def lstm_no_time_major(*args, **kwargs):
    kwargs.pop('time_major', None)
    return LSTM(*args, **kwargs)

# Load model and preprocessing
model = load_model(model_path, custom_objects={'LSTM': lstm_no_time_major})
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

MAX_LEN = 100
print("✅ Model and preprocessing objects loaded successfully!")

# -------------------- DATABASE MODELS --------------------
class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    severity = db.Column(db.String(40))
    score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    assessments = Assessment.query.all()
    return render_template('phq9.html', assessments=assessments)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    answers = [int(request.form.get(f'q{i}')) for i in range(1, 10)]
    total = sum(answers)

    # Preprocess structured input
    scaled = np.array([answers]).reshape(1, -1)
    scaled = scaler.transform(scaled)

    # Preprocess text input
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict
    pred = model.predict([seq, scaled])
    severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

    # Save assessment
    db.session.add(Assessment(text=text, severity=severity, score=float(total)))
    db.session.commit()

    flash(f'AI Prediction: {severity} (PHQ-9 total: {total})')
    return redirect(url_for('dashboard'))

# -------------------- POST API --------------------
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'answers' not in data:
            return jsonify({'error': 'Missing "text" or "answers" in JSON body'}), 400

        text = data['text']
        answers = data['answers']

        # Preprocess
        scaled = np.array([answers]).reshape(1, -1)
        scaled = scaler.transform(scaled)

        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        pred = model.predict([seq, scaled])
        severity = le.inverse_transform(np.argmax(pred, axis=1))[0]

        return jsonify({
            'severity': severity,
            'confidence': float(np.max(pred))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- APP RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=port)
