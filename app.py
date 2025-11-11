from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import LSTM
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'devkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///safemind.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

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

# Load Keras model
model = load_model(model_path, custom_objects={'LSTM': lstm_no_time_major})

# Load tokenizer from JSON
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())

# Load scaler and label encoder
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

MAX_LEN = 100
print("✅ Model and preprocessing objects loaded successfully!")

# -------------------- DATABASE MODELS --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    text = db.Column(db.Text)
    severity = db.Column(db.String(40))
    score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid): 
    return User.query.get(int(uid))

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard')) if current_user.is_authenticated else redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        if User.query.filter_by(username=u).first():
            flash('User exists')
            return redirect(url_for('register'))
        hashed = bcrypt.generate_password_hash(p).decode('utf-8')
        db.session.add(User(username=u, password=hashed))
        db.session.commit()
        flash('Registered! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        user = User.query.filter_by(username=u).first()
        if user and bcrypt.check_password_hash(user.password, p):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    assessments = Assessment.query.filter_by(user_id=current_user.id).all()
    return render_template('phq9.html', assessments=assessments)

@app.route('/analyze', methods=['POST'])
@login_required
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
    db.session.add(Assessment(user_id=current_user.id, text=text, severity=severity, score=float(total)))
    db.session.commit()

    flash(f'AI Prediction: {severity} (PHQ-9 total: {total})')
    return redirect(url_for('dashboard'))

# -------------------- APP RUN --------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
