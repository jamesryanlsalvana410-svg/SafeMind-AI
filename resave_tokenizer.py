import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM

# -------------------------------
# PATHS
# -------------------------------
CSV_PATH = "Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model_v2.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
print("📥 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.")

# -------------------------------
# IDENTIFY TEXT COLUMNS
# -------------------------------
text_cols = [c for c in df.columns if df[c].dtype == "object"]

if text_cols:
    df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1)
else:
    df["text_all"] = ""

# -------------------------------
# CREATE TOKENIZER
# -------------------------------
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text_all"])

# Save tokenizer
joblib.dump(tokenizer, TOKENIZER_PATH, protocol=4)
print(f"✅ Tokenizer saved at {TOKENIZER_PATH}")

# -------------------------------
# LOAD MODEL
# -------------------------------
def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

# -------------------------------
# LOAD METADATA
# -------------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

TEXT_COLS = meta["text_cols"]
NUM_COLS = meta["num_cols"]
LABEL_CLASSES = meta["label_classes"]
MAX_LEN = meta["tokenizer_params"]["max_len"]

# -------------------------------
# TEST PREDICTION
# -------------------------------
# Create dummy input
dummy_text = "I feel sad and anxious today."
dummy_numeric = [0.0 for _ in NUM_COLS]  # all numeric inputs = 0

# Preprocess text
seq = tokenizer.texts_to_sequences([dummy_text])
seq = pad_sequences(seq, maxlen=MAX_LEN)

# Numeric array
X_num = np.array(dummy_numeric).reshape(1, -1)

# Predict
pred = model.predict([seq, X_num], verbose=0)
idx = int(np.argmax(pred))
severity = LABEL_CLASSES[idx]
confidence = float(np.max(pred))

print(f"\n🎯 Dummy Prediction Test:")
print(f"Severity: {severity}")
print(f"Confidence: {confidence:.3f}")