import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# === Paths ===
CSV_PATH = "SafeMind_AI_Powered_Depression_Severity_Identification_with_Consultation.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
TOKENIZER_JSON_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

# === Load CSV and combine text columns ===
df = pd.read_csv(CSV_PATH)

# Combine all text columns into one
text_cols = df.select_dtypes(include='object').columns.tolist()
df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1) if text_cols else ""

texts = df["text_all"].tolist()

# === Recreate tokenizer ===
MAX_WORDS = 10000
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# === Save tokenizer as JSON ===
with open(TOKENIZER_JSON_PATH, "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

print(f"✅ tokenizer.json created at {TOKENIZER_JSON_PATH}")
