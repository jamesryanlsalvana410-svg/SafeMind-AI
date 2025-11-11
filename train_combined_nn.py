import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIG ===
CSV_PATH = "SafeMind_AI_Powered_Depression_Severity_Identification_with_Consultation.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model.h5")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")

# === LOAD DATA ===
print("📥 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.")

# === IDENTIFY COLUMNS ===
text_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

print(f"🧾 Text columns: {text_cols}")
print(f"🔢 Numeric columns: {num_cols}")

# Use first text column or combine multiple
df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1) if text_cols else ""

# Find or create label column
cand_targets = [c for c in df.columns if re.search(r"(severity|label|diagnosis|target|class)", str(c), re.I)]
target_col = cand_targets[0] if cand_targets else None

if target_col is None:
    print("⚠️ No label column found — auto-generating labels from numeric total.")
    total = df[num_cols].fillna(0).sum(axis=1)
    q1, q2 = total.quantile([0.33, 0.66])
    df["severity"] = pd.cut(total, bins=[-np.inf, q1, q2, np.inf], labels=["low", "moderate", "high"])
    target_col = "severity"

# === PREPARE LABELS ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_col].astype(str))

# === TEXT PREPROCESSING ===
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text_all"])
X_text = pad_sequences(tokenizer.texts_to_sequences(df["text_all"]), maxlen=MAX_LEN, padding='post')

# === NUMERIC PREPROCESSING ===
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_cols].fillna(0))

# === SPLIT DATA ===
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# === BUILD MODEL ===
print("🧠 Building combined LSTM + Dense model...")

# Text input branch
text_input = Input(shape=(MAX_LEN,), name="text_input")
x = Embedding(MAX_WORDS, 128)(text_input)
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
x = Dense(64, activation='relu')(x)

# Numeric input branch
num_input = Input(shape=(X_num.shape[1],), name="num_input")
y_num = Dense(64, activation='relu')(num_input)
y_num = Dropout(0.3)(y_num)

# Combine both
combined = Concatenate()([x, y_num])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(len(label_encoder.classes_), activation='softmax')(z)

model = Model(inputs=[text_input, num_input], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🚀 Training started...")
history = model.fit(
    [X_train_text, X_train_num], y_train,
    validation_data=([X_test_text, X_test_num], y_test),
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# === SAVE MODEL & METADATA ===
model.save(MODEL_PATH)

meta = {
    "text_cols": text_cols,
    "num_cols": num_cols,
    "tokenizer_params": {"num_words": MAX_WORDS, "max_len": MAX_LEN},
    "label_classes": label_encoder.classes_.tolist()
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"💾 Model saved to {MODEL_PATH}")
print(f"📝 Metadata saved to {META_PATH}")

# === EVALUATION ===
loss, acc = model.evaluate([X_test_text, X_test_num], y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {acc:.3f}")