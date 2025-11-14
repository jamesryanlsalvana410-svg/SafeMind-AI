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
CSV_PATH = "Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "safemind_model.h5")
META_PATH = os.path.join(MODEL_DIR, "safemind_meta.json")

# === LOAD DATA ===
print("📥 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.\n")

# ======================================
# STEP 1: IDENTIFY TEXT & NUMERIC COLUMNS
# ======================================

# Remove obvious non-input columns
drop_like = ["id", "date", "time", "timestamp"]
cols_to_drop = [c for c in df.columns if any(d in c.lower() for d in drop_like)]
df = df.drop(columns=cols_to_drop, errors="ignore")

# Convert all numeric-looking columns properly
def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

numeric_candidates = []
text_candidates = []

for col in df.columns:
    if df[col].dtype == "object":
        # Try to convert object columns to numeric
        numeric_version = safe_to_numeric(df[col])
        ratio_nan = numeric_version.isna().mean()

        # If most values convert to numbers → treat as numeric
        if ratio_nan < 0.4:
            df[col] = numeric_version
            numeric_candidates.append(col)
        else:
            text_candidates.append(col)
    else:
        numeric_candidates.append(col)

print(f"🧾 Text columns detected: {text_candidates}")
print(f"🔢 Numeric columns detected: {numeric_candidates}\n")

# Combine text into a single column (if any)
if text_candidates:
    df["text_all"] = df[text_candidates].fillna("").agg(" ".join, axis=1)
else:
    df["text_all"] = ""


# ======================================
# STEP 2: IDENTIFY TARGET LABEL
# ======================================

possible_targets = [
    c for c in df.columns
    if re.search(r"(severity|label|diagnosis|target|class|phq)", c, re.I)
]

# Prefer a PHQ score if available
phq_candidates = [c for c in possible_targets if "phq" in c.lower()]

if phq_candidates:
    target_col = phq_candidates[0]
else:
    # Look for the first categorical column
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    target_col = cat_cols[0] if cat_cols else None

if target_col is None:
    print("⚠️ No label found — generating severity classes based on numeric totals.")
    total = df[numeric_candidates].fillna(0).sum(axis=1)
    q1, q2 = total.quantile([0.33, 0.66])
    df["severity"] = pd.cut(
        total,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "moderate", "high"]
    )
    target_col = "severity"

print(f"🎯 Using target column: {target_col}\n")

# Drop target from feature lists
if target_col in numeric_candidates:
    numeric_candidates.remove(target_col)
if target_col in text_candidates:
    text_candidates.remove(target_col)

# ======================================
# STEP 3: LABEL ENCODING
# ======================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_col].astype(str))

# ======================================
# STEP 4: TEXT PROCESSING
# ======================================
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text_all"])

X_text = pad_sequences(
    tokenizer.texts_to_sequences(df["text_all"]),
    maxlen=MAX_LEN,
    padding="post"
)

# ======================================
# STEP 5: NUMERIC PROCESSING
# ======================================
X_num = df[numeric_candidates].fillna(0)
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# ======================================
# SPLIT DATA
# ======================================
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# BUILD MODEL
# ======================================
print("🧠 Building combined LSTM + Dense model...")

# Text branch
text_input = Input(shape=(MAX_LEN,), name="text_input")
x = Embedding(MAX_WORDS, 128)(text_input)
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
x = Dense(64, activation='relu')(x)

# Numeric branch
num_input = Input(shape=(X_num.shape[1],), name="num_input")
y_num = Dense(64, activation='relu')(num_input)
y_num = Dropout(0.3)(y_num)

# Combine
combined = Concatenate()([x, y_num])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(len(label_encoder.classes_), activation='softmax')(z)

model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# ======================================
# TRAINING
# ======================================
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("🚀 Training started...")
history = model.fit(
    [X_train_text, X_train_num], y_train,
    validation_data=([X_test_text, X_test_num], y_test),
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# ======================================
# SAVE MODEL
# ======================================
model.save(MODEL_PATH)

meta = {
    "text_cols": text_candidates,
    "num_cols": numeric_candidates,
    "target_col": target_col,
    "tokenizer_params": {"num_words": MAX_WORDS, "max_len": MAX_LEN},
    "label_classes": label_encoder.classes_.tolist()
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"💾 Model saved to {MODEL_PATH}")
print(f"📝 Metadata saved to {META_PATH}")

# ======================================
# FINAL ACCURACY
# ======================================
loss, acc = model.evaluate([X_test_text, X_test_num], y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {acc:.3f}")
