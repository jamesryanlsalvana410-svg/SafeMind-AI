import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import os

# Load your model (same as in app.py)
MODEL_PATH = "model/safemind_model_v2.keras"

def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(MODEL_PATH, compile=False, custom_objects={"LSTM": lstm_no_time_major})

# Convert to TFLite with fixes for LSTM
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the TFLite model
TFLITE_PATH = "model/safemind_model_v2.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to {TFLITE_PATH}")