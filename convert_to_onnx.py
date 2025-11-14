import tf2onnx
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

# Load your model
MODEL_PATH = "model/safemind_model_v2.keras"

def lstm_no_time_major(*args, **kwargs):
    kwargs.pop("time_major", None)
    return LSTM(*args, **kwargs)

model = load_model(MODEL_PATH, compile=False, custom_objects={"LSTM": lstm_no_time_major})

# Define input signatures (adjust based on your model's inputs)
# Check your meta.json for MAX_LEN and len(NUM_COLS), or run model.summary()
# Example: If MAX_LEN=100 and NUM_COLS has 5 features
spec = (
    tf.TensorSpec((None, 100), tf.int32, name="text_input"),  # Replace 100 with your MAX_LEN
    tf.TensorSpec((None, 5), tf.float32, name="num_input")    # Replace 5 with len(NUM_COLS)
)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save ONNX model
ONNX_PATH = "model/safemind_model_v2.onnx"
with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ ONNX model saved to {ONNX_PATH}")
