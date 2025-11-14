import tensorflow as tf

model_path = "model/safemind_model_v2.h5"
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Fallback to TF ops for unsupported ops like LSTM dynamic TensorArray
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,      # default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS         # select TF ops
]

# Optional: do not lower TensorList ops (may help)
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("model/safemind_model_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved with SELECT_TF_OPS fallback")
