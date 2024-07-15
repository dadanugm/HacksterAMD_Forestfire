import tensorflow as tf
import tf2onnx

# Path to the TensorFlow SavedModel directory
saved_model_path = 'C:/Users/dadan/Documents/HacksterAMD_ForestFire_AI/TEST/YoloV8/model'

# Path to save the converted ONNX model
onnx_model_path = 'C:/Users/dadan/Documents/HacksterAMD_ForestFire_AI/TEST/YoloV8/model'

# Load the TensorFlow SavedModel
model = tf.saved_model.load(saved_model_path)

# Convert the TensorFlow model to ONNX format
onnx_model, _ = tf2onnx.convert.from_saved_model(saved_model_path, opset=13)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved at:", onnx_model_path)
