import cv2
import numpy as np
import onnxruntime as ort

# Path to the ONNX model file
onnx_model_path = 'model/tf_model.onnx'

# Path to the image file
path_to_image = 'test_img/test1.jpg'

# Load the ONNX model with CPUExecutionProvider
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Read the image
image = cv2.imread(path_to_image)

# Preprocess the image
input_image = cv2.resize(image, (640, 640))  # Assuming your model expects 640x640 images
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = np.expand_dims(input_image.astype(np.float32) / 255.0, axis=0)

# Run inference
model_inputs = ort_session.get_inputs()
for inp in model_inputs:
    print("- Name: {}, Shape: {}".format(inp.name, inp.shape))

outputs = ort_session.run(None, {model_inputs[0].name: input_image})

# Process the outputs to get bounding box coordinates
# Assuming the output is a list containing the bounding box coordinates for each detected object
# Each bounding box is represented as [class_id, confidence, x_min, y_min, x_max, y_max]
# You may need to adjust the post-processing based on your model's output format

# Draw bounding boxes on the image
for box in outputs[0]:
    class_id, confidence, x_min, y_min, x_max, y_max = box.astype(int)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Display the image
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
