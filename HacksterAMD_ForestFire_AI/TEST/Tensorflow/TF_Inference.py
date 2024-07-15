import cv2
import numpy as np
import tensorflow as tf
from official.vision.utils.object_detection import visualization_utils as vis_util

# Load the TensorFlow model
model_path = 'exported_models'
model = tf.saved_model.load(model_path)

# Define a function for inference
def model_fn(image):
    return model(image)

# Define the category index (mapping class IDs to class names)
category_index = {1: 'fire', 2: 'smoke'}  # Update with your actual category index

# Load an image
image_path = 'test_img/test1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
input_image_size = 640  # Assuming the input size for the model is 640x640
image_resized = cv2.resize(image_rgb, (input_image_size, input_image_size))
image_np = np.expand_dims(image_resized, axis=0)

# Perform inference
result = model_fn(tf.convert_to_tensor(image_np, dtype=tf.float32))

# Visualize the bounding boxes
min_score_thresh = 0.30
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    result['detection_boxes'][0].numpy(),
    result['detection_classes'][0].numpy().astype(int) + 1,  # Add 1 to match category index (assuming 0-based classes)
    result['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=min_score_thresh,
    line_thickness=2)

# Display the image with bounding boxes
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
