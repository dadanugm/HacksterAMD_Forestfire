# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import cv2
import tabulate as tab
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
import numpy as np
from PIL import Image
import onnxruntime_extensions
import onnxruntime as ort
from pathlib import Path


confidence_thresh = 0.2
score_thresh = 0.2
iou_thresh = 0.2

# Load the class names from the COCO dataset
classes = yaml_load(check_yaml("datasets/data.yaml"))["names"]
color_palette = np.random.uniform(0, 255, size=(len(classes), 3))


def draw_detections(img, box, score, class_id):
    """
            Draws bounding boxes and labels on the input image based on the detected objects.

            Args:
                img: The input image to draw detections on.
                box: Detected bounding box.
                score: Corresponding detection score.
                class_id: Class ID for the detected object.

            Returns:
                None
            """

    # Extract the coordinates of the bounding box
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    try:
        color = color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    except:
        print ('color out of bounds')

def postprocess(input_image, output):
    """
            Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

            Args:
                input_image (numpy.ndarray): The input image.
                output (numpy.ndarray): The output of the model.

            Returns:
                numpy.ndarray: The input image with detections drawn on it.
            """

    img_height, img_width, _ = input_image.shape

    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output[0]))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = img_width / 640
    y_factor = img_height / 640

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thresh:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_thresh, iou_thresh
    )

    table = [["Class-ID", "Score", "xmin", "ymin", "xmax", "ymax"]]
    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        table.append([class_id, score, box[1], box[0], box[3], box[2]])
        # Draw the detection on the input image
        draw_detections(input_image, box, score, class_id)
    table_data = tab.tabulate(table, headers="firstrow", tablefmt="grid")
    with open("output.txt", "w") as fp:
        fp.write(table_data)
    print(tab.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    # Return the modified input image
    return input_image

def run_inference(onnx_model_file: Path, images_path):
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    #image = np.frombuffer(open('test_img/test1.jpg', 'rb').read(), dtype=np.uint8)
    orig_img = cv2.imread(images_path)
    output = cv2.resize(orig_img, (640, 640))
    # Normalize the image data by dividing it by 255.0
    image = np.array(output) / 255.0
    # Transpose the image to have the channel dimension as the first dimension
    image = np.transpose(image, (2, 0, 1))  # Channel first
    # Expand the dimensions of the image data to match the expected input shape
    image = np.expand_dims(image, axis=0).astype(np.float32)

    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    model_inputs = session.get_inputs()
    for inp in model_inputs:
        print("- Name: {}, Shape: {}".format(inp.name, inp.shape))

    output = session.run(None, {model_inputs[0].name: image})
    output_img = postprocess(orig_img, output)
    print(type(output_img))
    output_filename = 'test_img/result_onnx.jpg'
    #open(output_filename, 'wb').write(output_img)
    img_data = Image.fromarray(output_img)
    img_data.save(output_filename)
    Image.open(output_filename).show()

if __name__ == '__main__':
    print("Testing updated model...")
    onnx_e2e_model_name = Path(f"model/forestfire_model.pre_post_process.onnx")
    onnx_model_name = Path(f"model/forestfire_model.onnx")
    onnx_tf_model_name = Path(f"model/tf_model.onnx")
    #run_inference(onnx_e2e_model_name, 'test_img/test1.jpg')
    run_inference(onnx_model_name, 'test_img/test1.jpg')
    #run_inference(onnx_tf_model_name, 'test_img/test1.jpg')