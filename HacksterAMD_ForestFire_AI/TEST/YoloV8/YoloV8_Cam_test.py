'''
TESTING Capture image from webcam
store the image to disk,
and perform inference using YOLOV8
'''

from ultralytics import YOLO
import cv2
import time

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO('model/forestfire_model.pt')  # load a pretrained model (recommended for training)
names = model.names
#capture image with camera
cam = cv2.VideoCapture(0)
img = cam.read()
#time.sleep(1)
img_save = cv2.imwrite('capture_img/opencv_cam.jpg', img[1])

#cv2.imshow("frame", img[1])
# Use the model for inference
results = model.predict("capture_img/opencv_cam.jpg", conf=0.35)  # predict on an image


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    print(results[0].boxes.conf)
    #print(results[0].boxes.cls)
    for c in results[0].boxes.cls:
        print(names[int(c)])
    result.show()  # display to screen
    result.save(filename='capture_img/result.jpg')  # save to disk
