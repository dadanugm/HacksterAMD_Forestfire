'''
TESTING read image of fire
and perform inference using YOLOV8
Store the result to disk
'''
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO('model/forestfire_model.pt')  # load a pretrained model (recommended for training)
names = model.names
# Use the model
#results = model.train(data='coco128.yaml', epochs=3, imgsz=640)
results = model.predict("test_img/test1.jpg", conf=0.35)  # predict on an image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    detection_count = result.boxes.shape[0]
    for i in range(detection_count):
        cls = int(result.boxes.cls[i].item())
        name = result.names[cls]
        confidence = float(result.boxes.conf[i].item())
        print("class: "+name+" confidence: "+str(confidence))
        #bounding_box = result.boxes.xyxy[i].cpu().numpy()

    result.show()  # display to screen
    result.save(filename='test_img/result.jpg')  # save to disk