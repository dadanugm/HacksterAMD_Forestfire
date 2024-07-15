from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8_trained_model/wildfire_yolov8n.pt")  # load a pretrained model (recommended for training)
#loadmodel = model.load("yolov8_trained_model/wildfire_yolov8n.pt")
# Use the model
model.train(data="datasets/data.yaml", epochs=1, save_dir="model/wildfire_yolov8n.pt")  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("test_img/test1.jpg")  # predict on an image
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
#success = model.save("/content/yolov8_trained_model/wildfire_yolov8m.pt")
#success = model.export(format="onnx")  # export the model to ONNX format