'''

Main program to run forest fire detections
AMD-Hackster project

Main Flow:
1. Detect PC-Camera
2. Use PC-Usb Camera to capture image
3. Using YoloV8 to do fire and smoke detection on the captured image
4. Request environment sensor Data
5. Send data (Image and environment sensor) to MQTT server

Requirements:
- Ultralytics YoloV8
- Paho Mqtt
- CV2

'''


from ultralytics import YOLO
import cv2
import time
import serial
import paho.mqtt.client as mqtt
import base64
import json
import random
from PIL import Image
import io

model = YOLO('forestfire_model.pt')  # load a pretrained model (recommended for training)
myserial = serial.Serial('COM3', 115200, timeout=1)
print(myserial.name)

# MQTT settings
mqtt_broker = "localhost"
mqtt_topic = "mqtt_data"
client = mqtt.Client()

def forestfire_detect():
    global detection_stat
    detection_stat = False
    names = model.names
    cam = cv2.VideoCapture(0)
    #time.sleep(2)
    img = cam.read()
    cv2.imwrite('image_cpt.jpg', img[1]) # save capture image
    results = model.predict("image_cpt.jpg", conf=0.35)  # predict on an image
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        confidence = ''.join(str(results[0].boxes.conf.tolist())[1:-1])
        print(confidence)
        for c in results[0].boxes.cls:
            print(names[int(c)]+" detected, confidence: " + confidence)
            detection_stat = True
        result.show()  # display to screen
        result.save(filename='detection_result.jpg')  # save to disk

def request_env_data():
    global sensor_data
    myserial.write(b'START\n')  # write a command to request env data
    time.sleep(5)
    for x in range(3):
        sensor_data = myserial.readline()
        print(sensor_data)
    time.sleep(1)

def send_payload():
    detection_stat = True
    data_str = sensor_data.decode('utf-8').strip()
    # Split the data into key-value pairs, ignoring the first numeric value
    _, key_value_pairs = data_str.split(', ', 1)
    # Split the remaining string into individual key-value pairs
    data_pairs = key_value_pairs.split(', ')
    time.sleep(1)
    parsed_data = {}
    # parsing env data
    # Loop through the key-value pairs and add them to the dictionary
    for pair in data_pairs:
        key, value = pair.split(": ")
        # Convert the value to its appropriate type
        if key in ['temperature', 'pressure', 'humidity', 'resistance']:
            value = float(value)
            parsed_data[key] = value
    # convert image to base64
    image_data = None
    if detection_stat:
        image_path = "detection_result.jpg"
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
    # Convert the dictionary to a JSON object
    parsed_data['image_available'] = detection_stat
    parsed_data['image_data'] = base64_image if detection_stat else None
    json_data = json.dumps(parsed_data, indent=4)
    # Print the JSON object
    print(json_data)
    # Connect to the MQTT broker
    client.connect(mqtt_broker)
    # Publish the JSON payload to the MQTT topic
    client.publish(mqtt_topic, json_data)
    # Disconnect from the MQTT broker
    client.disconnect()



if __name__ == "__main__":
    print(" Forest fire detection AMD-Hackster Main program")
    forestfire_detect()
    request_env_data()
    send_payload()


