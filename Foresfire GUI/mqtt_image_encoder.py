import paho.mqtt.client as mqtt
import base64
import json
import random
from PIL import Image
import io

# Generate random data
temperature = round(random.uniform(20.0, 30.0), 2)
humidity = round(random.uniform(40.0, 60.0), 2)
gas = round(random.uniform(100, 500), 2)
image_available = random.choice([True, False])

# If image is available, load and convert to base64
image_data = None
if image_available:
    image_path = "Path to JPG"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

# Create JSON payload
payload = {
    "temperature": temperature,
    "humidity": humidity,
    "gas": gas,
    "image_available": image_available,
    "image_data": base64_image if image_available else None
}

# MQTT settings
mqtt_broker = "localhost"
mqtt_topic = "mqtt_data"

# Connect to the MQTT broker
client = mqtt.Client()
client.connect(mqtt_broker)

# Publish the JSON payload to the MQTT topic
client.publish(mqtt_topic, json.dumps(payload))

# Disconnect from the MQTT broker
client.disconnect()
