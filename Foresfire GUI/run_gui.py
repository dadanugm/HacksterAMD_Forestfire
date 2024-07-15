import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import subprocess
import base64
import json
import io

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("1480x720")  # Set the size of the window

        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.start_mqtt_button = tk.Button(button_frame, text="Start MQTT Server", command=self.start_mqtt_server)
        self.start_mqtt_button.pack(fill=tk.X)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start)
        self.start_button.pack(fill=tk.X)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(fill=tk.X)

        self.view_button = tk.Button(button_frame, text="Load Image", command=self.load_image_from_data)
        self.view_button.pack(fill=tk.X)

        # Frame for table
        table_frame = tk.Frame(self.master)
        table_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.table = ttk.Treeview(table_frame, columns=('Connection Status', 'Topic', 'Values'), show='headings')
        self.table.heading('Connection Status', text='Connection Status', anchor=tk.W)
        self.table.heading('Topic', text='Topic', anchor=tk.W)
        self.table.heading('Values', text='Values', anchor=tk.W)
        self.table.pack()

        # Image loader
        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.running = False
        self.client = mqtt.Client()
        self.client.on_message = self.on_message

    def start_mqtt_server(self):
        self.mqtt_process = subprocess.Popen(["mosquitto", "-p", "1883"])

    def start(self):
        self.running = True
        self.client.connect("localhost", 1883, 60)
        self.client.subscribe("mqtt_data")
        self.client.loop_start()
        self.table.insert("", "end", values=("Connected", "", ""))  # Update the connection status
        print("Connected to MQTT server")

    def stop(self):
        self.running = False
        self.client.disconnect()
        self.client.loop_stop()
        self.table.insert("", "end", values=("Disconnected", "", ""))  # Update the connection status
        print("Disconnected from MQTT server")

    def load_image_from_data(self):
        if hasattr(self, 'last_image_data'):
            image_data = base64.b64decode(self.last_image_data)
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((640, 640))  # Resize the image to 640x640
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def on_message(self, client, userdata, message):
        data = json.loads(message.payload.decode())
        if "temperature" in data:
            self.table.insert("", "end", values=("","",f"Temperature: {data['temperature']}"))
        if "pressure" in data:
            self.table.insert("", "end", values=("","",f"Pressure: {data['pressure']}"))
        if "humidity" in data:
            self.table.insert("", "end", values=("","",f"Humidity: {data['humidity']}"))
        if "resistance" in data:
            self.table.insert("", "end", values=("","",f"Resistance: {data['resistance']}"))
        if "image_available" in data:
            self.table.insert("", "end", values=("","",f"Image Available: {data['image_available']}"))
        if "image_data" in data:
            self.last_image_data = data["image_data"]
        self.table.insert("", "end", values=("mqtt connection status", message.topic, ""))

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
