# mqtt_collector.py
import paho.mqtt.client as mqtt
import pandas as pd
import time
import threading

MQTT_TOPICS = [
    ("telecontact.adcon.TC1.WS", 0),
    ("telecontact.adcon.TC1.WD", 0),
    ("telecontact.adcon.TC1.TC", 0),
    ("telecontact.adcon.TC1.RAD", 0),
    ("telecontact.adcon.TC1.RH", 0),
    ("meshlium3d4c/DDNI/PM10", 0)
]

data_dict = {}

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    for topic, qos in MQTT_TOPICS:
        client.subscribe(topic, qos)

def on_message(client, userdata, msg):
    # Save the latest value for each topic with timestamp
    data_dict[msg.topic] = {
        'timestamp': pd.Timestamp.now(),
        'value': msg.payload.decode(errors='ignore')
    }

def save_data_periodically(interval=60):
    while True:
        if data_dict:
            # Convert to DataFrame and save
            df = pd.DataFrame([{**{'topic': k}, **v} for k, v in data_dict.items()])
            df.to_csv('mqtt_data.csv', index=False)
        time.sleep(interval)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("mqtt.beia-telemetrie.ro", 1883, 60)

# Start the periodic save in a background thread
threading.Thread(target=save_data_periodically, daemon=True).start()

client.loop_forever()