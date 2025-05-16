import serial
import numpy as np
import time

import paho.mqtt.client as mqtt_client
import random

broker = '127.0.0.1'
port = 1883
# topic = "/test"
client_id = f'python-mqtt-{random.randint(0, 1000)}'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, topic, msg):
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

client = connect_mqtt()
client.loop_start()

END_MARKER = 0x7E

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0)  # open serial port
print(ser.name)         # check which port was really used
time.sleep(2)

newData = False

user = -1
gesture = "Leave"
activity = "None"

tags_count = np.zeros((10,2), dtype = np.int32)
tags_rssi = np.empty((10,2,100))
tags_rssi[:] = np.nan

while 1: 
    loop_end_time = time.monotonic() + 2
    # print(loop_end_time)
    
    # Handle data and predict human behavior
    tags_rssi_max = np.nanmax(tags_rssi, axis = 2)
    print(tags_count[0:3][:])
    print(tags_rssi_max[0:3][:])

    # Consider the user with highest tag count as the user that is currently using the table
    tags_count_argmax = np.argmax(np.sum(tags_count, axis = 0), axis = 0)
    tags_count_user = tags_count[:, tags_count_argmax] # (10, )
    tags_rssi_max_user = tags_rssi_max[:, tags_count_argmax] # (10, )
    user = tags_count_argmax

    # Generate an array that show whether one type of tag appears or not
    tags_count_bool = tags_count > 0

    # Algorithm (for posture) 
    if (tags_count_user[2] > 1 and tags_rssi_max_user[2] > -50):
        gesture = "sleep"
    elif (tags_count_user[0] > 1 and tags_count_user[1] == 0): 
        gesture = "stand"
    elif (tags_count_user[0] == 0 and tags_count_user[1] > 1): 
        gesture = "sit"
    elif (tags_count_user[0] > 0 and tags_count_user[1] > 0 and tags_rssi_max_user[0] - tags_rssi_max_user[1] > 2): 
        gesture = "stand"
    elif (tags_count_user[0] > 0 and tags_count_user[1] > 0 and tags_rssi_max_user[1] - tags_rssi_max_user[0] > 2): 
        gesture = "sit"

    # Algorithm (for activity)
    if (tags_count_user[3] > 1 and tags_count_user[4] == 0): 
        activity = "Read Book"
    elif (tags_count_user[3] == 0 and tags_count_user[4] > 1): 
        activity = "Look Phone"
    elif (tags_count_user[3] > 0 and tags_count_user[4] > 0 and tags_rssi_max_user[3] > tags_rssi_max_user[4]): 
        activity = "Read Book"
    elif (tags_count_user[3] > 0 and tags_count_user[4] > 0 and tags_rssi_max_user[3] < tags_rssi_max_user[4]): 
        activity = "Look Phone"
    else:
        activity = "None"


    # Print (and send) Result
    print(user)
    print(gesture)
    print(activity)
    publish(client, "/har/posture", gesture)
    publish(client, "/har/posture/tag_count", np.array2string(tags_count[0:3][:]))
    publish(client, "/har/posture/tag_rssi", np.array2string(tags_rssi_max[0:3][:]))

    publish(client, "/har/activity", activity)
    publish(client, "/har/activity/tag_count", np.array2string(tags_count[3:5][:]))
    publish(client, "/har/activity/tag_rssi", np.array2string(tags_rssi_max[3:5][:]))

    publish(client, "/har/user", str(user))

    # Clean data z
    tags_count[:] = 0
    tags_rssi[:] = np.nan

    # Start new inventory
    ser.write(b'\xBB\x00\x27\x00\x03\x22\x00\x20\x6C\x7E')  # 20 Inventory
    ser.flush()

    result_buffer = []

    # When still within the time limit of this loop, continue to receive data (inventory result)
    while (time.monotonic() < loop_end_time):
        # print(time.monotonic())
        in_bin = ser.read() # in_bin is in byte type

        if (len(in_bin) > 0):
            if (int.from_bytes(in_bin,byteorder='little') == END_MARKER):
                # print(result_buffer)
                # Handle this observation and add back to the global data array
                if (result_buffer[1] == 0x02): # If a tag is found
                    tag_id = result_buffer[19] # We only use the last byte of the ID as the tag ID
                    # print(tag_id)
                    tag_type = (int)(tag_id / 16)
                    tag_user = (int)(tag_id % 16)
                    # print(tag_type)
                    # print(tag_type)

                    if (tag_type > 9 or tag_user > 1): continue # Throw away tag with invalid id

                    rssi = result_buffer[5] - 256

                    current_count = tags_count[tag_type][tag_user]
                    tags_rssi[tag_type][tag_user][current_count] = rssi
                    tags_count[tag_type][tag_user] = current_count + 1

                result_buffer.clear()
                continue

            result_buffer.append(int.from_bytes(in_bin,byteorder='little')) 

        # in_hex = hex(int.from_bytes(in_bin,byteorder='little')) 
        # print(in_hex)
    
client.loop_stop()

# ser.flush()

# result = ser.readline()
# print(result)

# ser.flush()

# result = ser.readline()
# print(result)

# ser.flush()

# result = ser.readline()
# print(result)

# ser.flush()

# ser.close()             # close port