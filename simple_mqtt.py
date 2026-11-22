import socket

def publish_mqtt_raw(topic, message="", broker_ip="127.0.0.1", port=1883):
    client_id = b"AaeonNode"
    topic_b = topic.encode('utf-8')
    msg_b = message.encode('utf-8')
    
    rem_len_conn = 12 + len(client_id)
    connect_packet = bytearray([0x10, rem_len_conn, 0x00, 0x04]) + b"MQTT" + bytearray([0x04, 0x02, 0x00, 0x3C, 0x00, len(client_id)]) + client_id
    
    rem_len_pub = 2 + len(topic_b) + len(msg_b)
    publish_packet = bytearray([0x30, rem_len_pub, 0x00, len(topic_b)]) + topic_b + msg_b
    
    disconnect_packet = bytearray([0xE0, 0x00])

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0) # Prevent the script from hanging if the broker is down
        s.connect((broker_ip, port))
        s.send(connect_packet)
        s.send(publish_packet)
        s.send(disconnect_packet)
        s.close()
    except Exception as e:
        print(f"MQTT Publish Error: {e}")
