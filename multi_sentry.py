import jetson.inference
import jetson.utils
import cv2
import numpy as np
import requests
import time
import datetime
import os
import threading

# --- CONFIGURATION ---
TASMOTA_URL = "http://192.168.0.178/cm?cmnd=" 
LOG_DIR = "/mnt/sdcard/captures"
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

# SHARED SETTINGS
CONFIDENCE_THRESHOLD = 0.7 
COOLDOWN_SECONDS = 2.0 
DOORBELL_SECONDS = 5.0
POLL_INTERVAL = 0.2  # 5 Hz check rate

TARGET_CLASSES = {1: 'person', 17: 'cat', 18: 'dog', 21: 'bear'}

# --- CAMERA DEFINITIONS ---
# 1. Linksys (HTTP Snapshot Mode)
CAM1_CONFIG = {
    "name": "Linksys",
    "type": "http",
    "url": "http://192.168.0.112/img/snapshot.cgi",
    "motion_threshold": 1000,
    "alpha": 0.2
}

# 2. Tapo C120 (RTSP Stream 2 Mode)
# Note the "/stream2" at the end for Low CPU usage!
CAM2_CONFIG = {
    "name": "Tapo",
    "type": "rtsp",
    "url": "rtsp://192.168.0.102:554/stream2",
    "motion_threshold": 500, # Lower threshold for lower resolution stream
    "alpha": 0.4
}

CAMERAS = [CAM1_CONFIG, CAM2_CONFIG]

# --- CLASSES ---

class HttpCamera:
    """Handles cameras that support single-frame HTTP GET (Linksys)"""
    def __init__(self, config):
        self.url = config['url']
        self.name = config['name']
        self.avg_frame = None

    def read(self):
        try:
            r = requests.get(self.url, timeout=0.5)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                return cv2.imdecode(arr, -1)
        except:
            pass
        return None
    
    def stop(self):
        pass # Nothing to stop

class RtspCamera:
    """Handles cameras that require persistent RTSP connection (Tapo)"""
    def __init__(self, config):
        self.url = config['url']
        self.name = config['name']
        self.avg_frame = None
        self.frame = None
        self.stopped = False
        self.stream = cv2.VideoCapture(self.url)
        # Start background drainer immediately
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame = frame
            else:
                # Auto-reconnect logic could go here, but keeping it simple
                time.sleep(1) 

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def tasmota_cmd(cmd):
    try:
        requests.get(f"{TASMOTA_URL}{cmd}", timeout=(0.05, 0.05))
    except:
        pass

# --- SETUP ---
print("Initializing AI...")
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
display = jetson.utils.videoOutput("display://0") 

# Initialize Camera Objects based on config
active_cams = []
for c in CAMERAS:
    print(f"Connecting to {c['name']}...")
    if c['type'] == 'http':
        active_cams.append((c, HttpCamera(c)))
    elif c['type'] == 'rtsp':
        active_cams.append((c, RtspCamera(c)))

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Camera,Class,Confidence,Image_Path\n")

last_save_time = 0
last_bell_time = 0

tasmota_cmd("Power1%20Blink")
print(f"Sentry Armed with {len(active_cams)} cameras. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(POLL_INTERVAL)
        
        # Loop through all cameras
        for config, cam_obj in active_cams:
            
            frame = cam_obj.read()
            if frame is None: continue

            # --- MOTION DETECTION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if cam_obj.avg_frame is None:
                cam_obj.avg_frame = gray.astype("float")
                continue

            cv2.accumulateWeighted(gray, cam_obj.avg_frame, config['alpha'])
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(cam_obj.avg_frame))
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            if np.count_nonzero(thresh) > config['motion_threshold']:
                
                # --- AI DETECTION ---
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                cuda_img = jetson.utils.cudaFromNumpy(frame_rgba)
                detections = net.Detect(cuda_img)
                
                current_time = time.time()
                primary_target = None 
                
                for d in detections:
                    if d.ClassID in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[d.ClassID]
                        conf_percent = round(d.Confidence * 100)
                        
                        if primary_target is None:
                            primary_target = (class_name, conf_percent)
                        
                        # Priority Override (Person > Dog)
                        if class_name == 'person' and primary_target[0] != 'person':
                            primary_target = (class_name, conf_percent)

                        # Draw (Optional - heavy on CPU if VNC is off, maybe skip?)
                        # cv2.rectangle(...) 

                # --- ACTIONS ---
                if primary_target and (current_time - last_save_time > COOLDOWN_SECONDS):
                    t_class, t_conf = primary_target
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{config['name']}_{timestamp}.jpg"
                    save_path = os.path.join(LOG_DIR, filename)
                    
                    cv2.imwrite(save_path, frame)
                    
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{datetime.datetime.now()}, {config['name']}, {t_class}, {t_conf}%, {save_path}\n")

                    print(f"Alert [{config['name']}]: {t_class} {t_conf}%")
                    
                    if (current_time - last_bell_time > DOORBELL_SECONDS):
                        if t_class == 'person':
                            tasmota_cmd("Power1%20Blink") 
                        else:
                            tasmota_cmd("Power2%20Blink")
                        last_bell_time = current_time
                    
                    last_save_time = current_time

except KeyboardInterrupt:
    for _, c in active_cams:
        c.stop()