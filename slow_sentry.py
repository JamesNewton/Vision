import jetson.inference
import jetson.utils
import cv2
import numpy as np
import requests
import time
import datetime
import os

""" 
Setup:
an Aaeon Boxer B with an extra SDcard to provide additional storage.
- 4 GB RAM, 2GB Swap, 16 GB of internal NVRAM
- SDcard

Because there is so little space available on the internal drive, docker needs to use the sdcard
and it needs to automount. Make a perminant mount point:
sudo mkdir -p /mnt/sdcard
 use 
sudo blkid
 to find the UUID of the sdcard
sudo nano /etc/fstab
 and add
UUID=<sdcard UUID>  /mnt/sdcard  ext4  defaults  0  2
 then save that and restart or continue with
sudo mount -a
sudo nano /etc/docker/daemon.json
 add `"data-root": "/mnt/sdcard/docker-data"`
 and restart with
sudo systemctl restart docker

 I also backed up the boxer OS just incase it got fried:
sudo dd if=/dev/mmcblk0 of=/media/your_user/USB_NAME/boxer_backup.img bs=4M status=progress

 The dustynv docker container is the basis for all the installed libraries.
sudo docker pull dustynv/jetson-inference:r32.4.3

To resolve an error with mobilenet not being found, we download it manually 

mkdir -p /mnt/sdcard/networks
cd /mnt/sdcard/networks
wget https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v2.tar.gz
tar -zxvf SSD-Mobilenet-v2.tar.gz

Run with:

sudo docker run --runtime nvidia -it --rm \
    --network host \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/sdcard:/mnt/sdcard \
    dustynv/jetson-inference:r32.4.3

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
cd /mnt/sdcard
python3 smart_sentry.py

 """

# --- CONFIGURATION ---
# Snapshot (vs video.mjpeg) URL for WVC80N webcam.
CAM_URL = "http://192.168.0.112/img/snapshot.cgi" 

TASMOTA_URL = "http://192.168.0.178/cm?cmnd=" 
LOG_DIR = "/mnt/sdcard/captures"
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

# Sensitivity Settings
MOTION_THRESHOLD = 1000  # How many pixels must change to trigger AI? (Tune this!)
CONFIDENCE_THRESHOLD = 0.7 # AI must be this sure it's an object
COOLDOWN_SECONDS = 2.0    # Don't save same object more than
DOORBELL_SECONDS = 5.0    # Don't ring the bell constantly
BACKGROUND_ALPHA = 0.1  # Changed from 0.5 to 0.1 for better stability

# POLL SPEED
# How long to SLEEP between checks.
# 0.2 = 5 FPS (Very responsive, low CPU)
# 0.5 = 2 FPS (Lowest CPU, acceptable security)
POLL_INTERVAL = 0.5 

TARGET_CLASSES = {1: 'person', 17: 'cat', 18: 'dog', 21: 'bear'}

def tasmota_cmd(cmd):
    try:
        # don't wait for a reply. It gets it or it doesn't
        requests.get(f"{TASMOTA_URL}{cmd}", timeout=(None, 0.01))
    except:
        pass

def get_snapshot():
    """
    Grabs a single frame from the camera.
    Returns: cv2 image (numpy array) or None
    """
    try:
        # 1. Fetch the raw JPEG data; much lighter than opening a video stream
        r = requests.get(CAM_URL, timeout=1.0)
        if r.status_code == 200:
            # 2. Decode JPEG to OpenCV format
            img_array = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(img_array, -1)
            return frame
    except:
        pass
    return None

# --- SETUP ---
print(f"Starting")
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
display = jetson.utils.videoOutput("display://0") 

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Class,Confidence,Image_Path\n")

# Motion Detection State
avg_frame = None
last_save_time = 0
last_bell_time = 0

tasmota_cmd("Power1%20Blink")

print("Sentry System Armed (Low Power Mode). Press Ctrl+C to stop.")

while True: # Main polling loop
    
    # 1. SLEEP FIRST (This is where the power saving happens)
    time.sleep(POLL_INTERVAL)

    # 2. GET SINGLE FRAME
    frame = get_snapshot()
    
    # If snapshot failed (or camera offline), skip this cycle
    if frame is None:
        continue

    # 3. MOTION DETECTION
    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background on first frame
    if avg_frame is None:
        avg_frame = gray.astype("float")
        continue

    # Update background (slower alpha prevents ghosts)
    cv2.accumulateWeighted(gray, avg_frame, BACKGROUND_ALPHA)

    # Calculate difference
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Count changed pixels
    change_count = np.count_nonzero(thresh)
    motion_detected = change_count > MOTION_THRESHOLD
    
    # 4. AI LOGIC (Only runs on motion)
    if motion_detected:
        # 4. AI Detection (Only runs if motion detected)
        # Convert BGR to RGBA for Jetson
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cuda_img = jetson.utils.cudaFromNumpy(frame_rgba)
        detections = net.Detect(cuda_img)
        #tasmota_cmd("Power2%20Blink")
        current_time = time.time()
        
        # Capture the most certain object that triggered the alert
        primary_target = None 
        saw_person = False
        
        for d in detections:
            if d.ClassID in TARGET_CLASSES:
                class_name = TARGET_CLASSES[d.ClassID]
                conf_percent = round(d.Confidence * 100)
                description = f"{class_name}"
                # If this is the first target found this frame, save it as the "Primary" for logging
                if primary_target is None:
                    primary_target = (class_name, conf_percent)
                
                if (d.ClassID == 1): 
                    saw_person = True 
                
                # Draw box on the OpenCV frame (for saving)
                # Note: d.Left, d.Top, etc are floats, cast to int
                col_conf = round(255 * d.Confidence)
                cv2.rectangle(frame, 
                    (int(d.Left), int(d.Top)), 
                    (int(d.Right), int(d.Bottom)), 
                    (0, col_conf, 0), 
                    2
                    )
                cv2.putText(frame, description, 
                    (int(d.Left), int(d.Top)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, col_conf, 0), 
                    2, 
                    cv2.LINE_AA
                    )
        # 5. ACTIONS
        if primary_target and (current_time - last_save_time > COOLDOWN_SECONDS):
            t_class, t_conf = primary_target # Unpack the data
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{timestamp}.jpg"
            save_path = os.path.join(LOG_DIR, filename)
            
            cv2.imwrite(save_path, frame)
            
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.datetime.now()}, {t_class}, {t_conf}%, {save_path}\n")

            print(f"Alert: {filename} {t_class} {t_conf}% ")
            last_save_time = current_time

        if (saw_person and current_time - last_bell_time > DOORBELL_SECONDS):
            tasmota_cmd("Power1%20Blink") #ring the bell
            last_bell_time = current_time

    # 6. Display (Optional - Remove if running headless to save more power)
    if display.IsStreaming():
        display_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        display.Render(display_frame)
        display.SetStatus(f"Motion: {change_count}")
