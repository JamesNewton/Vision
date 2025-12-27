import jetson.inference
import jetson.utils
import cv2
import numpy as np
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
CAM_URL = "http://192.168.0.112/img/video.mjpeg"
LOG_DIR = "/mnt/sdcard/captures"
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

# Sensitivity Settings
MOTION_THRESHOLD = 1000  # How many pixels must change to trigger AI? (Tune this!)
CONFIDENCE_THRESHOLD = 0.7 # AI must be this sure it's an object
COOLDOWN_SECONDS = 2.0    # Don't save same object more than once every 2 seconds

# Objects we care about (Standard COCO classes)
TARGET_CLASSES = {1: 'person', 17: 'cat', 18: 'dog', 21: 'bear'}

# --- SETUP ---
print(f"Connecting to {CAM_URL}...")
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
cap = cv2.VideoCapture(CAM_URL)
display = jetson.utils.videoOutput("display://0") # Remove this line if running headless

# Verify Output Directory
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
# Initialize CSV header if new
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Class,Confidence,Image_Path\n")

# Motion Detection State
avg_frame = None
last_save_time = 0

print("Sentry System Armed. Press Ctrl+C to stop.")

while display.IsStreaming():
    # we don't need to burn up the processor
    time.sleep(0.1)

    # 1. Capture Frame (OpenCV)
    ret, frame = cap.read()
    if not ret:
        print("Video stream lost. Retrying...")
        time.sleep(1)
        cap.open(CAM_URL)
        continue

    # 2. Motion Detection (Lightweight CPU)
    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background on first frame
    if avg_frame is None:
        avg_frame = gray.astype("float")
        continue

    # Accumulate weighted average (Background learns slowly)
    cv2.accumulateWeighted(gray, avg_frame, 0.5)
    
    # Calculate difference
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Count changed pixels
    change_count = np.count_nonzero(thresh)
    #if change_count > 10:
    #    print( change_count )

    # 3. Decision Gate
    motion_detected = change_count > MOTION_THRESHOLD
    
    # Visual Debug: Draw "Motion" text on screen if moving
    if motion_detected:
        #cv2.putText(frame, "MOTION DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 4. AI Detection (Only runs if motion detected)
        # Convert BGR to RGBA for Jetson
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cuda_img = jetson.utils.cudaFromNumpy(frame_rgba)
        
        detections = net.Detect(cuda_img)
        
        current_time = time.time()
        
        # Check if we found something interesting
        found_target = False
        description = ""
        
        for d in detections:
            if d.ClassID in TARGET_CLASSES:
                found_target = True
                class_name = TARGET_CLASSES[d.ClassID]
                description = f"{class_name}"
                
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

        # 5. Save Evidence (With Cooldown)
        if found_target and (current_time - last_save_time > COOLDOWN_SECONDS):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{timestamp}.jpg"
            save_path = os.path.join(LOG_DIR, filename)
            
            # Save the image
            cv2.imwrite(save_path, frame)
            
            # Log to CSV
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.datetime.now()}, {description}, {round(d.Confidence*100)}%, {save_path}\n")
            
            print(f"Alert: {filename} {description} {round(d.Confidence*100)}% ")
            last_save_time = current_time

    # 6. Display Live Feed (Optional)
    # Convert back to CUDA for display output (since display.Render expects CUDA)
    # OR simpler: just imshow with OpenCV since we have the VNC window open anyway?
    # Let's stick to Jetson Utils display for consistency with your previous test
    display_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    display.Render(display_frame)
    display.SetStatus(f"Motion: {change_count} | AI: {net.GetNetworkFPS():.0f} FPS")

cap.release()
