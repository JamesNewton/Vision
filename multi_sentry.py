import jetson.inference
import jetson.utils
import cv2
import numpy as np
import requests
'''
requires:
pip install requests
after docker startup, or build a new image from the original dustynv
'''
import time
import datetime
import os
import threading

# --- CONFIGURATION ---
TASMOTA_URL = "http://192.168.0.178/cm?cmnd=" 
LOG_DIR = "/mnt/sdcard/captures"
ALERT_FILE = os.path.join(LOG_DIR, "detection.csv")

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
    "motion_threshold": 800,
    "alpha": 0.2,

    # Use ['motion'] for simple motion detection (No AI).
    # Use ['person', 'dog'] to run AI and only alert on those objects.
    "alarm_class": ['person'],
    # Active from (24h format HH;mm)
    "start_time": "00:00",
    "end_time": "23:59",

    # ROI: Region of Interest (x, y, width, height)
    # The AI will ONLY process this box. 
    "roi": None # Full frame
}

# 2. Tapo C120 (RTSP Stream 2 Mode)
CAM2_CONFIG = {
    "name": "Tapo",
    "type": "rtsp",
    "url": "rtsp://192.168.0.102:554/stream2",
    "motion_threshold": 1200, 
    "alpha": 0.3,
    
    "alarm_class": ['motion'], # Just look for movement (e.g. at night)
    
    "start_time": "17:00", # Only active at night
    "end_time": "08:00",   # Crosses midnight correctly
    # Region of Interest (x, y, width, height)
    # Example: (100, 100, 300, 200) to ignore trees at edges.
    # Set to None for full frame.
    "roi": (110, 120, 500, 220)  
}

CAMERAS = [CAM1_CONFIG, CAM2_CONFIG]

# --- HELPER SUBROUTINES ---

def tasmota_cmd(cmd):
    try:
        requests.get(f"{TASMOTA_URL}{cmd}", timeout=(None, 0.05))
    except:
        pass

def is_time_active(start_str, end_str):
    """
    Checks if current time is within range (handles midnight crossing).
    """
    if not start_str or not end_str: return True
    
    now = datetime.datetime.now().time()
    start = datetime.datetime.strptime(start_str, "%H:%M").time()
    end = datetime.datetime.strptime(end_str, "%H:%M").time()
    
    if start <= end:
        return start <= now <= end
    else: # Crosses midnight (e.g. 22:00 to 06:00)
        return now >= start or now <= end

def check_motion_level(frame, avg_frame, alpha):
    """
    Subroutine to check an area for motion.
    Returns: pixel_count, updated_avg_frame
    """
    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background on first frame
    if avg_frame is None:
        avg_frame = gray.astype("float")
        return 0, avg_frame, None

    # Update background
    cv2.accumulateWeighted(gray, avg_frame, alpha)
    
    # Calculate difference
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    return np.count_nonzero(thresh), avg_frame, thresh

def enhance_night_image(frame, avg_brightness):
    """
    Inverse-Dehazing for Night Vision.
    Boosts dark regions while preserving bright lights.
    Auto-disables if the scene is already bright.
    """
    # (Threshold 80/255 is roughly twilight/office lighting)
    if avg_brightness > 80:
        return frame
    # Math: t = 1 - omega * min(inverted_image)
    # Simplified: t = 1 - omega * (1 - local_max_brightness)
    kernel = np.ones((5, 5), np.uint8) # kernel size for local max brightness
    local_max = cv2.dilate(frame, kernel) #each kernal becomes the brightest part
    local_max = cv2.GaussianBlur(local_max, (21, 21), 0) #blur those changes
    # Calculate 't' (0.0 to 1.0) Lower 't' = it's darker = More boost needed.
    # base value sets max boost, and prevents dividing by zero
    t = 0.25 + 0.75 * (local_max.astype(float) / 255.0)
    # Because 't' is a 3-channel matrix, this boosts colors individually
    enhanced = frame.astype(float) / t
    # If brightness is 0 (Pitch Black), blend = 1.0 (Full Boost)
    # If brightness is 80 (Twilight), blend = 0.0 (No Boost)
    blend_factor = max(0, (80 - avg_brightness) / 80.0)
    final = frame.astype(float) * (1 - blend_factor) + enhanced * blend_factor
    # Clip back to valid image range
    return np.clip(final, 0, 255).astype(np.uint8)

# --- CAMERA CLASSES ---

class HttpCamera:
    """Handles cameras that support single-frame HTTP GET (Linksys)"""
    def __init__(self, config):
        self.url = config['url']
        self.name = config['name']
        self.avg_frame = None
        self.last_valid_frame = None # For display persistence

    def set_active(self, state):
        pass # HTTP camera doesn't need a background thread sleep

    def read(self):
        try:
            r = requests.get(self.url, timeout=0.5)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                frame = cv2.imdecode(arr, -1)
                self.last_valid_frame = frame
                return frame
        except:
            pass
        return None
    
    def stop(self):
        pass # Nothing to stop

class RtspCamera:
    """
    Handles cameras that require persistent RTSP connection (Tapo).
    Optimized to sleep deeply and throttle frozen-frame checks.
    """
    def __init__(self, config):
        self.url = config['url']
        self.name = config['name']
        self.avg_frame = None
        self.frame = None
        self.last_valid_frame = None
        
        # State Management
        self.active = True  # Controls deep sleep
        self.stopped = False
        self.stream = None
        self.lock = threading.Lock()
        
        # Start background drainer immediately
        threading.Thread(target=self.update, daemon=True).start()

    def set_active(self, state):
        """Enable or disable the stream to save CPU"""
        self.active = state

    def start_stream(self):
        """Internal: Safely opens the capture"""
        print(f"[{self.name}] Connecting RTSP...")
        if self.stream:
            self.stream.release()
        self.stream = cv2.VideoCapture(self.url)
        # Optional: reduce buffer size to ensure freshness
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def update(self):
        self.start_stream()
        # Settings to detect frozen cameras (frames all the same)
        last_raw_slice = None
        stale_count = 0
        STALE_LIMIT = 5 # Restart after 5 failed *checks* (approx 5 seconds)
        frame_counter = 0

        while not self.stopped:
            # DEEP SLEEP: If inactive, release everything and wait
            if not self.active:
                if self.stream and self.stream.isOpened():
                    self.stream.release()
                time.sleep(1.0) # Sleep long to save CPU
                continue

            # RECONNECT: If we woke up but have no stream, connect
            if self.stream is None or not self.stream.isOpened():
                self.start_stream()

            grabbed, frame = self.stream.read()
            
            if grabbed and frame is not None:
                frame_counter += 1
                
                # --- THROTTLED FROZEN CHECK ---
                # Only check once every 30 frames (~1 second) to save CPU
                if frame_counter % 30 == 0:
                    h, w = frame.shape[:2]
                    c_y, c_x = h // 2, w // 2
                    curr_slice = frame[max(0,c_y-64):c_y+64, max(0,c_x-64):c_x+64]
                    
                    # Brightness check (don't check if pitch black)
                    if np.mean(curr_slice) > 5 and last_raw_slice is not None:
                        diff = cv2.absdiff(curr_slice, last_raw_slice)
                        if np.count_nonzero(diff) == 0:
                            stale_count += 1
                        else:
                            # Reset counter if it's too dark to tell, or first frame
                            stale_count = 0 
                    
                    last_raw_slice = curr_slice.copy() # Save for next cycle

                    if stale_count > STALE_LIMIT:
                        print(f"[{self.name}] FROZEN DETECTED! Restarting...")
                        with self.lock: self.frame = None 
                        self.start_stream()
                        stale_count = 0
                        last_raw_slice = None
                
                # Update the valid frame
                with self.lock:
                    self.frame = frame
                    self.last_valid_frame = frame
                
                # Tiny sleep to yield CPU, but don't overflow buffer
                time.sleep(0.01) 

            else:
                # Standard 'Connection Lost' logic
                print(f"[{self.name}] Signal lost. Retrying...")
                with self.lock:
                    self.frame = None
                time.sleep(2)
                self.start_stream()

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        if self.stream: self.stream.release()
        
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
with open(ALERT_FILE, "w") as f:
    f.write(f"Starting\n")

last_save_time = 0
last_bell_time = 0

tasmota_cmd("Power1%20Blink")
print(f"Sentry Armed with {len(active_cams)} cameras. Press Ctrl+C to stop.")

try:
    while True:
        # If we are rendering video, we want to update faster than the poll interval
        # so the persistent RTSP feed looks smooth, even if we only run AI logic 5 times/sec.
        # But to keep it simple and low power, we will stick to the poll interval for now.
        time.sleep(POLL_INTERVAL)
        
        display_images = []
        print(".                    ", end="\r", flush=True)

        for config, cam_obj in active_cams:
            # CHECK SCHEDULE
            is_active = is_time_active(config.get('start_time'), config.get('end_time'))
            
            # TELL CAMERA TO SLEEP/WAKE
            # This shuts down the background RTSP thread if inactive
            cam_obj.set_active(is_active)

            if not is_active:
                # Add dummy frame for display if needed
                if display.IsStreaming():
                    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, "SLEEPING", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)
                    display_images.append(dummy)
                continue

            frame = cam_obj.read()
            
            # --- DISPLAY PREP ---
            # If camera is offline, use the last known frame or a black box
            if frame is None:
                if cam_obj.last_valid_frame is not None:
                    frame = cam_obj.last_valid_frame
                    cv2.putText(frame, "OLD", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    # Create 640x360 black placeholder
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "OFFLINE", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            # Save for the "Stitcher" at the end
            display_images.append(frame)

            # --- MOTION LOGIC (Skip if frame is the dummy black box) ---
            if cam_obj.last_valid_frame is None: continue 
            
            # (Re-read fresh frame for processing logic to ensure we don't process stale data)
            proc_frame = cam_obj.read()
            if proc_frame is None: continue

            # If ROI is set ONLY look for motion / objects in that box. 
            # This saves CPU and ignores wind/trees outside the box.
            roi_x, roi_y = 0, 0 # Offsets for drawing later
            motion_input = proc_frame
            
            if config.get('roi'):
                roi_x, roi_y, roi_w, roi_h = config['roi']
                # Safety check for image bounds
                h, w = proc_frame.shape[:2]
                if roi_x+roi_w <= w and roi_y+roi_h <= h:
                    motion_input = proc_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            # Note: avg_frame will automatically size itself to the ROI
            non_zero_thresh, cam_obj.avg_frame, motion_mask = check_motion_level(
                motion_input, cam_obj.avg_frame, config['alpha']
            )
            alarm_classes = config.get("alarm_class", [])
            if motion_mask is not None and 'motion' in alarm_classes: # add the mask back into the input in color
                tint_color = (30, 30, 60, 0)
                cv2.add(motion_input, tint_color, dst=motion_input, mask=motion_mask)
            print(non_zero_thresh, end=" ")
            
            # Draw ROI box on the main frame so we can see where we are monitoring
            if config.get('roi'):
                cv2.rectangle(proc_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 1)

            if non_zero_thresh > config['motion_threshold']:
                print("M", end=" ")
                cv2.putText(proc_frame, "M", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                display_images[-1] = proc_frame 
                
                primary_target = None

                # CASE A: 'motion' is in the allowed classes -> Trigger immediately
                if 'motion' in alarm_classes:
                    primary_target = ('motion', 100)
                    # Draw a box around the ROI to show what triggered it
                    if config.get('roi'):
                        cv2.rectangle(proc_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 0, 255), 2)

                # CASE B: Object Detection Required
                else:
                    # RUN AI on the CROPPED ROI (motion_input)
                    # This is much faster/more accurate than resizing the full frame
                    frame_rgba = cv2.cvtColor(motion_input, cv2.COLOR_BGR2RGBA)
                    cuda_img = jetson.utils.cudaFromNumpy(frame_rgba)
                    detections = net.Detect(cuda_img)
                    
                    for d in detections:
                        if d.ClassID in TARGET_CLASSES:
                            class_name = TARGET_CLASSES[d.ClassID]
                            
                            # Only accept if this class is in our list
                            if class_name in alarm_classes:
                                conf_percent = round(d.Confidence * 100)
                                
                                if primary_target is None:
                                    primary_target = (class_name, conf_percent)
                                
                                # Priority override
                                if class_name == 'person' and primary_target[0] != 'person':
                                    primary_target = (class_name, conf_percent)

                                # Draw Boxes
                                # IMPORTANT: We must add the ROI offset (roi_x, roi_y) 
                                # because the AI coordinates are relative to the crop
                                left = int(d.Left) + roi_x
                                top = int(d.Top) + roi_y
                                right = int(d.Right) + roi_x
                                bottom = int(d.Bottom) + roi_y
                                
                                col_conf = round(255 * d.Confidence)
                                cv2.rectangle(proc_frame, (left, top), (right, bottom), (0, col_conf, 0), 2)
                    
                    # Update the display image list with this annotated frame
                    # (We replace the raw frame we added earlier)
                    display_images[-1] = proc_frame

                # --- TRIGGER ACTIONS ---
                current_time = time.time()
                if primary_target and (current_time - last_save_time > COOLDOWN_SECONDS):
                    t_class, t_conf = primary_target
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{config['name']}_{timestamp}.jpg"
                    save_path = os.path.join(LOG_DIR, filename)
                    avg_brightness = np.mean(cam_obj.avg_frame)
                    final_image = enhance_night_image(proc_frame, avg_brightness)
                    # Save the FULL frame (context), not just the crop
                    cv2.imwrite(save_path, final_image)

                    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    log_path = os.path.join(LOG_DIR, f"{date_str}_detection_log.csv")
                    if not os.path.exists(log_path):
                        with open(log_path, "w") as f:
                            f.write("Timestamp,Camera,Class,Confidence,Image_Path\n")
                    with open(log_path, "a") as f:
                        f.write(f"{datetime.datetime.now()}, {config['name']}, {t_class}, {t_conf}%, {save_path}\n")

                    with open(ALERT_FILE, "w") as f:
                        f.write(f"{datetime.datetime.now()}, {config['name']}, {t_class}, {t_conf}%, {filename}\n")

                    print(f"Alert [{config['name']}]: {t_class} {t_conf}%")
                    
                    if (current_time - last_bell_time > DOORBELL_SECONDS):
                        # Simple rule: Persons ring bell, everything else blinks light
                        if t_class == 'person':
                            tasmota_cmd("Power1%20Blink") 
                        else:
                            tasmota_cmd("Power2%20Blink")
                        last_bell_time = current_time
                    
                    last_save_time = current_time
                    #end recog
                #end motion detected
            print(", ", end="")
            #end camera

        # --- DISPLAY COMPOSITOR ---
        if display.IsStreaming() and len(display_images) >= 2:
            # Resize all to a standard height (e.g. 360p) so they stack nicely
            target_h = 400
            resized_imgs = []
            
            for img in display_images:
                h, w = img.shape[:2]
                scale = target_h / h
                new_w = int(w * scale)
                resized = cv2.resize(img, (new_w, target_h))
                
                resized_imgs.append(resized)

            # hstack requires same height, which we just ensured
            combined_img = np.hstack(resized_imgs)
            
            frame_rgba = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGBA)
            cuda_display = jetson.utils.cudaFromNumpy(frame_rgba)
            display.Render(cuda_display)
            display.SetStatus(f"Sentry Mode | Cameras: {len(active_cams)}")

except KeyboardInterrupt:
    for _, c in active_cams:
        c.stop()