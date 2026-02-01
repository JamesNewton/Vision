import cv2
import numpy as np
import time
import json
import urllib.request
import sys
from datetime import datetime, timedelta

# Check the door on the chicken coup is closed by comapring it's darkness with the wall next to it.
# If the door is open, the darkness inside will show us. 
# If it's close, the door should be painted brighter than the wall.

# --- CONFIGURATION ---
LAT = "33.1247"   # Replace with your Lat
LNG = "-117.0809" # Replace with your Long
# UTC Offset for Pacific Standard Time (PST is -8, PDT is -7)
UTC_OFFSET_HOURS = -8
RTSP_URL = "rtsp://user:pass@192.168.0.102:554/stream1"
CHECK_DELAY_MINUTES = 30

# [y1:y2, x1:x2] - Adjust these based on your debug_frame.jpg
DOOR_ZONE = (600, 700, 1340, 1440) 
WALL_ZONE = (600, 700, 1225, 1325)

# Email Settings
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
TO_EMAIL = "alert_me@example.com"

def get_sunset():
    """Fetches sunset for the current local date to avoid UTC day-rollover issues."""
    try:
        # Get your local date (e.g., 2026-01-31)
        local_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LNG}&date={local_date}&formatted=0"
        print(url)
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            sunset_str = data['results']['sunset']
            
            # Extract the time portion before the '+' or 'Z'
            clean_date = sunset_str.split('+')[0].replace('Z', '')
            return datetime.strptime(clean_date, "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        print(f"Error fetching sunset: {e}")
        return None

def analyze_door(is_test=False):
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not access camera stream.")
        return

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness
    door_crop = gray[DOOR_ZONE[0]:DOOR_ZONE[1], DOOR_ZONE[2]:DOOR_ZONE[3]]
    wall_crop = gray[WALL_ZONE[0]:WALL_ZONE[1], WALL_ZONE[2]:WALL_ZONE[3]]
    door_br = np.mean(door_crop)
    wall_br = np.mean(wall_crop)

    # --- VISUAL DEBUGGING ---
    # Draw boxes on the original color frame (y1, x1, y2, x2)
    # Door = Red, Wall = Green
    cv2.rectangle(frame, (DOOR_ZONE[2], DOOR_ZONE[0]), (DOOR_ZONE[3], DOOR_ZONE[1]), (0, 0, 255), 3)
    cv2.rectangle(frame, (WALL_ZONE[2], WALL_ZONE[0]), (WALL_ZONE[3], WALL_ZONE[1]), (0, 255, 0), 3)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"check_{timestamp}.jpg"
    cv2.imwrite("captures/"+filename, frame)
    print(f"Debug image saved: {filename}")

    log_msg = f"RESULT: Door={door_br:.2f}, Wall={wall_br:.2f}"
    print(log_msg)

    if door_br < wall_br:
        alert_text = f"? ALERT: Door is DARKER than wall. Likely OPEN. ({log_msg})"
        if is_test:
            print(f"[TEST MODE] Would send email: {alert_text}")
        else:
            send_email("Chicken Door Alert!", alert_text)
    else:
        print("SECURE: Door is lighter than wall. Closed.")

def send_email(subject, body):
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = TO_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    is_test = "--test" in sys.argv
    
    sunset_utc = get_sunset()
    if not sunset_utc:
        print("Critical Error: Could not determine sunset time.")
        sys.exit(1)

    # Convert UTC sunset to Local Time
    local_offset = timedelta(hours=UTC_OFFSET_HOURS)
    sunset_local = sunset_utc + local_offset
    
    # Calculate target check time (Local)
    check_time_local = sunset_local + timedelta(minutes=CHECK_DELAY_MINUTES)
    
    print(f"--- SUNSET INFO (LOCAL) ---")
    print(f"Sunset:           {sunset_local.strftime('%H:%M:%S')}")
    print(f"Target Check:     {check_time_local.strftime('%H:%M:%S')}")
    print(f"Current Time:     {datetime.now().strftime('%H:%M:%S')}")
    print(f"---------------------------")

    if is_test:
        print("Test mode active. Running analysis immediately...")
        analyze_door(is_test=True)
    else:
        print("Waiting for scheduled check time...")
        while True:
            # Now comparing Local Time to Local Time
            if datetime.now() >= check_time_local:
                print("Time reached. Checking camera...")
                analyze_door()
                break
            time.sleep(30)
