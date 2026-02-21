import serial
import time
import os
from datetime import datetime

# Setup Serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Setup Directory
SAVE_DIR = "captures"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_gmc_data():
    ser.reset_input_buffer()
    
    # 1. Get CPM (4 bytes)
    ser.write(b'<GETCPM>>')
    cpm_data = ser.read(4)
    cpm = int.from_bytes(cpm_data, byteorder='big') if len(cpm_data) == 4 else None

    return cpm

print(f"Logging started (Fahrenheit mode). Saving to {SAVE_DIR}/")

try:
    while True:
        cpm = get_gmc_data()
        
        if cpm is not None:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            datestamp = now.strftime("%Y%m%d")
            filename = os.path.join(SAVE_DIR, f"radmon_{datestamp}.csv")
            
            file_exists = os.path.isfile(filename)
            
            with open(filename, "a") as f:
                if not file_exists:
                    f.write("Timestamp,CPM\n")
                f.write(f"{timestamp},{cpm}\n")
            
            print(f"[{timestamp}] CPM: {cpm}")
            
        time.sleep(5) 
except KeyboardInterrupt:
    ser.close()
    print("\nLogging stopped.")

