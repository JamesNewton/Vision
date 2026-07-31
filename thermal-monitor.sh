#!/bin/bash
# Watches the temperature sensors and shuts down if it overheats. 
# Configuration
LOG_FILE="/mnt/ssd_data/thermal_log.txt"
MAX_TEMP=85000 # 85.0 degrees Celsius

while true; do
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  
  # Read zones
  TEMP_0=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
  TEMP_1=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null || echo 0)
  TEMP_2=$(cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null || echo 0)

  # Log to SSD
  echo "$TIMESTAMP | Z0: $TEMP_0 | Z1: $TEMP_1 | Z2: $TEMP_2" >> "$LOG_FILE"

  # Check against maximums
  if [ "$TEMP_0" -ge "$MAX_TEMP" ] || [ "$TEMP_1" -ge "$MAX_TEMP" ] || [ "$TEMP_2" -ge "$MAX_TEMP" ]; then
    echo "$TIMESTAMP | CRITICAL: Temperature limit exceeded! Initiating shutdown." >> "$LOG_FILE"
    /sbin/shutdown -h now "Thermal critical limit reached."
    exit 0
  fi

  # Wait 60 seconds before next check
  sleep 60
done
