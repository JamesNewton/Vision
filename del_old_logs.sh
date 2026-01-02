#!/bin/bash

# --- Configuration ---
# Set the directory where your files are located
TARGET_DIR="/mnt/sdcard/captures"
# Set the number of days to keep
DAYS_TO_KEEP=7

# --- Cleanup Logic ---

# 1. Delete .log files older than X days
find "$TARGET_DIR" -type f -name "*.log" -mtime +$DAYS_TO_KEEP -delete

# 2. Delete .jpg files older than X days
find "$TARGET_DIR" -type f -name "*.jpg" -mtime +$DAYS_TO_KEEP -delete

# Optional: Log the cleanup action
#echo "Cleanup performed on $(date): Files older than $DAYS_TO_KEEP days removed." >> /var/log/cleanup_script.log

# --- Automate ---
# Open crontab: crontab -e
# Add this line at the bottom: 0 0 * * * /mnt/sdcard/captures/del_old_logs.sh