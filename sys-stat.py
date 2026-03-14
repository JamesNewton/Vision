#!/usr/bin/env python3
import psutil #sudo apt install python3-psutil
import shutil
import argparse
import sys

def get_stats(for_speech=False):
    # RAM
    ram = psutil.virtual_memory()
    ram_mb = ram.available // (1024 * 1024)
    ram_pct = 100 - ram.percent

    # Disk - OS
    os_disk = shutil.disk_usage("/")
    os_gb = os_disk.free // (1024**3)
    os_pct = int((os_disk.free / os_disk.total) * 100)

    # Disk - SD Card
    sd_disk = shutil.disk_usage("/mnt/sdcard")
    sd_gb = sd_disk.free // (1024**3)
    sd_pct = int((sd_disk.free / sd_disk.total) * 100)
    
    # CPU & Temp
    cpu_load = int(psutil.cpu_percent(interval=0.5))
    temps = psutil.sensors_temperatures()
    core_temp_c = 0
    for zone in ['thermal-fan-est', 'CPU-therm', 'cpu_thermal']:
        if zone in temps:
            core_temp_c = temps[zone][0].current
            break
    
    # Convert C to F
    core_temp_f = int((core_temp_c * 9/5) + 32)

    if for_speech:
        # Optimized for TTS pronunciation
        return (
            f"System Status. Available ram is {ram_mb} megabytes, which is {ram_pct} percent free. "
            f"O.S. drive is {os_pct} percent free with {os_gb} gigabytes remaining. "
            f"S.D. card is {sd_pct} percent free with {sd_gb} gigabytes remaining. "
            f"Average cpu load is {cpu_load} percent. "
            f"Core temperature is {core_temp_f} degrees Fahrenheit."
        )
    else:
        # Optimized for Console reading
        return (
            f"SYS STATS: RAM: {ram_pct}% ({ram_mb}MB) | "
            f"OS: {os_pct}% ({os_gb}GB) | SD: {sd_pct}% ({sd_gb}GB) | "
            f"CPU: {cpu_load}% | TEMP: {core_temp_f}°F"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--say", action="store_true", help="Speak the output")
    args = parser.parse_args()

    if args.say:
        stats_text = get_stats(for_speech=True)
        try:
            from speak import BoxerTTS
            tts = BoxerTTS()
            tts.say(stats_text)
        except ImportError:
            print("Error: speak.py not found.", file=sys.stderr)
            print(get_stats(for_speech=False))
    else:
        print(get_stats(for_speech=False))

