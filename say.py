#!/usr/bin/env python3
import subprocess
import os
import sys

class BoxerTTS:
    def __init__(self):
        self.base_path = "/mnt/sdcard/piper-tts"
        self.piper_bin = os.path.join(self.base_path, "piper/piper")
        self.voice_model = os.path.join(self.base_path, "voices/en_US-lessac-low.onnx")
        self.sample_rate = "16000"

    def say(self, text):
        if not text: return
        
        piper_proc = subprocess.Popen(
            [self.piper_bin, "--model", self.voice_model, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        aplay_proc = subprocess.Popen(
            ["aplay", "-r", self.sample_rate, "-f", "S16_LE", "-t", "raw"],
            stdin=piper_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        piper_proc.stdin.write(text.encode('utf-8'))
        piper_proc.stdin.close()
        aplay_proc.wait()

if __name__ == "__main__":
    # If called directly, it reads from STDIN (piping support)
    tts = BoxerTTS()
    input_text = sys.stdin.read().strip()
    if input_text:
        tts.say(input_text)

