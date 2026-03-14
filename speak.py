import subprocess
import os

class BoxerTTS:
    def __init__(self):
        # Paths relative to your SD card setup
        self.base_path = "/mnt/sdcard/piper-tts"
        self.piper_bin = os.path.join(self.base_path, "piper/piper")
        self.voices_path = os.path.join(self.base_path, "voices")
        
        # Default voice settings (Adjust these as needed)
        self.voice_model = os.path.join(self.voices_path, "en_US-lessac-low.onnx")
        self.sample_rate = "16000" # Use 16000 for 'low', 22050 for 'medium'

    def say(self, text):
        # 1. Start the Piper process
        piper_proc = subprocess.Popen(
            [self.piper_bin, "--model", self.voice_model, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        # 2. Start the aplay process to play the output of Piper
        aplay_proc = subprocess.Popen(
            ["aplay", "-r", self.sample_rate, "-f", "S16_LE", "-t", "raw"],
            stdin=piper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Send the text to Piper
        piper_proc.stdin.write(text.encode('utf-8'))
        piper_proc.stdin.close()
        
        # Wait for speech to finish
        aplay_proc.wait()

if __name__ == "__main__":
    tts = BoxerTTS()
    tts.say("Piper t.t.s. is ready.")

