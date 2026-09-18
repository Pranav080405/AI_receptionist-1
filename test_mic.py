import sounddevice as sd
import numpy as np

print("--- AVAILABLE AUDIO INPUT DEVICES ---")
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"[{i}] {d['name']} (default_in: {i == sd.default.device[0]})")

print("\n--- LISTENING FOR 5 SECONDS (Say something aloud!) ---")
duration = 5  # seconds
sample_rate = 16000

recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()

max_amplitude = np.max(np.abs(recording))
rms_energy = np.sqrt(np.mean(recording**2))

print(f"\n📊 Peak Amplitude: {max_amplitude:.4f}")
print(f"📊 RMS Energy:     {rms_energy:.4f}")

if max_amplitude < 0.01:
    print("⚠️ WARNING: Audio is basically silent! Check Mac Settings -> Sound -> Input Volume or Mic permissions.")
else:
    print("✅ Microphone is working and capturing voice!")