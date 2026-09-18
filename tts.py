import numpy as np
from kokoro_onnx import Kokoro

# Patch numpy to load pickled voice vectors
_orig_load = np.load
def _patched_load(*args, **kwargs):
    kwargs['allow_pickle'] = True
    return _orig_load(*args, **kwargs)
np.load = _patched_load

class Synthesizer:
    def __init__(self):
        print("⏳ Loading Kokoro-82M ONNX engine...")
        self.kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

    def speak(self, text: str, voice: str = "af_bella") -> np.ndarray:
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Generates 24kHz float32 audio
        samples, sample_rate = self.kokoro.create(
            text=text.strip(),
            voice=voice,
            speed=1.0,
            lang="en-us"
        )
        return samples