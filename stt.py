import io
import wave
import numpy as np
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="base.en"):
        print(f"⏳ Loading Faster-Whisper ({model_size}) on M1 CPU...")
        # 'int8' or 'float32' with compute_type='int8' runs fastest on Apple Silicon
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: np.ndarray) -> str:
        # Convert float32 array to 16-bit PCM in memory
        int_data = (audio_data * 32767).astype(np.int16)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(int_data.tobytes())
        wav_io.seek(0)

        segments, _ = self.model.transcribe(
            wav_io,
            beam_size=1,
            language="en",
            vad_filter=False # VAD already handled upstream
        )
        
        text = " ".join([seg.text for seg in segments]).strip()
        return text