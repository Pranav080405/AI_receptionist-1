import sounddevice as sd
import numpy as np
import queue

class AudioIO:
    def __init__(self, input_rate=16000, output_rate=24000, block_size=512):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.block_size = block_size
        self.input_queue = queue.Queue()
        self.input_stream = None
        self.is_playing = False

    def _input_callback(self, indata, frames, time_info, status):
        # Flatten incoming audio to 1D float32
        data = indata.copy().flatten()
        self.input_queue.put(data)

    def start(self):
        # 16kHz Mono 512 blocks = 32ms per callback
        self.input_stream = sd.InputStream(
            samplerate=self.input_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size,
            callback=self._input_callback
        )
        self.input_stream.start()

    def read_mic_chunk(self):
        return self.input_queue.get()

    def play_audio(self, audio_data: np.ndarray):
        if audio_data is None or len(audio_data) == 0:
            return

        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0:
            audio_data = (audio_data / max_val) * 0.9

        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
        self.is_playing = True
        try:
            sd.play(audio_data, samplerate=self.output_rate, blocking=False)
        except Exception as e:
            print(f"Playback error: {e}")

    def stop_playback(self):
        if self.is_playing:
            sd.stop()
            self.is_playing = False

    def close(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        sd.stop()