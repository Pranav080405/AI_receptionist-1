import os
import onnxruntime as ort
import numpy as np

class SileroVAD:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.model_path = "models/silero_vad.onnx"
        
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(self.model_path, opts, providers=['CPUExecutionProvider'])
        self.reset_states()

    def reset_states(self):
        # Silero VAD v5 recurrent state shape is [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> bool:
        # Ensure 1D float32 array
        chunk = np.ascontiguousarray(chunk.flatten(), dtype=np.float32)
        
        # Guard against volume clipping (Silero requires [-1.0, 1.0])
        max_val = np.max(np.abs(chunk))
        if max_val > 1.0:
            chunk = chunk / max_val

        # Exact 512 samples required at 16kHz
        if len(chunk) != 512:
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            else:
                chunk = chunk[:512]

        # Input shape: [1, 512]
        tensor_chunk = np.expand_dims(chunk, axis=0)
        sr = np.array(16000, dtype=np.int64)

        try:
            inputs = {
                'input': tensor_chunk,
                'sr': sr,
                'state': self._state
            }
            out, self._state = self.session.run(None, inputs)
            prob = float(out[0][0])
            vad_detected = prob >= self.threshold
        except Exception as e:
            # Fallback if ONNX hiccups
            vad_detected = False

        # Energy fallback: if RMS energy is clearly speech volume (> 0.02)
        rms = np.sqrt(np.mean(chunk**2))
        energy_detected = rms > 0.025

        return vad_detected or energy_detected