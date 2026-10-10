"""Silero's ONNX speech filter without a Python-version-specific native wrapper."""
from functools import lru_cache
import hashlib
import tempfile
import urllib.request
from pathlib import Path

_REVISION = "be95df9152c0d7618fa1edfeb296fc3dae32376f"  # Silero v6.2
_DIGEST = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
_URL = f"https://raw.githubusercontent.com/snakers4/silero-vad/{_REVISION}/src/silero_vad/data/silero_vad.onnx"


@lru_cache(maxsize=8)
def _session(cache_dir):
    import onnxruntime as ort
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"silero-{_REVISION}.onnx"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != _DIGEST:
        with urllib.request.urlopen(_URL, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != _DIGEST:
            raise ValueError("Silero model checksum mismatch")
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class OnnxSpeechDetector:
    window_size_samples = 512

    def __init__(self):
        import numpy as np
        from hermes_constants import get_hermes_home
        self.session = _session(str(get_hermes_home() / "cache" / "speech-models"))
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def process(self, window):
        import numpy as np
        audio = np.asarray(window, dtype=np.float32)
        if audio.shape != (self.window_size_samples,):
            raise ValueError("Silero requires a full 512-sample window at 16 kHz")
        audio = np.concatenate((self.context, audio.reshape(1, -1)), axis=1)
        probability, state = self.session.run(None, {
            "input": audio, "state": self.state, "sr": np.array(16000, dtype=np.int64),
        })
        self.state = state
        self.context = audio[:, -64:].copy()
        return float(probability.reshape(-1)[0])
