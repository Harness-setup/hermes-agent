"""Silero VAD wrapper for AudioRecorder's fast-path silence detection
(tools/voice_mode.py).

Uses silero-vad-lite (zero Python dependencies, bundled ONNX Runtime CPU
backend) rather than jarvis-voice's torch-based silero-vad package, since
hermes-agent has no existing torch dependency and this keeps the wake/voice
extras onnx-based, consistent with Sherpa-ONNX and openWakeWord already
used here. The fail-open contract mirrors jarvis-voice's vad.py exactly:
if VAD itself breaks, assume speech might be present rather than going
silently deaf.

silero-vad-lite's native library only accepts a sample rate of 8000 or
16000 -- passing anything else does not raise a Python exception, it
segfaults the process (a null model handle from the C constructor gets
dereferenced immediately after). AudioRecorder records at the input
device's native rate (44100 Hz is common, not 16 kHz), so callers MUST
always load the model at VAD_SAMPLE_RATE and resample chunks to that rate
with resample_for_vad before calling speech_probability -- never pass the
device's raw capture rate to load_vad_model.
"""
import numpy as np

VAD_SAMPLE_RATE = 16000
_WINDOW_SIZE = 512  # fallback only; speech_probability prefers model.window_size_samples


def load_vad_model(sample_rate: int = VAD_SAMPLE_RATE):
    """Loads the Silero VAD model. Raises ValueError (catchable) for any
    rate other than 8000/16000, rather than letting the native library
    segfault on an unsupported rate."""
    if sample_rate not in (8000, 16000):
        raise ValueError(f"unsupported VAD sample rate: {sample_rate} (must be 8000 or 16000)")
    from silero_vad_lite import SileroVAD
    return SileroVAD(sample_rate)


def resample_for_vad(chunk_int16, from_rate: int, to_rate: int = VAD_SAMPLE_RATE):
    """Linear-interpolation resample of an int16 PCM chunk to to_rate.
    Adequate for VAD's speech/silence classification -- not intended as
    audio-fidelity resampling. No-op when from_rate already matches
    to_rate (returns chunk unchanged)."""
    if chunk_int16.size == 0 or from_rate == to_rate:
        return chunk_int16
    duration = chunk_int16.size / float(from_rate)
    target_length = max(1, int(round(duration * to_rate)))
    original_indices = np.arange(chunk_int16.size)
    target_indices = np.linspace(0, chunk_int16.size - 1, num=target_length)
    resampled = np.interp(target_indices, original_indices, chunk_int16.astype(np.float32))
    return resampled.astype(np.int16)


def speech_probability(model, chunk_int16) -> float:
    """Max per-window speech probability across chunk, in [0, 1]. Windows
    internally at the model's own window_size_samples (falls back to the
    fixed 512 used by silero-vad-lite at 16kHz if the model doesn't expose
    that attribute, e.g. a test double) -- the final partial window (if
    any) is zero-padded rather than dropped. Fails open (returns 1.0 --
    "assume speech") if the model raises, so a broken VAD degrades callers
    to RMS-only behavior instead of silent deafness."""
    try:
        if chunk_int16.size == 0:
            return 0.0
        window_size = getattr(model, "window_size_samples", _WINDOW_SIZE)
        float_chunk = chunk_int16.astype(np.float32) / 32768.0
        max_probability = 0.0
        for start in range(0, float_chunk.size, window_size):
            window = float_chunk[start:start + window_size]
            if window.size < window_size:
                window = np.pad(window, (0, window_size - window.size))
            probability = model.process(window)
            max_probability = max(max_probability, probability)
        return max_probability
    except Exception:
        return 1.0
