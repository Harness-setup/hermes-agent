"""Smart Turn v3.1 semantic turn-completion classifier (Pipecat/Daily,
onnx-community/smart-turn-v3-ONNX, BSD-2-Clause, ~8M params, int8 ONNX).

Distinguishes "the user is done talking" from "the user paused mid-thought"
using the audio itself (intonation, pace) -- unlike tools/vad_lite.py's pure
acoustic speech/silence classification, this is a semantic judgment about
whether the utterance SOUNDS finished. Runs once per candidate end-of-turn
pause (not per chunk) -- ~12ms CPU inference is negligible at that rate.

Lazy-loaded the same way as vad_lite.py's model: nothing imports/loads at
module scope, so a recorder with smart_turn_enabled=False (the default)
pays zero cost. Input shape, feature extraction, and the sigmoid threshold
convention below are verified against the model's own reference
implementation (pipecat-ai/smart-turn's inference.py/audio_utils.py), not
guessed -- both `transformers` and `onnxruntime` are already dependencies
here (pulled in by NeuTTS and vad_lite respectively).
"""
from __future__ import annotations

SMART_TURN_SAMPLE_RATE = 16000
_SMART_TURN_REPO = "onnx-community/smart-turn-v3-ONNX"
_SMART_TURN_ONNX_FILE = "onnx/model_int8.onnx"
_SMART_TURN_WINDOW_SECONDS = 8


def load_smart_turn_model():
    """Download (first use; cached by huggingface_hub thereafter) + build an
    onnxruntime.InferenceSession with the model's own recommended CPU
    session options. Raises on failure -- caller is responsible for
    fail-open/latch behavior, same division of responsibility as
    tools.vad_lite.load_vad_model."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=_SMART_TURN_REPO, filename=_SMART_TURN_ONNX_FILE)
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so)


def _truncate_or_pad_to_window(audio_array):
    """Last _SMART_TURN_WINDOW_SECONDS of audio, zero-padded at the START if
    shorter -- matches the reference implementation's own convention
    (pipecat-ai/smart-turn's audio_utils.truncate_audio_to_last_n_seconds)
    exactly, since the model was trained on inputs shaped this way."""
    import numpy as np
    max_samples = _SMART_TURN_WINDOW_SECONDS * SMART_TURN_SAMPLE_RATE
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    if len(audio_array) < max_samples:
        return np.pad(audio_array, (max_samples - len(audio_array), 0), mode="constant")
    return audio_array


def turn_complete_probability(session, audio_array_int16) -> float:
    """``audio_array_int16``: already resampled to 16kHz (see
    tools.vad_lite.resample_for_vad -- same target rate, reused directly,
    not reimplemented), int16 PCM -- same calling convention as
    tools.vad_lite.speech_probability's ``chunk_int16``, normalized to
    float32 in [-1, 1] here rather than pushed onto the caller. Returns the
    probability that this utterance is COMPLETE (>0.5 = complete, matching
    the reference implementation's threshold convention -- callers should
    treat the raw float as the source of truth and pick their own
    threshold rather than assume this function binarizes).

    Verified live against the actual onnx-community/smart-turn-v3-ONNX
    int8 file: its output tensor is named "logits" and returns raw
    (pre-sigmoid) values, e.g. 3.57 for a 2-second silence input --
    unlike pipecat-ai/smart-turn's own reference inference.py, which
    documents its (differently-exported) ONNX file as already returning
    sigmoid probabilities. Sigmoid is applied here explicitly rather than
    trusting that comment for this specific community export."""
    import numpy as np
    from transformers import WhisperFeatureExtractor

    float_audio = audio_array_int16.astype(np.float32) / 32768.0
    windowed = _truncate_or_pad_to_window(float_audio)
    feature_extractor = WhisperFeatureExtractor(chunk_length=_SMART_TURN_WINDOW_SECONDS)
    inputs = feature_extractor(
        windowed, sampling_rate=SMART_TURN_SAMPLE_RATE, return_tensors="np",
        padding="max_length", max_length=_SMART_TURN_WINDOW_SECONDS * SMART_TURN_SAMPLE_RATE,
        truncation=True, do_normalize=True,
    )
    input_features = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
    outputs = session.run(None, {"input_features": input_features})
    logit = float(outputs[0][0].item())
    return float(1.0 / (1.0 + np.exp(-logit)))
