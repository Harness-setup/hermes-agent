"""Tests for tools/vad_lite.py -- the Silero VAD wrapper used by
AudioRecorder's fast-path silence detection (tools/voice_mode.py).

speech_probability's fail-open contract is the safety property that
matters most here: if VAD itself breaks, callers must fall back to
today's RMS-only behavior, never to a more "deaf" state.
"""
import numpy as np
import pytest

from tools.vad_lite import VAD_SAMPLE_RATE, load_vad_model, resample_for_vad, speech_probability


class _FakeModel:
    """Records every window it's called with and returns a preset
    probability per call (or the same one every time)."""

    def __init__(self, probabilities):
        self.calls = []
        self._probabilities = list(probabilities)

    def process(self, window):
        self.calls.append(np.array(window))
        return self._probabilities.pop(0) if self._probabilities else 0.0


def test_empty_chunk_returns_zero():
    model = _FakeModel([])
    assert speech_probability(model, np.array([], dtype=np.int16)) == 0.0
    assert model.calls == []


def test_single_window_returns_that_windows_probability():
    model = _FakeModel([0.73])
    chunk = np.zeros(512, dtype=np.int16)
    assert speech_probability(model, chunk) == 0.73
    assert len(model.calls) == 1


def test_multiple_windows_returns_max_probability():
    model = _FakeModel([0.1, 0.9, 0.4])
    chunk = np.zeros(512 * 3, dtype=np.int16)
    assert speech_probability(model, chunk) == 0.9
    assert len(model.calls) == 3


def test_partial_final_window_is_zero_padded_not_dropped():
    model = _FakeModel([0.2, 0.5])
    chunk = np.zeros(512 + 100, dtype=np.int16)  # second window only 100 samples
    result = speech_probability(model, chunk)
    assert result == 0.5
    assert len(model.calls) == 2
    assert model.calls[1].shape == (512,)  # padded up to the fixed window size


def test_int16_input_is_normalized_to_float32_range():
    model = _FakeModel([0.0])
    chunk = np.full(512, 32767, dtype=np.int16)  # max positive int16
    speech_probability(model, chunk)
    window = model.calls[0]
    assert window.dtype == np.float32
    assert 0.99 <= window.max() <= 1.0


def test_model_exception_fails_open_to_one():
    class _BrokenModel:
        def process(self, window):
            raise RuntimeError("onnx runtime crashed")

    chunk = np.zeros(512, dtype=np.int16)
    assert speech_probability(_BrokenModel(), chunk) == 1.0


def test_windows_at_models_own_window_size_not_hardcoded_512():
    """silero-vad-lite's window size is a property of the loaded model
    (256 samples at 8kHz, 512 at 16kHz) -- feeding it the wrong window size
    raises inside the model. speech_probability must ask the model, not
    assume 16kHz's 512."""
    model = _FakeModel([0.4, 0.6])
    model.window_size_samples = 256
    chunk = np.zeros(256 * 2, dtype=np.int16)
    assert speech_probability(model, chunk) == 0.6
    assert len(model.calls) == 2
    assert model.calls[0].shape == (256,)


def test_windowing_falls_back_to_512_when_model_has_no_window_size_attr():
    # A test double (or an unexpected model implementation) without
    # window_size_samples must not crash -- fall back to the 16kHz default.
    model = _FakeModel([0.5])
    chunk = np.zeros(512, dtype=np.int16)
    assert speech_probability(model, chunk) == 0.5


class TestLoadVadModel:
    def test_rejects_device_native_rate_instead_of_segfaulting(self):
        """silero-vad-lite's native constructor does not raise on an
        unsupported rate (e.g. 44100, a common device default) -- it
        segfaults the process via a null model handle. load_vad_model must
        validate first so this becomes a normal, catchable exception."""
        with pytest.raises(ValueError, match="8000 or 16000"):
            load_vad_model(44100)

    def test_rejects_zero_and_negative_rates(self):
        with pytest.raises(ValueError):
            load_vad_model(0)

    def test_accepts_16000(self):
        model = load_vad_model(16000)
        assert model.window_size_samples == 512

    def test_accepts_8000(self):
        model = load_vad_model(8000)
        assert model.window_size_samples == 256

    def test_default_argument_is_vad_sample_rate_constant(self):
        model = load_vad_model()
        assert model.window_size_samples == 512
        assert VAD_SAMPLE_RATE == 16000


class TestResampleForVad:
    def test_noop_when_rates_already_match(self):
        chunk = np.arange(100, dtype=np.int16)
        result = resample_for_vad(chunk, from_rate=16000, to_rate=16000)
        assert result is chunk

    def test_empty_chunk_returns_unchanged(self):
        chunk = np.array([], dtype=np.int16)
        result = resample_for_vad(chunk, from_rate=44100, to_rate=16000)
        assert result.size == 0

    def test_downsamples_44100_to_16000_produces_expected_length(self):
        # 44100 samples at 44100Hz = 1.0s -> 16000 samples at 16000Hz
        chunk = np.zeros(44100, dtype=np.int16)
        result = resample_for_vad(chunk, from_rate=44100, to_rate=16000)
        assert result.dtype == np.int16
        assert abs(result.size - 16000) <= 1

    def test_preserves_constant_signal_value(self):
        # A constant-value chunk resampled should stay (near) that same
        # constant value -- sanity check that this isn't scrambling data.
        chunk = np.full(4410, 1000, dtype=np.int16)
        result = resample_for_vad(chunk, from_rate=44100, to_rate=16000)
        assert result.size > 0
        assert np.all(np.abs(result.astype(np.int32) - 1000) <= 1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
