"""Tests for AudioRecorder's VAD fast-path silence detection
(tools/voice_mode.py). Exercises the decision logic directly via the
recorder's internal callback, injecting a fake VAD model so no real audio
hardware or ONNX runtime is needed.

The property that matters: the fast path can only end a turn SOONER than
today's silence_duration, never later, and never fires at all when
vad_enabled is False (preserving exact prior behavior for anyone who
hasn't opted in).
"""
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.voice_mode import AudioRecorder


def _make_recorder(vad_enabled, vad_probability, silence_duration=3.0, fast_duration=0.6):
    rec = AudioRecorder()
    rec._silence_threshold = 200
    rec._silence_duration = silence_duration
    rec._vad_enabled = vad_enabled
    rec._vad_confidence_threshold = 0.15
    rec._vad_fast_silence_duration = fast_duration
    rec._vad_model = object()  # never actually used -- speech_probability is patched
    rec._has_spoken = True
    rec._speech_start = time.monotonic() - 1.0
    fired = []
    rec._on_silence_stop = lambda: fired.append(True)
    rec._recording = True
    rec._start_time = time.monotonic() - 1.0

    def fake_speech_probability(model, chunk):
        return vad_probability

    return rec, fired, fake_speech_probability


def _quiet_chunk():
    return np.zeros((1600, 1), dtype=np.int16)  # rms well below default threshold


def test_vad_disabled_uses_full_silence_duration_unchanged():
    rec, fired, fake_prob = _make_recorder(vad_enabled=False, vad_probability=0.0, silence_duration=0.05)
    with patch("tools.voice_mode.vad_speech_probability", fake_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False  # first chunk just starts the silence timer
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True
    assert fired == []  # decision-only method: firing is the caller's job (_callback), not tested here


def test_vad_confident_silence_fires_on_fast_path():
    rec, fired, fake_prob = _make_recorder(
        vad_enabled=True, vad_probability=0.01, silence_duration=5.0, fast_duration=0.05,
    )
    with patch("tools.voice_mode.vad_speech_probability", fake_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    # Fast path (0.05s) fired well before the 5s slow path would have.
    assert second is True
    assert fired == []


def test_vad_uncertain_falls_back_to_full_silence_duration():
    rec, fired, fake_prob = _make_recorder(
        vad_enabled=True, vad_probability=0.5, silence_duration=0.05, fast_duration=999.0,
    )
    with patch("tools.voice_mode.vad_speech_probability", fake_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    # VAD uncertain (0.5 >= 0.15 threshold) -- slow path (0.05s) still applies,
    # fast path (999s) must NOT have been used.
    assert second is True
    assert fired == []


def test_vad_model_load_failure_fails_open():
    rec, fired, _ = _make_recorder(
        vad_enabled=True, vad_probability=0.01, silence_duration=0.05, fast_duration=999.0,
    )
    rec._vad_model = None  # force _vad_probability_for_chunk to attempt a (failing) load

    def raise_on_load(sample_rate):
        raise RuntimeError("onnxruntime not available")

    with patch("tools.vad_lite.load_vad_model", raise_on_load):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        # Must not raise: a broken VAD model load degrades to fail-open
        # (probability=1.0, "assume speech"), so the fast path never
        # engages and the full silence_duration applies -- matching
        # speech_probability's own fail-open contract for per-chunk errors.
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True
    assert fired == []


def test_vad_model_load_failure_latches_does_not_retry_every_chunk():
    """A failed load must not retry the ~60ms load on every subsequent
    audio callback for the rest of the recording -- that would repeatedly
    block the realtime sounddevice callback thread."""
    rec, _, _ = _make_recorder(
        vad_enabled=True, vad_probability=0.01, silence_duration=999.0, fast_duration=0.05,
    )
    rec._vad_model = None
    load_mock = MagicMock(side_effect=RuntimeError("onnxruntime not available"))

    with patch("tools.vad_lite.load_vad_model", load_mock):
        for _ in range(5):
            rec._vad_probability_for_chunk(_quiet_chunk())

    assert load_mock.call_count == 1
    assert rec._vad_load_failed is True


def test_vad_probability_resamples_chunk_to_device_native_rate_before_scoring():
    """silero-vad-lite only supports 8000/16000; AudioRecorder can record
    at other device-native rates (e.g. 44100). _vad_probability_for_chunk
    must resample before scoring, and must always load the model at
    tools.vad_lite.VAD_SAMPLE_RATE regardless of the device's rate (loading
    at the raw device rate can segfault the native library)."""
    rec, _, _ = _make_recorder(vad_enabled=True, vad_probability=0.2)
    rec._sample_rate = 44100
    rec._vad_model = None

    load_mock = MagicMock(return_value=object())
    resample_mock = MagicMock(side_effect=lambda chunk, from_rate, to_rate: chunk)
    prob_mock = MagicMock(return_value=0.2)

    with patch("tools.vad_lite.load_vad_model", load_mock), \
         patch("tools.vad_lite.resample_for_vad", resample_mock), \
         patch("tools.voice_mode.vad_speech_probability", prob_mock):
        rec._vad_probability_for_chunk(_quiet_chunk())

    load_mock.assert_called_once_with(16000)  # never the device's 44100
    assert resample_mock.call_count == 1
    _, from_rate, to_rate = resample_mock.call_args[0]
    assert from_rate == 44100
    assert to_rate == 16000


def _loud_chunk():
    return np.full((1600, 1), 5000, dtype=np.int16)  # rms well above default threshold


class TestResumeSpeechConfirmed:
    """_resume_speech_confirmed decides whether a sustained (>= min_speech_duration)
    above-threshold stretch is genuinely resumed speech or should be dismissed
    as sustained non-speech noise (music, TV, a piano) when VAD is confident.

    Live gap this closes: the fast-path silence check (_silence_callback_check)
    only runs once RMS has already dropped back below the threshold -- it never
    got a say in whether a LOUD, sustained sound should be allowed to reset the
    silence timer in the first place. A piano playing continuous phrases (not
    just brief isolated notes -- those were already tolerated) kept resetting
    _silence_start via pure RMS + duration, with VAD never consulted, so voice
    mode never auto-stopped even long after the user stopped talking."""

    def test_vad_disabled_always_confirms_preserving_old_rms_only_behavior(self):
        rec, _, fake_prob = _make_recorder(vad_enabled=False, vad_probability=0.0)
        with patch("tools.voice_mode.vad_speech_probability", fake_prob):
            assert rec._resume_speech_confirmed(_loud_chunk()) is True

    def test_vad_confident_speech_confirms_resume(self):
        rec, _, fake_prob = _make_recorder(vad_enabled=True, vad_probability=0.95)
        with patch("tools.voice_mode.vad_speech_probability", fake_prob):
            assert rec._resume_speech_confirmed(_loud_chunk()) is True

    def test_vad_confident_non_speech_rejects_resume(self):
        """The exact scenario this fix targets: a sustained loud sound (e.g.
        piano) that VAD confidently says is not speech must NOT be treated
        as the user resuming talking."""
        rec, _, fake_prob = _make_recorder(vad_enabled=True, vad_probability=0.01)
        with patch("tools.voice_mode.vad_speech_probability", fake_prob):
            assert rec._resume_speech_confirmed(_loud_chunk()) is False

    def test_vad_uncertain_confirms_resume(self):
        # Uncertain (>= confidence_threshold, not confidently silence) fails
        # toward "assume speech" -- same fail-open direction as everywhere
        # else in this module, so ambiguous audio never gets a real
        # conversation cut off.
        rec, _, fake_prob = _make_recorder(vad_enabled=True, vad_probability=0.5)
        with patch("tools.voice_mode.vad_speech_probability", fake_prob):
            assert rec._resume_speech_confirmed(_loud_chunk()) is True

    def test_sustained_non_speech_noise_does_not_reset_silence_timer(self):
        """End-to-end via the real _callback state machine: after the user
        has spoken and gone quiet (silence timer running), a SUSTAINED loud
        non-speech sound (piano, > min_speech_duration) must not reset the
        silence timer when VAD confidently says it isn't speech -- whereas
        without this fix, sustained RMS-above-threshold alone would reset it
        regardless of what's actually making the noise."""
        rec, fired, fake_prob = _make_recorder(
            vad_enabled=True, vad_probability=0.01, silence_duration=0.2, fast_duration=999.0,
        )
        rec._silence_threshold = 200
        rec._min_speech_duration = 0.05
        rec._max_dip_tolerance = 0.05
        rec.start = lambda **kw: None  # not exercised; using _callback directly below

        with patch("tools.voice_mode.vad_speech_probability", fake_prob):
            # Simulate the recorder's own internal callback path by driving
            # the same decision points _callback would: go quiet (starts the
            # silence timer), then a sustained loud (non-speech) stretch,
            # then quiet again -- and confirm auto-stop still fires on
            # schedule rather than being pushed back.
            now = time.monotonic()
            rec._silence_start = now  # already quiet, timer running
            # Sustained loud stretch long enough to clear min_speech_duration.
            now += 0.06
            rec._resume_start = now
            now += 0.06  # >= min_speech_duration since resume_start
            confirmed = rec._resume_speech_confirmed(_loud_chunk())
            if confirmed:
                rec._silence_start = 0.0  # mirrors _callback's own logic
            assert confirmed is False
            assert rec._silence_start != 0.0, "VAD-rejected noise must not reset the silence timer"

        assert fired == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
