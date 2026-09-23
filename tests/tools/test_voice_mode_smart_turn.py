"""Tests for AudioRecorder's Smart Turn semantic gate (tools/voice_mode.py),
layered on top of the VAD/RMS acoustic gate (tests/tools/test_voice_mode_vad.py).

The property that matters: Smart Turn can only end a turn LATER than the
acoustic decision alone (an extension), never sooner, is capped at
smart_turn_max_extensions, and fails open (lets the turn fire) on any
error -- the opposite fail-open bias from VAD, since a broken semantic
layer must never hang the recorder forever.
"""
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.voice_mode import AudioRecorder


def _make_recorder(
    smart_turn_enabled, turn_probability, silence_duration=0.05,
    confidence_threshold=0.5, extend_seconds=0.05, max_extensions=2,
):
    rec = AudioRecorder()
    rec._silence_threshold = 200
    rec._silence_duration = silence_duration
    rec._vad_enabled = False
    rec._smart_turn_enabled = smart_turn_enabled
    rec._smart_turn_confidence_threshold = confidence_threshold
    rec._smart_turn_extend_seconds = extend_seconds
    rec._smart_turn_max_extensions = max_extensions
    rec._smart_turn_model = object()  # never actually used -- probability fn is patched
    rec._has_spoken = True
    rec._speech_start = time.monotonic() - 1.0
    rec._recording = True
    rec._start_time = time.monotonic() - 1.0
    rec._frames = [np.zeros((1600, 1), dtype=np.int16)]

    def fake_probability(session, audio):
        return turn_probability

    return rec, fake_probability


def _quiet_chunk():
    return np.zeros((1600, 1), dtype=np.int16)


def test_disabled_is_a_complete_noop():
    """smart_turn_enabled=False must not change VAD/RMS-only behavior at all."""
    rec, _ = _make_recorder(smart_turn_enabled=False, turn_probability=0.0)
    mock_prob = MagicMock(return_value=0.0)
    with patch("tools.voice_mode.smart_turn_complete_probability", mock_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True
    mock_prob.assert_not_called()


def test_unfinished_utterance_extends_instead_of_firing():
    rec, fake_prob = _make_recorder(
        smart_turn_enabled=True, turn_probability=0.1,  # confidently "incomplete"
        silence_duration=0.05, extend_seconds=0.2, max_extensions=2,
    )
    with patch("tools.voice_mode.smart_turn_complete_probability", fake_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        # Acoustic layer wants to fire (0.05s elapsed) but Smart Turn says unfinished.
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is False
    assert rec._smart_turn_extension_count == 1


def test_finished_utterance_fires_normally():
    rec, fake_prob = _make_recorder(
        smart_turn_enabled=True, turn_probability=0.9,  # confidently "complete"
        silence_duration=0.05,
    )
    with patch("tools.voice_mode.smart_turn_complete_probability", fake_prob):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True
    assert rec._smart_turn_extension_count == 0


def test_extension_grants_more_time_than_acoustic_alone():
    """The whole point: an unfinished read must make the turn end LATER than the
    acoustic-only decision would have, never sooner."""
    rec, fake_prob = _make_recorder(
        smart_turn_enabled=True, turn_probability=0.1,
        silence_duration=0.05, extend_seconds=0.3, max_extensions=3,
    )
    with patch("tools.voice_mode.smart_turn_complete_probability", fake_prob):
        rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        time.sleep(0.06)
        # Acoustic layer alone would fire here (0.05s silence_duration elapsed) --
        # must NOT fire because Smart Turn reads it as unfinished.
        extended = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert extended is False
        # Only 0.06s later -- less than the 0.3s extension -- must still not fire.
        time.sleep(0.06)
        still_extended = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert still_extended is False


def test_extension_count_is_a_hard_cap():
    """After smart_turn_max_extensions unfinished reads, the turn fires regardless of
    what Smart Turn thinks -- the safety bound against a systematically wrong read or a
    genuinely very long trailing pause."""
    rec, fake_prob = _make_recorder(
        smart_turn_enabled=True, turn_probability=0.0,  # always reads as "unfinished"
        silence_duration=0.03, extend_seconds=0.03, max_extensions=2,
    )
    with patch("tools.voice_mode.smart_turn_complete_probability", fake_prob):
        rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        results = []
        for _ in range(6):
            time.sleep(0.04)
            results.append(rec._silence_callback_check(_quiet_chunk(), now=time.monotonic()))
    assert rec._smart_turn_extension_count == 2
    assert results[-1] is True  # eventually fires despite Smart Turn always saying "unfinished"
    assert True in results


def test_new_recording_resets_extension_count():
    rec, fake_prob = _make_recorder(smart_turn_enabled=True, turn_probability=0.1)
    rec._smart_turn_extension_count = 2
    rec._reset_detection_state()
    assert rec._smart_turn_extension_count == 0


def test_model_load_failure_fails_open_toward_firing():
    """Opposite bias from VAD: a broken Smart Turn must let the turn END, not hang."""
    rec, _ = _make_recorder(smart_turn_enabled=True, turn_probability=0.0, silence_duration=0.05)
    rec._smart_turn_model = None  # force a (failing) load attempt

    with patch("tools.smart_turn.load_smart_turn_model", side_effect=RuntimeError("no onnxruntime")):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True
    assert rec._smart_turn_load_failed is True


def test_model_load_failure_latches_does_not_retry_every_chunk():
    rec, _ = _make_recorder(smart_turn_enabled=True, turn_probability=0.0)
    rec._smart_turn_model = None
    load_mock = MagicMock(side_effect=RuntimeError("no onnxruntime"))

    with patch("tools.smart_turn.load_smart_turn_model", load_mock):
        for _ in range(5):
            rec._smart_turn_probability_for_buffer()

    assert load_mock.call_count == 1
    assert rec._smart_turn_load_failed is True


def test_inference_failure_fails_open_toward_firing():
    rec, _ = _make_recorder(smart_turn_enabled=True, turn_probability=0.0, silence_duration=0.05)

    def raise_on_score(session, audio):
        raise RuntimeError("inference blew up")

    with patch("tools.voice_mode.smart_turn_complete_probability", raise_on_score):
        first = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        assert first is False
        time.sleep(0.06)
        second = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert second is True


def test_composes_with_vad_fast_path():
    """VAD can fire sooner (acoustic layer), Smart Turn can extend later (semantic
    layer) -- both active at once must still respect Smart Turn's extension."""
    rec, fake_turn_prob = _make_recorder(
        smart_turn_enabled=True, turn_probability=0.1,
        silence_duration=5.0, extend_seconds=0.2, max_extensions=1,
    )
    rec._vad_enabled = True
    rec._vad_confidence_threshold = 0.15
    rec._vad_fast_silence_duration = 0.05
    rec._vad_model = object()

    def fake_vad_prob(model, chunk):
        return 0.01  # confidently silent -- VAD fast path applies

    with patch("tools.voice_mode.vad_speech_probability", fake_vad_prob), \
         patch("tools.voice_mode.smart_turn_complete_probability", fake_turn_prob):
        rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
        time.sleep(0.06)
        # VAD fast path (0.05s) would fire alone, but Smart Turn extends it.
        result = rec._silence_callback_check(_quiet_chunk(), now=time.monotonic())
    assert result is False
    assert rec._smart_turn_extension_count == 1
