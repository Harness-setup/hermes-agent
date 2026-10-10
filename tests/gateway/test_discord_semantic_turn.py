"""A pause in an unfinished Discord utterance does not submit a partial prompt."""
from array import array
from unittest.mock import Mock

from tests.gateway.test_discord_voice_capture_latency import receiver
from plugins.platforms.discord.adapter_voice_turns import SpeechTurnDetector


def test_onnx_vad_keeps_context_and_state_separate_for_each_speaker(monkeypatch):
    import numpy as np
    from plugins.platforms.discord import adapter_voice_vad
    calls = []
    def run(outputs, inputs):
        calls.append({key: value.copy() for key, value in inputs.items()})
        return np.array([[0.9]], dtype=np.float32), np.ones((2, 1, 128), dtype=np.float32)
    monkeypatch.setattr(adapter_voice_vad, '_session', lambda path: Mock(run=run))
    first = adapter_voice_vad.OnnxSpeechDetector()
    second = adapter_voice_vad.OnnxSpeechDetector()
    window = np.ones(512, dtype=np.float32)
    assert first.process(window) > 0.5
    first.process(window)
    second.process(window)
    assert not calls[0]['state'].any()
    assert calls[1]['state'].all()
    assert calls[1]['input'][0, :64].all()
    assert not calls[2]['state'].any()
    assert not calls[2]['input'][0, :64].any()
    assert calls[0]['sr'] == 16000


def test_packet_vad_accumulates_full_windows_without_padding(monkeypatch):
    import numpy as np
    detector = SpeechTurnDetector({})
    windows = []
    def process(window):
        windows.append(window.copy())
        return float(np.all(window > 0))
    detector.vad_factory = lambda: Mock(window_size_samples=512, process=process)
    monkeypatch.setattr('plugins.platforms.discord.adapter_voice_turns._mono_audio',
                        lambda pcm: np.full(320, 300, dtype=np.int16))
    assert detector.is_speech(7, b'packet')
    assert windows == []
    assert detector.is_speech(7, b'packet')
    assert len(windows) == 1
    assert len(windows[0]) == 512
    assert np.all(windows[0] > 0)
    assert len(detector._packet_audio[7]) == 128


def test_unfinished_pause_keeps_audio_until_semantically_complete(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    value._turn_detector = Mock()
    value._turn_detector.complete.side_effect = [False, True]
    speech = array('h', [300] * (48000 * 2)).tobytes()
    value._buffer_pcm(7, speech)
    clock[0] += 0.9
    assert value.check_silence() == []
    value._buffer_pcm(7, speech)
    clock[0] += 0.9
    assert value.check_silence() == [(42, speech * 2)]


def test_background_noise_is_discarded_without_transcription(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    value._turn_detector = Mock()
    value._turn_detector.complete.return_value = "noise"
    value._buffer_pcm(7, array('h', [300] * (48000 * 2)).tobytes())
    clock[0] += 0.9
    assert value.check_silence() == []
    assert not value._buffers


def test_non_speech_packets_do_not_start_or_extend_voice_activity(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    value._turn_detector = Mock()
    value._turn_detector.is_speech.side_effect = [False, True, False]
    pcm = array('h', [300] * (48000 * 2)).tobytes()
    value._buffer_pcm(7, pcm)
    assert not value._buffers
    value._buffer_pcm(7, pcm)
    clock[0] += 0.9
    value._buffer_pcm(7, pcm)
    assert value._last_packet_time[7] == 100.0


def test_speech_arriving_during_inference_cannot_be_split(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    speech = array('h', [300] * (48000 * 2)).tobytes()
    value._buffer_pcm(7, speech)
    clock[0] += 0.9
    def complete(*args):
        value._buffer_pcm(7, speech)
        return True
    value._turn_detector = Mock(complete=complete)
    assert value.check_silence() == []
    assert len(value._buffers[7]) == len(speech) * 2


def test_uncertain_semantics_have_a_bounded_wait(monkeypatch):
    detector = SpeechTurnDetector({})
    detector.smart = object()
    monkeypatch.setattr('tools.smart_turn.turn_complete_probability', lambda *args: 0.79)
    pcm = array('h', [300] * (48000 * 2)).tobytes()
    assert detector.complete(pcm, 0.8) is False
    assert detector.complete(pcm, 3.0) is True


def test_semantic_inference_failure_keeps_silence_fallback(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    value._turn_detector = Mock()
    value._turn_detector.complete.side_effect = RuntimeError('model unavailable')
    speech = array('h', [300] * (48000 * 2)).tobytes()
    value._buffer_pcm(7, speech)
    clock[0] += 0.9
    assert value.check_silence() == [(42, speech)]
