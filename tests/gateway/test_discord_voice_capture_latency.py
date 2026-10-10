"""Quiet Discord packets cannot postpone transcription or create silent turns."""
from array import array
from collections import defaultdict
import threading
from types import SimpleNamespace

from plugins.platforms.discord.adapter import VoiceReceiver


def receiver():
    value = object.__new__(VoiceReceiver)
    value._lock = threading.Lock()
    value._buffers = defaultdict(bytearray)
    value._last_packet_time = {}
    value._ssrc_to_user = {7: 42}
    return value


def test_continuous_quiet_packets_close_spoken_turn(monkeypatch):
    value = receiver()
    clock = [100.0]
    monkeypatch.setattr('plugins.platforms.discord.adapter.time.monotonic', lambda: clock[0])
    speech = array('h', [300] * (48000 * 2)).tobytes()
    value._buffer_pcm(7, speech)
    clock[0] += value.SILENCE_THRESHOLD + .01
    value._buffer_pcm(7, bytes(3840))
    assert value.check_silence() == [(42, speech + bytes(3840))]
    value._buffer_pcm(7, bytes(3840))
    clock[0] += 2
    assert value.check_silence() == []


def test_quiet_packets_alone_never_start_an_utterance():
    value = receiver()
    for _ in range(50):
        value._buffer_pcm(7, bytes(3840))
    assert not value._buffers
    assert not value._last_packet_time
