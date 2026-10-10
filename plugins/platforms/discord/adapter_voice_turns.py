"""Speech filtering and semantic completion for Discord's per-speaker PCM buffers."""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class SpeechTurnDetector:
    def __init__(self, config):
        self.threshold = float(config.get("smart_turn_threshold", 0.8))
        self.pause_limit = float(config.get("semantic_pause_limit_seconds", 3.0))
        self.smart = None
        self.vad_factory = None
        self._packet_models = {}
        self._packet_audio = {}
        self._packet_probability = {}
        if config.get("smart_turn_enabled", False):
            try:
                import numpy as np
                from tools.smart_turn import load_smart_turn_model, turn_complete_probability
                self.smart = load_smart_turn_model()
                turn_complete_probability(self.smart, np.zeros(16000, dtype=np.int16))
            except Exception:
                self.smart = None
                logger.warning("Discord semantic detection unavailable; using silence timing", exc_info=True)
        if config.get("vad_enabled", False):
            try:
                from plugins.platforms.discord.adapter_voice_vad import OnnxSpeechDetector
                OnnxSpeechDetector()
                self.vad_factory = OnnxSpeechDetector
            except Exception:
                logger.warning("Discord speech filter unavailable; using amplitude filter", exc_info=True)

    def is_speech(self, ssrc, pcm):
        if self.vad_factory is None:
            return True
        import numpy as np
        if ssrc not in self._packet_models:
            self._packet_models[ssrc] = self.vad_factory()
        model = self._packet_models[ssrc]
        audio = np.concatenate((self._packet_audio.get(ssrc, np.empty(0, dtype=np.int16)), _mono_audio(pcm)))
        window_size = model.window_size_samples
        count = len(audio) // window_size
        probability = self._packet_probability.get(ssrc, 1.0)
        for offset in range(0, count * window_size, window_size):
            probability = float(model.process(audio[offset:offset + window_size].astype(np.float32) / 32768.0))
        self._packet_audio[ssrc] = audio[count * window_size:]
        self._packet_probability[ssrc] = probability
        return probability >= 0.5

    def complete(self, pcm, silence):
        if self.smart is None and self.vad_factory is None:
            return True
        from tools.vad_lite import speech_probability
        audio = _mono_audio(pcm)
        # A fresh VAD state keeps unrelated speakers/utterances from influencing this one.
        if self.vad_factory and speech_probability(self.vad_factory(), audio) < 0.5:
            return "noise"
        if self.smart is not None and silence < self.pause_limit:
            from tools.smart_turn import turn_complete_probability
            probability = turn_complete_probability(self.smart, audio)
            logger.debug("Discord semantic completion probability=%.2f", probability)
            return probability >= self.threshold
        return True


def _mono_audio(pcm):
    import numpy as np
    from tools.vad_lite import resample_for_vad
    audio = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).astype(np.float32).mean(axis=1).astype(np.int16)
    return resample_for_vad(audio, 48000)


def make_detector():
    from hermes_cli.config import load_config
    config = load_config().get("voice", {})
    if config.get("smart_turn_enabled") or config.get("vad_enabled"):
        try:
            import pm
            pm.ensure_import("speech-turn")
        except Exception:
            logger.warning("Discord speech dependencies unavailable; detection will degrade", exc_info=True)
    detector = SpeechTurnDetector(config)
    logger.info("Discord voice detection ready: semantic=%s speech_filter=%s threshold=%.2f pause_limit=%.1fs",
                detector.smart is not None, detector.vad_factory is not None, detector.threshold, detector.pause_limit)
    return detector


def warm_detector(adapter):
    if getattr(adapter, "_voice_detector_task", None) is None:
        adapter._voice_detector_task = asyncio.create_task(asyncio.to_thread(make_detector))


async def ready_detector(adapter):
    warm_detector(adapter)
    return await asyncio.shield(adapter._voice_detector_task)


def buffer_pcm(receiver, ssrc, pcm):
    from array import array
    samples = array("h", pcm)
    audible = bool(samples) and sum(value * value for value in samples) / len(samples) >= 35 ** 2
    detector = getattr(receiver, "_turn_detector", None)
    if samples and detector is not None:
        try:
            speech = detector.is_speech(ssrc, pcm)
            audible = audible and speech
        except Exception:
            logger.debug("Discord packet speech filter failed; using amplitude", exc_info=True)
    with receiver._lock:
        if audible:
            receiver._buffers[ssrc].extend(pcm)
            receiver._last_packet_time[ssrc] = time.monotonic()
        elif receiver._buffers.get(ssrc):
            receiver._buffers[ssrc].extend(pcm)


def check_silence(receiver):
    now = time.monotonic()
    with receiver._lock:
        candidates = [(ssrc, bytes(buf), receiver._last_packet_time.get(ssrc, now))
                      for ssrc, buf in receiver._buffers.items()]
    completed = []
    for ssrc, pcm, last_time in candidates:
        silence = now - last_time
        duration = len(pcm) / (receiver.SAMPLE_RATE * receiver.CHANNELS * 2)
        if silence < receiver.SILENCE_THRESHOLD:
            continue
        decision = True
        detector = getattr(receiver, "_turn_detector", None)
        if duration >= receiver.MIN_SPEECH_DURATION and detector is not None:
            try:
                decision = detector.complete(pcm, silence)
            except Exception:
                logger.warning("Discord turn detection failed; using silence timing", exc_info=True)
        if decision is False:
            continue
        with receiver._lock:
            # Inference runs without blocking UDP intake. Speech arriving meanwhile
            # invalidates the candidate, so it cannot split a continuing utterance.
            if receiver._last_packet_time.get(ssrc, now) != last_time:
                continue
            if duration >= receiver.MIN_SPEECH_DURATION:
                user = receiver._ssrc_to_user.get(ssrc, 0) or receiver._infer_user_for_ssrc(ssrc)
                if user and decision != "noise":
                    completed.append((user, bytes(receiver._buffers[ssrc])))
                receiver._buffers.pop(ssrc, None)
                receiver._last_packet_time.pop(ssrc, None)
            elif silence >= receiver.SILENCE_THRESHOLD * 2:
                receiver._buffers.pop(ssrc, None)
                receiver._last_packet_time.pop(ssrc, None)
    return completed
