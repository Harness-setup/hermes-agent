"""Cancelable working reminders; connection greetings belong to the voice owner."""
import asyncio
import contextlib
import json
import logging
import tempfile
import threading
from pathlib import Path

from gateway.config import Platform
from gateway.voice_profile import PhrasePicker, resolve_voice_profile

logger = logging.getLogger(__name__)

_DISCORD_TOOL_NOTE = (
    "[Use dedicated tools when they support the user's task. Do not probe this computer, run shell "
    "commands, or install packages just to answer a general question about a product, weather, or model "
    "mode. Use terminal when the requested task actually needs shell execution. If a command is denied, "
    "do not retry it through another shell command; explain the limitation and continue with available tools.]"
)


def voice_channel_enabled(runner, source, *, voice_input=False):
    if source.platform != Platform.DISCORD:
        return None
    adapter = runner._delivery_adapter_for(source)
    if adapter is None:
        return None
    mode = runner._voice_mode.get(runner._voice_key_for_source(source))
    enabled = mode == "all" or (mode == "voice_only" and voice_input)
    if mode is None:
        enabled = bool(adapter._should_auto_tts_for_chat(source.chat_id))
    if not enabled:
        return None
    channels = getattr(adapter, "_voice_text_channels", {})
    if not any(str(chat) == str(source.chat_id) and adapter.is_in_voice_channel(guild) for guild, chat in channels.items()):
        return None
    return adapter


def _render_phrase(text):
    from tools.tts_tool import text_to_speech_tool
    # Own temporary files inside the worker even if the awaiting turn is canceled.
    with tempfile.TemporaryDirectory(prefix="hermes-voice-phrase-") as directory:
        result = json.loads(text_to_speech_tool(text=text, output_path=str(Path(directory) / "phrase.wav")))
        if not result.get("success"):
            return None
        path = Path(result["file_path"])
        return path.read_bytes(), path.suffix


async def speak_phrase(adapter, chat_id, text, *, allowed=lambda: True):
    try:
        rendered = await asyncio.to_thread(_render_phrase, text)
        if rendered is None or not allowed():
            return
        data, suffix = rendered
        with tempfile.TemporaryDirectory(prefix="hermes-voice-play-") as directory:
            path = Path(directory) / ("phrase" + suffix)
            path.write_bytes(data)
            if allowed():
                await adapter.play_tts(chat_id, str(path), allowed=allowed)
    except Exception:
        logger.debug("Voice phrase failed; text delivery continues", exc_info=True)


class VoicePhraseTurn:
    def __init__(self, adapter, chat_id, profile, current, registry=None, session_key=None):
        self.adapter = adapter
        self.picker = phrase_picker(adapter)
        self.chat_id = chat_id
        self.profile = profile
        self.current = current
        self._ended = threading.Event()
        self._loop = asyncio.get_running_loop()
        self._task = None
        self._registry = registry
        self._session_key = session_key

    def allowed(self):
        return not self._ended.is_set() and self.current()

    async def start(self):
        if self.allowed():
            self._task = asyncio.create_task(self._filler())

    async def _filler(self):
        timing = self.profile["timing"]
        await asyncio.sleep(max(timing["status_silence_seconds"], timing["filler_delay_seconds"]))
        if self.allowed():
            await speak_phrase(self.adapter, self.chat_id,
                               self.picker.choose("fillers", self.profile["phrases"]["fillers"]), allowed=self.allowed)

    def answer_started(self):
        # Called from the agent thread: latch first, before scheduling cancellation.
        self._ended.set()
        if self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)

    async def close(self):
        self.answer_started()
        if self._registry is not None and self._registry.get(self._session_key) is self:
            self._registry.pop(self._session_key)
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


def make_voice_phrase_turn(runner, source, voice_input, current, session_key=None):
    if not voice_input:
        return None
    try:
        adapter = voice_channel_enabled(runner, source, voice_input=True)
        if adapter is not None:
            profile = resolve_voice_profile()
            registry = getattr(runner, "_voice_phrase_turns", None)
            if registry is None:
                registry = runner._voice_phrase_turns = {}
            turn = VoicePhraseTurn(adapter, source.chat_id, profile, current, registry, session_key)
            registry[session_key] = turn
            return turn
    except Exception:
        logger.debug("Voice profile unavailable; text delivery continues", exc_info=True)
    return None


async def speak_event_phrase(runner, source, group, *, voice_input=False):
    try:
        with runner._profile_scope_for_source(source):
            adapter = voice_channel_enabled(runner, source, voice_input=voice_input)
            if adapter is not None:
                phrase = phrase_picker(adapter).choose(group, resolve_voice_profile()["phrases"][group])
                await speak_phrase(adapter, source.chat_id, phrase)
    except Exception:
        logger.debug("Voice status unavailable; text delivery continues", exc_info=True)


def proxy_answer_started(runner, session_key):
    turn = getattr(runner, "_voice_phrase_turns", {}).get(session_key)
    if turn is not None:
        turn.answer_started()


def turn_reply_note(voice_input, platform=None, voice_mode=None):
    if voice_input or (platform == Platform.DISCORD and voice_mode == "all"):
        from tui_gateway.voice_reply_style import voice_chat_turn_note
        note = voice_chat_turn_note()
        return f"{note}\n{_DISCORD_TOOL_NOTE}" if platform == Platform.DISCORD else note
    if platform == Platform.DISCORD:
        return (
            "[Discord text conversation: Reply in normal readable text. Use formatting when useful. "
            "Keep the configured persona and form of address. This message was typed, not spoken. "
            "Do not describe hearing the user, listening through a microphone, or speaking the reply aloud. "
            "Do not carry voice-call instructions from earlier turns into this text reply.]\n" + _DISCORD_TOOL_NOTE
        )
    return ""


def spoken_turn_message(message, voice_input, platform=None, voice_mode=None):
    note = turn_reply_note(voice_input, platform, voice_mode)
    return f"{note}\n\n{message}" if note else message


def phrase_picker(adapter):
    picker = getattr(adapter, "_spoken_phrase_picker", None)
    if picker is None:
        picker = PhrasePicker()
        adapter._spoken_phrase_picker = picker
    return picker
