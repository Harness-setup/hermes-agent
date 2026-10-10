"""Cancelable Discord voice playback shared by replies and fixed phrases."""
import asyncio
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def play_in_voice_channel(self, guild_id: int, audio_path: str, *, allowed=None) -> bool:
    """Play audio in the VC: via the mixer (layered over the ambient bed, ducking it)
    when installed, else the legacy one-shot FFmpegPCMAudio path."""
    from plugins.platforms.discord.adapter import (
        _voice_mixer_module, resolve_ffmpeg_executable, discord, logger,
    )
    vc = self._voice_clients.get(guild_id)
    if not vc or not vc.is_connected():
        return False
    # Playback counts as activity: suspend the inactivity timer, re-arm in finally.
    self._cancel_voice_timeout(guild_id)
    speech_started = False
    mixer = None
    try:
        if allowed is not None and not allowed():
            return False
        playback_timeout = await self._playback_timeout_for_audio(audio_path)
        # ── Mixer path (overlap + ducking) ──────────────────────────────
        mixer = getattr(self, "_voice_mixers", {}).get(guild_id) if getattr(self, "_voice_mixers", None) else None
        if mixer is not None:
            decode_to_pcm = _voice_mixer_module().decode_to_pcm
            pcm = await asyncio.to_thread(decode_to_pcm, audio_path)
            if pcm:
                if allowed is not None and not allowed():
                    return False
                speech_started = True
                speech_gain = float(self._voice_fx_cfg.get("speech_gain", 1.0))
                mixer.play_speech(self._lead_silence_bytes() + pcm, gain=speech_gain)
                # Block until speech drains so callers serialise replies; ambient keeps playing.
                wait_start = time.monotonic()
                while mixer.speech_active:
                    if time.monotonic() - wait_start > playback_timeout:
                        logger.warning("Mixer speech playback timed out after %.1fs", playback_timeout)
                        mixer.stop_speech()
                        break
                    await asyncio.sleep(0.05)
                return True
            logger.warning("Mixer decode failed for %s; falling back to legacy playback", audio_path)
        # Legacy one-shot path: pause receiver while playing (echo prevention).
        receiver = self._voice_receivers.get(guild_id)
        if receiver:
            receiver.pause()
        try:
            wait_start = time.monotonic()
            while vc.is_playing():
                if time.monotonic() - wait_start > playback_timeout:
                    logger.warning("Timed out waiting for previous playback to finish")
                    vc.stop()
                    break
                await asyncio.sleep(0.1)
            done = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _after(error):
                if error:
                    logger.error("Voice playback error: %s", error)
                loop.call_soon_threadsafe(done.set)
            # Lead silence so socket warm-up doesn't clip the first word (mirrors mixer path).
            ffmpeg_opts: Dict[str, Any] = {}
            _fx_cfg = getattr(self, "_voice_fx_cfg", None) or {}
            try:
                lead_ms = int(_fx_cfg.get("lead_silence_ms", 0) or 0)
            except (TypeError, ValueError):
                lead_ms = 0
            if lead_ms > 0:
                ffmpeg_opts["options"] = f"-af adelay={lead_ms}:all=1"
            source = discord.FFmpegPCMAudio(
                audio_path, executable=resolve_ffmpeg_executable(), **ffmpeg_opts,
            )
            source = discord.PCMVolumeTransformer(source, volume=1.0)
            if allowed is not None and not allowed():
                source.cleanup()
                return False
            speech_started = True
            vc.play(source, after=_after)
            try:
                await asyncio.wait_for(done.wait(), timeout=playback_timeout)
            except asyncio.TimeoutError:
                logger.warning("Voice playback timed out after %.1fs", playback_timeout)
                vc.stop()
            return True
        finally:
            if receiver:
                receiver.resume()
    except asyncio.CancelledError:
        if speech_started:
            if mixer is not None:
                mixer.stop_speech()
            else:
                vc.stop()
        raise
    finally:
        self._reset_voice_timeout(guild_id)



def claim_voice_owner(adapter, member, channel):
    """First successful initiating human owns this bot's guild connection."""
    guild_id = channel.guild.id
    if not adapter.is_in_voice_channel(guild_id):
        return
    owner = adapter._voice_owners.get(guild_id)
    if owner is not None and owner[1] == channel.id:
        return
    existing = adapter._voice_greetings.pop(guild_id, None)
    if existing is not None:
        existing.cancel()
    adapter._voice_owners[guild_id] = (int(member.id), channel.id)
    adapter._voice_owner_channels[guild_id] = channel.id
    _greet_voice_owner(adapter, guild_id)


def _greet_voice_owner(adapter, guild_id):
    from gateway.voice_phrases import phrase_picker, speak_phrase
    from gateway.voice_profile import resolve_voice_profile
    existing = adapter._voice_greetings.get(guild_id)
    if existing is not None and not existing.done():
        return
    owner = adapter._voice_owners[guild_id]
    profile = resolve_voice_profile()
    phrase = phrase_picker(adapter).choose("greetings", profile["phrases"]["greetings"])
    adapter._voice_greetings[guild_id] = asyncio.create_task(speak_phrase(
        adapter, str(adapter._voice_text_channels[guild_id]), phrase,
        allowed=lambda: adapter._voice_owners.get(guild_id) == owner
        and guild_id not in adapter._voice_owner_departing and adapter.is_in_voice_channel(guild_id)))


async def handle_voice_owner_state(adapter, member, before, after):
    guild_id = member.guild.id
    owner = adapter._voice_owners.get(guild_id)
    if owner is None or int(member.id) != owner[0] or before.channel == after.channel:
        return
    channel_id = after.channel.id if after.channel is not None else None
    if adapter._voice_owner_channels.get(guild_id) == channel_id:
        return
    adapter._voice_owner_channels[guild_id] = channel_id
    if after.channel is not None and after.channel.id == owner[1]:
        _greet_voice_owner(adapter, guild_id)
        return
    if before.channel is None or before.channel.id != owner[1]:
        return
    # Latch before yielding: queued captures and greeting renders lose authority now.
    adapter._voice_owner_departing.add(guild_id)
    chat_id = adapter._voice_text_channels.get(guild_id)
    adapter.discard_pending_voice_input(guild_id)
    greeting = adapter._voice_greetings.get(guild_id)
    if greeting is not None:
        greeting.cancel()
    if adapter._on_voice_disconnect and chat_id is not None:
        adapter._on_voice_disconnect(str(chat_id))
    try:
        await adapter.leave_voice_channel(guild_id)
    finally:
        adapter._voice_owner_departing.discard(guild_id)


async def handle_voice_state_update(adapter, member, before, after):
    """Keep auto-join, ownership and DAVE handling on the existing event path."""
    await adapter._handle_voice_owner_state(member, before, after)
    auto_join_channel = adapter._voice_auto_join_target(member, before, after)
    if auto_join_channel is not None:
        try:
            await adapter._voice_auto_join_handler(adapter, member, auto_join_channel)
        except Exception as e:
            logger.exception("[%s] voice auto-join handler failed: %s", adapter.name, e)
    bot_guild_ids = set(adapter._voice_clients.keys())
    if not bot_guild_ids:
        return
    guild_id = member.guild.id
    if guild_id not in bot_guild_ids:
        return
    if member == adapter._client.user:
        return
    joined = before.channel is None and after.channel is not None
    left = before.channel is not None and after.channel is None
    switched = (
        before.channel is not None
        and after.channel is not None
        and before.channel != after.channel
    )
    if joined or left or switched:
        logger.info(
            "Voice state: %s (%d) %s (guild %d)",
            member.display_name,
            member.id,
            "joined " + after.channel.name if joined
            else "left " + before.channel.name if left
            else f"moved {before.channel.name} -> {after.channel.name}",
            guild_id,
        )
        # Any membership change in the bot's channel bumps the
        # DAVE (E2EE) epoch — re-resolve the receiver's decryption
        # state so it never decodes against a stale session.
        vc = adapter._voice_clients.get(guild_id)
        receiver = adapter._voice_receivers.get(guild_id)
        if vc is not None and receiver is not None:
            bot_channel = getattr(vc, "channel", None)
            if bot_channel is not None and (
                before.channel == bot_channel
                or after.channel == bot_channel
            ):
                receiver.refresh_credentials("membership change")
