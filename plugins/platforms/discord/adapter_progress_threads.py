"""Serialize progress delivery with cleanup so stopped turns cannot reopen threads."""
import asyncio
import logging
from typing import Any, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)


def _lock(adapter, key):
    return adapter._tool_progress_locks.setdefault(key, asyncio.Lock())

async def get_or_create(adapter, chat_id, message_id):
    if not message_id:
        return None
    key = str(message_id)
    async with _lock(adapter, key):
        if key in adapter._closed_tool_progress_turns:
            return None
        return await _create(adapter, chat_id, key)

async def send(adapter, chat_id, message_id, text):
    if not message_id:
        return
    key = str(message_id)
    try:
        async with _lock(adapter, key):
            if key in adapter._closed_tool_progress_turns:
                return
            thread = await _create(adapter, chat_id, key)
            if thread is None:
                return
            formatted = adapter.format_message(text)
            for chunk in adapter._cap_split_chunks(adapter.truncate_message(formatted, adapter.MAX_MESSAGE_LENGTH)):
                await thread.send(content=chunk)
    except Exception:
        logger.warning("Discord tool-progress delivery failed for %s", key, exc_info=True)

async def close(adapter, message_id):
    if not message_id:
        return
    key = str(message_id)
    # Latch before waiting for an in-flight create/send, so queued callbacks are rejected.
    adapter._closed_tool_progress_turns[key] = None
    async with _lock(adapter, key):
        await _archive(adapter, key)
    # Bound retired turns and their locks together; only completed turns are evicted.
    while len(adapter._closed_tool_progress_turns) > 256:
        retired = next(iter(adapter._closed_tool_progress_turns))
        del adapter._closed_tool_progress_turns[retired]
        adapter._tool_progress_locks.pop(retired, None)

async def _create(adapter, chat_id: str, event_message_id: Optional[str]) -> Optional[Any]:
    """Lazily create (once per turn, cached by event_message_id) the thread tool-call progress
    lines land in. Mirrors _auto_create_thread's retry/fallback shape, but starts from a fetched
    message (chat_id + event_message_id) rather than a live gateway event's Message object,
    since the progress pipeline only carries IDs, not the discord.py object."""
    from plugins.platforms.discord.adapter import discord
    if not event_message_id:
        return None
    cached = adapter._tool_progress_threads.get(event_message_id)
    if cached is not None:
        return cached
    try:
        channel = await adapter._resolve_channel(chat_id)
        if channel is None:
            return None
        # Tony, 2026-09-28: "does not even show threads anymore" -- root cause was a real
        # Discord 400 (error code 50024, "Cannot execute action on this channel type"): he
        # was chatting INSIDE an existing thread, and Discord has no nested threads, so
        # seed_msg.create_thread() always failed. When the source channel already IS a
        # thread, use it as-is -- no fetch_message, no create_thread, nothing that can 50024.
        if isinstance(channel, discord.Thread):
            adapter._tool_progress_threads[event_message_id] = channel
            return channel
        seed_msg = await channel.fetch_message(int(event_message_id))
        existing_thread = getattr(seed_msg, "thread", None)
        if existing_thread is not None:
            if existing_thread.archived:
                await existing_thread.edit(archived=False)
            adapter._tool_progress_threads[event_message_id] = existing_thread
            return existing_thread
    except Exception as e:
        logger.debug("[%s] tool-progress thread: couldn't fetch seed message %s: %s", adapter.name, event_message_id, e, exc_info=True)
        return None
    thread_name = f"\U0001f6e0️ {adapter._derive_auto_thread_name(seed_msg.content or '')}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            thread = await seed_msg.create_thread(name=thread_name, auto_archive_duration=60)
            adapter._tool_progress_threads[event_message_id] = thread
            return thread
        except Exception as e:
            logger.debug("Discord thread creation attempt failed", exc_info=True)
            last_error = e
            if attempt == 0:
                await asyncio.sleep(0.75)
    logger.warning("[%s] tool-progress thread creation failed: %s", adapter.name, last_error)
    return None


async def _archive(adapter, event_message_id: Optional[str]) -> None:
    """Remove human membership and archive the completed detail thread."""
    if not event_message_id:
        return
    thread = adapter._tool_progress_threads.pop(str(event_message_id), None)
    if thread is None:
        logger.debug("[%s] tool-progress thread archive: no cached thread for message %s",
                     adapter.name, event_message_id)
        return
    removed = 0
    try:
        members = await thread.fetch_members()
        bot_id = getattr(getattr(adapter._client, "user", None), "id", None)
        for member in members:
            if bot_id is not None and member.id == bot_id:
                continue
            try:
                await thread.remove_user(member)
                removed += 1
            except Exception as e:
                logger.warning("[%s] tool-progress thread member removal failed for %s: %s",
                               adapter.name, member.id, e, exc_info=True)
    except Exception as e:
        logger.warning("[%s] tool-progress thread member fetch failed: %s", adapter.name, e, exc_info=True)
    try:
        await thread.edit(archived=True)
        logger.info("[%s] tool-progress thread %s archived (%d member(s) removed)",
                    adapter.name, getattr(thread, "id", event_message_id), removed)
    except Exception as e:
        logger.warning("[%s] tool-progress thread archive failed: %s", adapter.name, e, exc_info=True)
