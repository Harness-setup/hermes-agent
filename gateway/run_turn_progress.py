"""Route gateway progress into Discord detail threads or the current stream."""
import logging
from gateway.config import Platform

logger = logging.getLogger("gateway.run")


def send_to_tool_thread(turn, text: str) -> bool:
    """Discord's discord.thread_tool_calls opt-in: tool-call progress lines AND reasoning
    (_thinking, routed here too as of 2026-09-23 -- see progress_callback's own comment) go to
    a lazily-created thread instead of the shared progress_queue, so the main channel only
    ever shows the final answer. Returns True when handled (caller must not also queue/
    native-render it); False falls through to the existing behavior unchanged (every other
    platform, or Discord with the flag off, is untouched by this)."""
    ctx = turn._ctx
    if ctx.source.platform != Platform.DISCORD:
        return False
    try:
        adapter = turn._runner._delivery_adapter_for(ctx.source)
    except Exception:
        logger.debug("Discord progress routing unavailable", exc_info=True)
        return False
    if adapter is None or not hasattr(adapter, "send_tool_progress_line"):
        return False
    try:
        if not adapter._discord_thread_tool_calls_enabled():
            return False
    except Exception:
        logger.debug("Discord progress routing unavailable", exc_info=True)
        return False
    turn._schedule(
        adapter.send_tool_progress_line(ctx.source.chat_id, turn._tool_progress_message_id, text),
        "discord tool-progress thread send error",
    )
    return True

def emit_progress(turn, msg: str) -> None:
    """Dedup consecutive identical lines (execute_code boilerplate), then route to the native
    stream bubble when the consumer accepts tool progress, else the progress queue -- unless
    Discord's tool-call thread split claims it first (see _send_to_tool_thread)."""
    ctx = turn._ctx
    sc = turn._stream_consumer()
    native = sc is not None and getattr(sc, "accepts_tool_progress", False)
    if msg == ctx.last_progress_msg[0]:
        ctx.repeat_count[0] += 1
        rendered = f"{msg} (×{ctx.repeat_count[0] + 1})"
        if turn._send_to_tool_thread(rendered):
            return
        if native:
            sc.on_tool_progress(rendered)
        else:
            ctx.progress_queue.put(("__dedup__", msg, ctx.repeat_count[0]))
        return
    ctx.last_progress_msg[0], ctx.repeat_count[0] = msg, 0
    if turn._send_to_tool_thread(msg):
        return
    if native:
        sc.on_tool_progress(msg)
    else:
        ctx.progress_queue.put(msg)
