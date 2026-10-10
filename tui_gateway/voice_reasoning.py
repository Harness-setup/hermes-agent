"""Reasoning policy for one desktop voice turn, including live-engine delegations."""

from contextlib import contextmanager


@contextmanager
def voice_turn_reasoning(session: dict, agent, voice_input: bool):
    """Keep the session choice authoritative and restore the cached agent after voice."""
    if not voice_input:
        yield
        return
    previous = agent.reasoning_config
    override = session.get("create_reasoning_override")
    agent.reasoning_config = override if override is not None else {"enabled": False}
    try:
        yield
    finally:
        # A reasoning control used mid-turn owns the new value. Never undo that pick.
        agent.reasoning_config = session.get("create_reasoning_override", previous)
