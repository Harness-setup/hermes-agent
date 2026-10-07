"""Per-turn guidance for the Jarvis voice app (``voice-chat`` surface): one natural answer, shown and spoken."""

VOICE_CHAT_NOTE = (
    "[Voice conversation: Reply naturally to the person speaking to you. "
    "Answer directly, then give the detail their question needs. "
    "Simple questions can be brief; explanations and troubleshooting can be detailed. "
    "Use clear sentences and conversational transitions. Preserve important details, "
    "qualifications, and uncertainty. Do not impose a sentence or word limit. "
    "Avoid unnecessary repetition, process narration, and routine offers to continue. "
    "Prefer plain prose. Include exact commands, code, or identifiers when requested "
    "or necessary, and explain their purpose conversationally. "
    "This is the single answer shown on screen and spoken aloud.]"
)


def voice_chat_turn_note() -> str:
    return VOICE_CHAT_NOTE
