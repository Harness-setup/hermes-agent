"""Shared non-secret spoken phrase profile for gateway and voice clients."""
import copy
import hashlib
import json
import math

DEFAULT_PHRASES = {'acknowledgements': ['Yeah?',
                      'Go ahead.',
                      'I’m listening.',
                      'What’s up?',
                      'Hey, what do you need?',
                      'I’m here.',
                      'Tell me.',
                      'Hey, go ahead.'],
 'fillers': ['Give me a moment.',
             'I need another moment.',
             'Hang on a little.',
             'I need a bit longer to answer.',
             'Just a little longer.',
             'I don’t have an answer yet.',
             'Bear with me.',
             'I’ll need a bit more time.',
             'One moment.',
             'I’m taking a little more time.',
             'Give me a little more time.',
             'I need another minute here.',
             'Hang tight for a moment.',
             'I’m not quite ready yet.',
             'I need a little longer.',
             'I’m taking a little longer.',
             'Thanks for waiting.',
             'I’m not ready to answer yet.',
             'Stay with me a moment.',
             'I’ll answer as soon as I’m ready.'],
 'boot_fillers': ['I’m starting up.',
                  'Give me a moment to get ready.',
                  'I’m getting connected.',
                  'I need a moment to start.',
                  'I’m getting ready to listen.',
                  'Hang on while I start up.',
                  'I’m still starting.',
                  'I need a little time to get ready.',
                  'I’m waiting to connect.',
                  'I’m getting things ready.',
                  'I’ll be ready in a moment.',
                  'Give me a little time to start.'],
 'long_wait': ['I still need a little more time.',
               'I need a little extra time.',
               'I’m taking longer than usual.',
               'I need more time before I can answer.',
               'Thanks for staying with me.',
               'I’m not ready to answer this yet.',
               'I’ll answer when I’m ready.',
               'I’m taking a bit longer with this.',
               'I need a little more time on this.',
               'I need more time for this one.',
               'I’ll speak as soon as I’m ready.',
               'I’m not ready with the answer yet.'],
 'greetings': ['Hey, I’m here.',
               'Hi, good to hear you.',
               'Hey, what’s up?',
               'Hi, I’m listening.',
               'Hey, go ahead.',
               'Hello, I’m here.',
               'Hey, what would you like to do?',
               'Hi, what do you need?'],
 'stop': ['I’ve stopped.', 'Okay, I stopped.', 'I’ve canceled that.', 'All right, I’ve stopped.'],
 'backend_error': ['I couldn’t reach the backend.',
                   'I can’t connect to the backend right now.',
                   'I couldn’t get a backend connection.',
                   'I’m unable to reach the backend right now.'],
 'stt_error': ['I couldn’t make that out.',
               'I didn’t catch what you said.',
               'I couldn’t transcribe that.',
               'I couldn’t hear that clearly enough.'],
 'assistant_error': ['I can’t answer right now.',
                     'I couldn’t finish that answer.',
                     'I’m having trouble answering right now.',
                     'I couldn’t answer that just now.'],
 'stop_error': ['I couldn’t stop the last request yet.',
                'I haven’t been able to cancel the last request.',
                'I couldn’t cancel that request yet.',
                'I’m still unable to stop the last request.'],
 'departure': ['I’m heading out.',
               'I’ll disconnect now.',
               'I’m leaving the channel.',
               'I’ll head out now.'],
 'approval': ['Should I proceed?',
              'Would you like me to proceed?',
              'Can I go ahead?',
              'Do you want me to continue?'],
 'pin_unreachable': ["I couldn't reopen your pinned session. Say 'start a new session' or 'follow "
                     "desktop'.",
                     "I can't reopen the pinned conversation. Say 'start a new session' or 'follow "
                     "desktop'.",
                     "I couldn't get back into your pinned session. Say 'start a new session' or "
                     "'follow desktop'.",
                     "I can't access your pinned session right now. Say 'start a new session' or "
                     "'follow desktop'."],
 'session_owned': ["I can't use that conversation while it's open in Hermes Desktop. Say 'start a "
                   "new session' to use a separate one.",
                   'I need a separate conversation while that one is open in Hermes Desktop. Say '
                   "'start a new session'.",
                   "I can't join that conversation because Hermes Desktop has it open. Say 'start "
                   "a new session' for a separate one.",
                   "I can't submit there while Hermes Desktop has it open. Say 'start a new "
                   "session' to use a separate conversation."],
 'pin_missing': ["I couldn't find your pinned session, so I started a new one.",
                 "I've started a new session because your pinned one is unavailable.",
                 "I couldn't access the pinned conversation, so I've started a new one.",
                 "I've opened a new session because the pinned one is no longer available."],
 'session_reopen_error': ["I couldn't reopen that conversation.",
                          "I can't reopen that session right now.",
                          "I couldn't get back into that conversation.",
                          "I couldn't access that session."],
 'session_create_error': ["I couldn't start a new session.",
                          "I couldn't create a new conversation.",
                          "I wasn't able to open a new session.",
                          "I couldn't get a new conversation started."],
 'session_created': ["I've started a new session.",
                     "I've opened a new conversation.",
                     "I've created a fresh session.",
                     "I'm ready in a new session."],
 'desktop_missing': ["I couldn't find a desktop session to join.",
                     "I couldn't find a desktop conversation I can use.",
                     "I don't see an available desktop session.",
                     "I couldn't locate a desktop session to connect to."],
 'desktop_joined': ["I've joined your desktop session.",
                    "I'm connected to your desktop conversation.",
                    "I've switched to your desktop session.",
                    "I'm using your desktop session now."],
 'session_choice_missing': ["I don't have that many sessions to choose from.",
                            "I can't find a session at that number.",
                            "I don't see that number in the session list.",
                            "I can't select that number from these sessions."],
 'session_joined': ["I've joined that session.",
                    "I've switched to that conversation.",
                    "I'm using that session now.",
                    "I'm connected to that conversation."],
 'session_join_error': ["I couldn't join that session.",
                        "I couldn't connect to that conversation.",
                        "I couldn't switch to that session.",
                        "I wasn't able to open that conversation."],
 'session_empty': ["I couldn't find any recent sessions.",
                   "I don't see any recent conversations.",
                   "I haven't found a recent session to list.",
                   "I don't have any recent sessions to show."],
 'session_list': ['I found these sessions: {sessions}. Which one would you like?',
                  'I can offer these sessions: {sessions}. Which one should I use?',
                  'I have these sessions available: {sessions}. Which one do you want?',
                  'I found these conversations: {sessions}. Which should I join?']}
DEFAULT_TIMING = {'filler_delay_seconds': 5.0,
 'filler_jitter_seconds': 0.0,
 'boot_delay_seconds': 5.0,
 'boot_jitter_seconds': 0.0,
 'second_reminder_seconds': 15.0,
 'third_reminder_seconds': 30.0,
 'status_silence_seconds': 6.0}

# Explicit allowlist: arbitrary provider configuration may contain credentials or commands.
_PUBLIC_TTS_FIELDS = {"voice", "voice_id", "model", "speed", "language", "lang", "speaker", "speaker_id", "length_scale"}


def resolve_voice_profile():
    from hermes_cli.config import load_config
    from tools.tts_tool import _load_tts_config, DEFAULT_PROVIDER
    config = load_config()
    tts = _load_tts_config()
    overrides = (config.get("voice") or {}).get("profile") or {}
    phrases = copy.deepcopy(DEFAULT_PHRASES)
    for group, values in (overrides.get("phrases") or {}).items():
        if group in phrases and isinstance(values, list) and values and all(isinstance(v, str) and v.strip() for v in values):
            phrases[group] = [v.strip() for v in values]
    timing = dict(DEFAULT_TIMING)
    for key, value in (overrides.get("timing") or {}).items():
        if key in timing and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            timing[key] = value
    provider = str(tts.get("provider") or DEFAULT_PROVIDER)
    settings = tts.get(provider) or {}
    public_tts = {"provider": provider}
    for key, value in settings.items():
        if key in _PUBLIC_TTS_FIELDS and isinstance(value, (str, int, float, bool)):
            public_tts[key] = value
    if isinstance(tts.get("speed"), (str, int, float)):
        public_tts["speed"] = tts["speed"]
    # Hash the full synthesis configuration privately, so local model/sample changes also
    # invalidate audio. The response never contains those paths, commands or secrets.
    revision = hashlib.sha256(json.dumps([tts, phrases, timing], sort_keys=True, default=str).encode()).hexdigest()
    return {"revision": revision, "tts": public_tts, "phrases": phrases, "timing": timing}


class PhrasePicker:
    """Keep a shuffled bag per category, including across consecutive turns."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._bags = {}
        self._last = {}

    def choose(self, group, values):
        import random
        with self._lock:
            signature = tuple(dict.fromkeys(values))
            previous, bag = self._bags.get(group, (None, []))
            if previous != signature or not bag:
                bag = list(signature)
                random.shuffle(bag)
                if len(bag) > 1 and bag[-1] == self._last.get(group):
                    bag[0], bag[-1] = bag[-1], bag[0]
                self._bags[group] = (signature, bag)
            result = bag.pop()
            self._last[group] = result
            return result
