---
title: "Streaming TTS Internals"
description: "Sentence chunker, streaming provider ABC, capability matrix and how to add a streaming TTS provider"
---

# Streaming TTS

Hermes can stream TTS audio as it arrives from the provider, instead of waiting
for the full audio before playing. This is used by voice mode (CLI/TUI live
conversation), the dashboard speak-stream WebSocket, and — via the gateway
`StreamingTTSConsumer` — any platform adapter that opts into streaming audio.
Voice replies start speaking after the first clause instead of after full
generation + synthesis.

## Architecture

The streaming pipeline has four parts:

1. **Producer** — the LLM emits text deltas as it generates a response
2. **Sentence chunker** — `tools.tts_streaming.SentenceChunker` accumulates
   deltas, strips `<think>` blocks (even split across deltas), and flushes
   complete sentences
3. **TTS provider** — a registered `StreamingTTSProvider` turns each sentence
   into raw PCM chunks (int16 mono at the provider's declared `sample_rate`)
4. **Audio sink** — `sounddevice.OutputStream` for local playback
   (`tools.tts_tool_speaker.stream_tts_to_speaker`), or a gateway platform adapter's
   `write_streaming_tts` seam (`gateway/streaming_tts_consumer.py`)

Providers with no chunked API still get per-*sentence* playback via the proven
sync `text_to_speech_tool` path, so edge (the default) is conversational too.
All spoken text is cleaned by `tools.tts_text_normalize.prepare_spoken_text`
(one cleaner, all paths).

## How to pick a provider

By default the dispatcher streams with the provider you already configured
(`tts.provider`) when that provider has a chunked API — it never silently
swaps your voice for a different provider just to get streaming.

To override, set `tts.streaming.provider` in your `config.yaml`:

- a provider name (`elevenlabs`, `gemini`, `openai`, `xai`) pins that streamer
- `auto` walks the priority list `elevenlabs → gemini → openai → xai` and uses
  the first one whose credentials resolve — an explicit opt-in to "best
  chunked voice available"

```yaml
tts:
  provider: gemini
  streaming:
    provider: gemini      # or "auto"
    min_len: 20           # shortest first sentence (chars) spoken on its own; CJK setups use ~6
  gemini:
    model: gemini-2.5-flash-preview-tts
    voice: Kore
```

## Capability matrix

| Provider    | Transport                             | Chunked PCM | Credentials |
|-------------|---------------------------------------|-------------|-------------|
| elevenlabs  | chunked HTTP (`pcm_24000`)            | yes         | `ELEVENLABS_API_KEY` / `tts.elevenlabs` |
| openai      | chunked HTTP (`with_streaming_response`, `pcm`) | yes | `tts.openai.api_key` → env → managed gateway |
| gemini      | SSE (`streamGenerateContent?alt=sse`) | yes         | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| xai         | WebSocket (`wss://api.x.ai/v1/tts`)   | yes         | `XAI_API_KEY` preferred, else xAI OAuth (the subscription bearer 403s on metered TTS) |
| edge, piper, kitten, neutts, mistral, minimax, deepinfra, … | — | no (per-sentence sync fallback) | as usual |

All credential lookups go through `resolve_provider_secret()`
(config > env/.env > credential pool) — never bare env reads. Streamed bodies
are capped at 16 MiB per sentence, mirroring the sync providers' bounded
upstream-body invariant.

## Adding a new streaming provider

1. Subclass `StreamingTTSProvider` in `tools/tts_streaming.py`
2. Set `sample_rate` (and `channels` / `sample_width` if not int16 mono)
3. Implement `available()` (a pure probe — never install anything) and
   `stream(self, text) -> Iterator[bytes]` yielding raw PCM chunks
4. Decorate with `@register("yourname")`
5. Add tests in `tests/tools/test_tts_streaming.py`

The ABC enforces the contract; the registry makes the provider discoverable;
the dispatcher (`stream_tts_to_speaker`) and the gateway consumer handle the
sentence buffer, stop events, and audio sink for free.

## Gateway streaming (platform adapters)

`gateway/streaming_tts_consumer.py` bridges agent deltas to an adapter's
streaming-audio seam. Adapters opt in by overriding, on
`BasePlatformAdapter`:

- `supports_streaming_tts(chat_id, audio_format) -> bool`
- `begin_streaming_tts / write_streaming_tts / finish_streaming_tts /
  abort_streaming_tts`

All default to unsupported/no-op, so existing adapters are untouched. When a
turn's streaming audio completes, the whole-file auto-TTS reply for that turn
is suppressed (no double playback); when streaming fails before any audio was
audible, the gateway falls back to the legacy whole-file voice reply.


## Shared Jarvis voice profile

Authenticated clients can read `GET /api/audio/voice-profile` (with the same optional
`profile` query parameter as `/api/audio/speak`). The response contains `revision`,
non-secret `tts` voice metadata, `phrases`, and `timing`. Synthesis uses the configured
Hermes provider; this endpoint never returns provider credentials or commands.

`voice.profile.phrases` overrides named groups: `acknowledgements`, `fillers`,
`boot_fillers`, `stop`, `backend_error`, `stt_error`, `assistant_error`, and `stop_error`.
Each group is a nonempty array of strings. `voice.profile.timing` overrides
`filler_delay_seconds` (5), `filler_jitter_seconds` (0.5), `boot_delay_seconds` (2),
and `boot_jitter_seconds` (0.3). Values are finite nonnegative seconds.

The revision hashes the full TTS configuration, phrases, and timing. Jarvis refreshes
it on backend connection for each turn and uses it to select a separate phrase audio
cache. Its bundled catalog and local Piper remain available during backend outages.
Discord reads the same profile on each accepted voice turn, speaks one acknowledgement,
and schedules at most one delayed filler. First answer text, stop, and turn cleanup
cancel that filler, including in-flight synthesis and active playback. Tool starts
produce no additional acknowledgement. Fixed stop and transcription/assistant error
phrases require a connected voice channel and enabled voice replies. Silence and
hallucination-filtered transcriptions remain quiet. Text replies survive profile/TTS
failures. The natural spoken-answer note is attached to the current voice turn without
changing the cached system prompt or transcript wording.
