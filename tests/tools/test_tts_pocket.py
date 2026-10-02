"""Tests for the in-process Pocket TTS local provider in tools/tts_tool_local.py.

Tony, 2026-09-16: measured live at ~1.2s model load + ~1.1s/sentence steady-state on CPU
alone -- 5-6x faster than NeuTTS/Chatterbox. The safetensors voice-embedding export/reuse
is the genuine "clone once (slower), reuse cheaply" design Tony originally asked for --
these tests exist specifically to pin that a ref_audio clone is exported exactly once and
a later call (even a fresh cache entry) reuses the cached .safetensors file instead of
re-processing the raw reference audio."""

from unittest.mock import MagicMock, patch

import pytest

# Imported at module (collection) level, not inside a test: some fixture/plugin in this test
# suite resets sys.modules to its pre-test snapshot between tests, and numpy's C-extension
# guard refuses to re-initialize its multiarray module a second time in the same process --
# a scipy import that first happens INSIDE a test (as _generate_pocket_tts's own `import
# scipy.io.wavfile` does) trips that on the very next test. Importing here once, before any
# test runs, keeps it in the baseline snapshot so it never gets swept away mid-suite.
import scipy.io.wavfile  # noqa: F401


@pytest.fixture(autouse=True)
def clear_pocket_tts_cache(tmp_path, monkeypatch):
    from tools import tts_tool_local as _tt
    _tt._pocket_tts_model_cache.clear()
    # Isolate the voice-cache directory per test so exported .safetensors files from one
    # test can't leak into another (the whole point of these tests is exact export-once
    # counting). Real code creates this dir itself (get_hermes_dir); the test double must too.
    cache_dir = tmp_path / "voice-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_tt, "_get_pocket_tts_voice_cache_dir", lambda: cache_dir)
    yield
    _tt._pocket_tts_model_cache.clear()


class _FakeAudioTensor:
    """Stands in for the torch tensor pocket_tts.generate_audio() really returns -- real
    scipy.io.wavfile.write() runs against this (not mocked: scipy/numpy are C-extension-backed
    and re-patching sys.modules around an already-imported numpy breaks its multiarray import),
    so .detach().cpu().numpy() must produce something scipy can actually write."""
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.zeros(2400, dtype=np.float32)  # 0.1s of silence at 24kHz


@pytest.fixture
def mock_pocket_tts_module():
    """Inject a fake pocket_tts.TTSModel + export_model_state (scipy.io.wavfile stays real)."""
    fake_model = MagicMock()
    fake_model.sample_rate = 24000
    fake_model.get_state_for_audio_prompt.side_effect = lambda src: f"state::{src}"
    fake_model.generate_audio.return_value = _FakeAudioTensor()
    fake_cls = MagicMock()
    fake_cls.load_model.return_value = fake_model

    fake_export = MagicMock(side_effect=lambda state, path: __import__("pathlib").Path(path).write_bytes(b"fake-tensor"))

    fake_module = MagicMock()
    fake_module.TTSModel = fake_cls
    fake_module.export_model_state = fake_export

    with patch.dict("sys.modules", {"pocket_tts": fake_module}):
        yield fake_model, fake_cls, fake_export


class TestGeneratePocketTTS:
    def test_successful_wav_generation_with_catalog_voice(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_pocket_tts("Hello world", output_path, {"pocket_tts": {"voice": "alba"}})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_model.get_state_for_audio_prompt.assert_called_once_with("alba")
        fake_model.generate_audio.assert_called_once()

    def test_model_loaded_once_across_multiple_calls(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        cfg = {"pocket_tts": {"voice": "alba"}}
        _generate_pocket_tts("First", str(tmp_path / "a.wav"), cfg)
        _generate_pocket_tts("Second", str(tmp_path / "b.wav"), cfg)

        fake_cls.load_model.assert_called_once()
        assert fake_model.generate_audio.call_count == 2

    def test_catalog_voice_state_resolved_once_and_cached(self, tmp_path, mock_pocket_tts_module):
        """Same voice across two calls must not re-resolve the voice state."""
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        cfg = {"pocket_tts": {"voice": "alba"}}
        _generate_pocket_tts("First", str(tmp_path / "a.wav"), cfg)
        _generate_pocket_tts("Second", str(tmp_path / "b.wav"), cfg)

        assert fake_model.get_state_for_audio_prompt.call_count == 1

    def test_ref_audio_clone_exports_a_safetensors_file_on_first_use(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        ref_audio = str(tmp_path / "my_voice.wav")
        (tmp_path / "my_voice.wav").write_bytes(b"fake audio")
        fake_model, fake_cls, fake_export = mock_pocket_tts_module

        _generate_pocket_tts("Hi", str(tmp_path / "out.wav"), {"pocket_tts": {"ref_audio": ref_audio}})

        fake_export.assert_called_once()
        # get_state_for_audio_prompt was called with the RAW ref_audio path (the slow,
        # one-time cloning cost), not a cached path, since nothing was cached yet.
        fake_model.get_state_for_audio_prompt.assert_called_once_with(ref_audio)

    def test_ref_audio_clone_reuses_cached_safetensors_on_a_later_cache_entry(self, tmp_path, mock_pocket_tts_module):
        """The whole point of this provider: a SECOND, independent voice-state resolution
        (e.g. a fresh in-memory cache after a restart) must load the cheap cached
        .safetensors file instead of re-processing the raw reference audio again."""
        from tools.tts_tool_local import _generate_pocket_tts, _pocket_tts_model_cache

        ref_audio = str(tmp_path / "my_voice.wav")
        (tmp_path / "my_voice.wav").write_bytes(b"fake audio")
        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        cfg = {"pocket_tts": {"ref_audio": ref_audio}}

        _generate_pocket_tts("Hi", str(tmp_path / "out1.wav"), cfg)
        # Simulate a fresh process/cache (the exported .safetensors file on disk survives,
        # only the in-memory voice_states dict is gone) by dropping just the model cache
        # entry's in-memory voice_states, not the exported file on disk.
        _pocket_tts_model_cache.clear()

        _generate_pocket_tts("Hi again", str(tmp_path / "out2.wav"), cfg)

        # First call resolved the raw ref_audio path; second call must resolve the CACHED
        # .safetensors path instead, never touching the raw audio file a second time.
        calls = [c.args[0] for c in fake_model.get_state_for_audio_prompt.call_args_list]
        assert calls[0] == ref_audio
        assert calls[1] != ref_audio
        assert calls[1].endswith(".safetensors")
        fake_export.assert_called_once()  # only exported once, not on the second (cached) call

    def test_default_ref_audio_falls_back_to_neutts_sample(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts, _NEUTTS_SAMPLES

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        _generate_pocket_tts("Hi", str(tmp_path / "out.wav"), {})

        called_with = fake_model.get_state_for_audio_prompt.call_args.args[0]
        assert called_with == str((_NEUTTS_SAMPLES / "jo.wav").expanduser())

    def test_device_other_than_cpu_moves_model(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        _generate_pocket_tts("Hi", str(tmp_path / "out.wav"), {"pocket_tts": {"device": "cuda", "voice": "alba"}})

        fake_model.to.assert_called_once_with("cuda")

    def test_cpu_device_does_not_call_to(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        _generate_pocket_tts("Hi", str(tmp_path / "out.wav"), {"pocket_tts": {"voice": "alba"}})

        fake_model.to.assert_not_called()

    def test_different_device_loads_a_separate_model_instance(self, tmp_path, mock_pocket_tts_module):
        from tools.tts_tool_local import _generate_pocket_tts

        fake_model, fake_cls, fake_export = mock_pocket_tts_module
        _generate_pocket_tts("A", str(tmp_path / "a.wav"), {"pocket_tts": {"voice": "alba", "device": "cpu"}})
        _generate_pocket_tts("B", str(tmp_path / "b.wav"), {"pocket_tts": {"voice": "alba", "device": "cuda"}})

        assert fake_cls.load_model.call_count == 2
