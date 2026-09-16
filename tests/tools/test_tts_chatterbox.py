"""Tests for the in-process Chatterbox local provider in tools/tts_tool_local.py.

Tony, 2026-09-16: measured live at 5-7s/utterance even warm on GPU (turbo) or CPU (nano) --
not the near-instant result hoped for, but built and tested as an available provider
regardless (quality comparison, not a default-provider decision)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_chatterbox_cache():
    from tools import tts_tool_local as _tt
    _tt._chatterbox_model_cache.clear()
    yield
    _tt._chatterbox_model_cache.clear()


@pytest.fixture
def mock_chatterbox_module():
    """Inject a fake chatterbox.tts_turbo.ChatterboxTurboTTS + torchaudio."""
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.return_value = "fake_wav_tensor"
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_model
    fake_tts_turbo_module = MagicMock()
    fake_tts_turbo_module.ChatterboxTurboTTS = fake_cls

    fake_ta = MagicMock()

    def _fake_save(path, wav, sr):
        import pathlib
        pathlib.Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_ta.save = _fake_save

    with patch.dict("sys.modules", {"chatterbox.tts_turbo": fake_tts_turbo_module, "torchaudio": fake_ta}):
        yield fake_model, fake_cls


class TestGenerateChatterbox:
    def test_successful_wav_generation_defaults_to_turbo_cuda(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_chatterbox("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.from_pretrained.assert_called_once_with(device="cuda", nano=False)
        fake_model.generate.assert_called_once()

    def test_nano_variant_defaults_to_cpu_and_nano_flag(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        output_path = str(tmp_path / "test.wav")
        _generate_chatterbox("Hi", output_path, {"chatterbox": {"variant": "nano"}})

        fake_cls.from_pretrained.assert_called_once_with(device="cpu", nano=True)

    def test_explicit_device_overrides_variant_default(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        output_path = str(tmp_path / "test.wav")
        _generate_chatterbox("Hi", output_path, {"chatterbox": {"variant": "turbo", "device": "cpu"}})

        fake_cls.from_pretrained.assert_called_once_with(device="cpu", nano=False)

    def test_model_loaded_once_across_multiple_calls_same_config(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        cfg = {"chatterbox": {"variant": "turbo", "device": "cuda"}}
        _generate_chatterbox("First", str(tmp_path / "a.wav"), cfg)
        _generate_chatterbox("Second", str(tmp_path / "b.wav"), cfg)

        fake_cls.from_pretrained.assert_called_once()
        assert fake_model.generate.call_count == 2

    def test_different_variant_loads_a_separate_model_instance(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        _generate_chatterbox("A", str(tmp_path / "a.wav"), {"chatterbox": {"variant": "turbo"}})
        _generate_chatterbox("B", str(tmp_path / "b.wav"), {"chatterbox": {"variant": "nano"}})

        assert fake_cls.from_pretrained.call_count == 2

    def test_ref_audio_passed_as_audio_prompt_path(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        _generate_chatterbox("Hi", str(tmp_path / "a.wav"), {"chatterbox": {"ref_audio": "/some/voice.wav"}})

        _, kwargs = fake_model.generate.call_args
        assert kwargs["audio_prompt_path"] == "/some/voice.wav"

    def test_exaggeration_and_cfg_weight_passed_through_when_configured(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        _generate_chatterbox(
            "Hi", str(tmp_path / "a.wav"),
            {"chatterbox": {"exaggeration": 0.7, "cfg_weight": 0.3}},
        )

        _, kwargs = fake_model.generate.call_args
        assert kwargs["exaggeration"] == 0.7
        assert kwargs["cfg_weight"] == 0.3

    def test_knobs_absent_from_call_when_not_configured(self, tmp_path, mock_chatterbox_module):
        from tools.tts_tool_local import _generate_chatterbox

        fake_model, fake_cls = mock_chatterbox_module
        _generate_chatterbox("Hi", str(tmp_path / "a.wav"), {})

        _, kwargs = fake_model.generate.call_args
        assert "exaggeration" not in kwargs
        assert "cfg_weight" not in kwargs
