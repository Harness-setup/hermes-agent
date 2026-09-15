"""Tests for the in-process NeuTTS local provider in tools/tts_tool_local.py.

NeuTTS moved from a subprocess-per-call design (tools/neutts_synth.py, ~144-200s per call
measured live -- reloading the whole model stack every time) to the same in-process
warm/release lease pattern already used by Piper/KittenTTS, with the encoded reference
voice cached separately from the model itself so it's computed once, not per call.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_neutts_cache():
    from tools import tts_tool_local as _tt
    _tt._neutts_model_cache.clear()
    yield
    _tt._neutts_model_cache.clear()


@pytest.fixture
def mock_neutts_module():
    """Inject a fake neutts.NeuTTS + soundfile that return stub objects."""
    fake_instance = MagicMock()
    fake_instance.encode_reference.side_effect = lambda path: f"encoded::{path}"
    fake_instance.infer.return_value = [0.0] * 48000  # 2s of silence at 24kHz
    fake_cls = MagicMock(return_value=fake_instance)
    fake_neutts_module = MagicMock()
    fake_neutts_module.NeuTTS = fake_cls

    fake_sf = MagicMock()

    def _fake_write(path, audio, samplerate):
        import pathlib
        pathlib.Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_sf.write = _fake_write

    with patch.dict("sys.modules", {"neutts": fake_neutts_module, "soundfile": fake_sf}):
        yield fake_instance, fake_cls


class TestGenerateNeutts:
    def test_successful_wav_generation(self, tmp_path, mock_neutts_module):
        from tools.tts_tool_local import _generate_neutts

        fake_instance, fake_cls = mock_neutts_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_neutts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.assert_called_once()
        fake_instance.infer.assert_called_once()

    def test_model_loaded_once_across_multiple_calls(self, tmp_path, mock_neutts_module):
        """The whole point of this rewrite: repeat calls must not reload the model."""
        from tools.tts_tool_local import _generate_neutts

        fake_instance, fake_cls = mock_neutts_module
        _generate_neutts("First", str(tmp_path / "a.wav"), {})
        _generate_neutts("Second", str(tmp_path / "b.wav"), {})
        _generate_neutts("Third", str(tmp_path / "c.wav"), {})

        fake_cls.assert_called_once()
        assert fake_instance.infer.call_count == 3

    def test_reference_encoded_once_across_multiple_calls(self, tmp_path, mock_neutts_module):
        """encode_reference is the expensive (~48s live) per-reference step -- must be cached,
        not recomputed on every synthesis call."""
        from tools.tts_tool_local import _generate_neutts

        fake_instance, _ = mock_neutts_module
        _generate_neutts("First", str(tmp_path / "a.wav"), {})
        _generate_neutts("Second", str(tmp_path / "b.wav"), {})

        fake_instance.encode_reference.assert_called_once()

    def test_different_reference_voice_is_encoded_separately(self, tmp_path, mock_neutts_module):
        from tools.tts_tool_local import _generate_neutts

        ref_audio_a = tmp_path / "a.wav"
        ref_audio_a.write_bytes(b"fake")
        ref_text_a = tmp_path / "a.txt"
        ref_text_a.write_text("voice a")
        ref_audio_b = tmp_path / "b.wav"
        ref_audio_b.write_bytes(b"fake")
        ref_text_b = tmp_path / "b.txt"
        ref_text_b.write_text("voice b")

        fake_instance, _ = mock_neutts_module
        cfg_a = {"neutts": {"ref_audio": str(ref_audio_a), "ref_text": str(ref_text_a)}}
        cfg_b = {"neutts": {"ref_audio": str(ref_audio_b), "ref_text": str(ref_text_b)}}
        _generate_neutts("Hi", str(tmp_path / "out_a.wav"), cfg_a)
        _generate_neutts("Hi", str(tmp_path / "out_b.wav"), cfg_b)

        assert fake_instance.encode_reference.call_count == 2

    def test_config_passes_model_and_device(self, tmp_path, mock_neutts_module):
        from tools.tts_tool_local import _generate_neutts

        _, fake_cls = mock_neutts_module
        config = {"neutts": {"model": "some/other-model", "device": "cpu"}}
        _generate_neutts("Hi", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_cls.call_args.kwargs
        assert call_kwargs["backbone_repo"] == "some/other-model"
        assert call_kwargs["codec_repo"] == "neuphonic/neucodec"

    def test_cuda_device_maps_backbone_to_gpu_not_cuda(self, tmp_path, mock_neutts_module):
        """llama_cpp (backbone) only offloads on the literal string 'gpu'; torch (codec) only
        accepts 'cuda' -- a single device value can't satisfy both."""
        from tools.tts_tool_local import _generate_neutts

        _, fake_cls = mock_neutts_module
        config = {"neutts": {"device": "cuda"}}
        _generate_neutts("Hi", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_cls.call_args.kwargs
        assert call_kwargs["backbone_device"] == "gpu"
        assert call_kwargs["codec_device"] == "cuda"

    def test_missing_neutts_raises_import_error(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "neutts", None)
        from tools.tts_tool_local import _generate_neutts

        with pytest.raises((ImportError, TypeError)):
            _generate_neutts("Hi", str(tmp_path / "out.wav"), {})


class TestWarmNeutts:
    def test_warm_loads_model_and_encodes_reference(self, tmp_path, mock_neutts_module):
        """Warming must pay the FULL cold cost up front (model + reference encoding) so the
        first real synthesis call after warm-up only pays the per-utterance infer step."""
        from tools.tts_tool_lifecycle import _warm_neutts

        fake_instance, fake_cls = mock_neutts_module
        _warm_neutts({})

        fake_cls.assert_called_once()
        fake_instance.encode_reference.assert_called_once()
        fake_instance.infer.assert_not_called()

    def test_synthesis_after_warm_reuses_warmed_model(self, tmp_path, mock_neutts_module):
        from tools.tts_tool_lifecycle import _warm_neutts
        from tools.tts_tool_local import _generate_neutts

        fake_instance, fake_cls = mock_neutts_module
        _warm_neutts({})
        _generate_neutts("Hello", str(tmp_path / "out.wav"), {})

        fake_cls.assert_called_once()
        fake_instance.encode_reference.assert_called_once()
        fake_instance.infer.assert_called_once()


class TestReleaseNeutts:
    def test_release_clears_model_and_reference_cache(self, tmp_path, mock_neutts_module):
        """release_tts_provider must drop BOTH the model and its cached reference encoding --
        they live in the same cache entry specifically so one release clears both."""
        from tools.tts_tool_lifecycle import release_tts_provider
        from tools.tts_tool_local import _generate_neutts, _neutts_model_cache

        _generate_neutts("Hello", str(tmp_path / "out.wav"), {})
        assert len(_neutts_model_cache) == 1

        released = release_tts_provider("neutts")
        assert released["released"] == 1
        assert len(_neutts_model_cache) == 0
