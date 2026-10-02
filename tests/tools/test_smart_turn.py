"""Tests for tools/smart_turn.py -- the Smart Turn v3.1 semantic
turn-completion wrapper. No real model download in CI: onnxruntime and
transformers are mocked/faked.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.smart_turn import SMART_TURN_SAMPLE_RATE, _truncate_or_pad_to_window


class TestTruncateOrPadToWindow:
    def test_shorter_than_window_padded_at_start(self):
        audio = np.ones(1000, dtype=np.float32)
        windowed = _truncate_or_pad_to_window(audio)
        expected_len = 8 * SMART_TURN_SAMPLE_RATE
        assert len(windowed) == expected_len
        # padding (zeros) at the start, original content preserved at the end
        assert windowed[0] == 0.0
        assert windowed[-1] == 1.0
        assert np.array_equal(windowed[-1000:], audio)

    def test_longer_than_window_truncated_keeping_the_end(self):
        expected_len = 8 * SMART_TURN_SAMPLE_RATE
        audio = np.arange(expected_len + 500, dtype=np.float32)
        windowed = _truncate_or_pad_to_window(audio)
        assert len(windowed) == expected_len
        # keeps the LAST expected_len samples
        assert np.array_equal(windowed, audio[-expected_len:])

    def test_exactly_window_length_unchanged(self):
        expected_len = 8 * SMART_TURN_SAMPLE_RATE
        audio = np.arange(expected_len, dtype=np.float32)
        windowed = _truncate_or_pad_to_window(audio)
        assert np.array_equal(windowed, audio)


class TestTurnCompleteProbability:
    def test_applies_sigmoid_to_raw_logit_output(self):
        """The actual onnx-community/smart-turn-v3-ONNX int8 file returns a raw
        (pre-sigmoid) logit named "logits" -- verified live (2s of silence input
        produced a raw output of 3.57, not a [0,1] probability). Sigmoid must be
        applied here explicitly, not assumed already applied."""
        from tools.smart_turn import turn_complete_probability

        fake_session = MagicMock()
        raw_logit = 3.5665376
        fake_session.run.return_value = [np.array([[raw_logit]], dtype=np.float32)]

        fake_extractor = MagicMock()
        fake_extractor.return_value.input_features = np.zeros((1, 80, 400), dtype=np.float32)

        audio_int16 = np.array([0, 16384, -16384, 32767], dtype=np.int16)

        with patch("transformers.WhisperFeatureExtractor", return_value=fake_extractor):
            probability = turn_complete_probability(fake_session, audio_int16)

        expected = 1.0 / (1.0 + np.exp(-raw_logit))
        assert probability == pytest.approx(expected)
        assert 0.0 <= probability <= 1.0

    def test_zero_logit_maps_to_half_probability(self):
        from tools.smart_turn import turn_complete_probability

        fake_session = MagicMock()
        fake_session.run.return_value = [np.array([[0.0]], dtype=np.float32)]
        fake_extractor = MagicMock()
        fake_extractor.return_value.input_features = np.zeros((1, 80, 400), dtype=np.float32)

        with patch("transformers.WhisperFeatureExtractor", return_value=fake_extractor):
            probability = turn_complete_probability(fake_session, np.zeros(1600, dtype=np.int16))

        assert probability == pytest.approx(0.5)

    def test_session_run_called_with_input_features_key(self):
        from tools.smart_turn import turn_complete_probability

        fake_session = MagicMock()
        fake_session.run.return_value = [np.array([[0.1]], dtype=np.float32)]
        fake_extractor = MagicMock()
        fake_extractor.return_value.input_features = np.zeros((1, 80, 400), dtype=np.float32)

        with patch("transformers.WhisperFeatureExtractor", return_value=fake_extractor):
            turn_complete_probability(fake_session, np.zeros(1600, dtype=np.int16))

        call_args = fake_session.run.call_args
        assert call_args[0][0] is None
        assert "input_features" in call_args[0][1]


class TestLoadSmartTurnModel:
    def test_builds_session_with_recommended_cpu_options(self):
        from tools.smart_turn import load_smart_turn_model

        fake_ort_module = MagicMock()
        fake_session = MagicMock()
        fake_ort_module.InferenceSession.return_value = fake_session
        fake_ort_module.ExecutionMode.ORT_SEQUENTIAL = "ORT_SEQUENTIAL"
        fake_ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

        with patch.dict("sys.modules", {"onnxruntime": fake_ort_module}), \
             patch("huggingface_hub.hf_hub_download", return_value="/fake/model_int8.onnx"):
            result = load_smart_turn_model()

        assert result is fake_session
        session_options = fake_ort_module.SessionOptions.return_value
        assert session_options.execution_mode == "ORT_SEQUENTIAL"
        assert session_options.inter_op_num_threads == 1
        assert session_options.graph_optimization_level == "ORT_ENABLE_ALL"

    def test_raises_on_download_failure(self):
        from tools.smart_turn import load_smart_turn_model

        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("network down")):
            with pytest.raises(OSError):
                load_smart_turn_model()
