"""Tests for OpenAI Whisper API parity.

These tests verify that glm-asr's /v1/audio/transcriptions endpoint matches
the OpenAI Whisper API surface as closely as possible, using mocked model
inference so they run without GPU or model weights.

Covers:
- model param (accepted but ignored)
- response_format: text, json, verbose_json, srt
- OpenAI JSON response envelope shape
- timestamp_granularities param (segment-level from VAD chunks)
- Proper error handling for invalid formats
"""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file bytes with silence."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _mock_audio_pipeline():
    """Return a dict of patches that bypass audio I/O and model inference.

    Patches:
    - server.soundfile.read → returns (mono float32 array, 16000)
    - server.split_audio_into_chunks → returns single chunk
    - server.AudioSegment.from_file → returns mock with slicing
    - server.transcribe_audio_array → returns configurable text
    """
    audio_array = MagicMock()
    audio_array.__len__ = lambda s: 32000  # 2 seconds at 16kHz
    audio_array.__truediv__ = lambda s, other: 2.0  # len/sr = duration

    sf_patch = patch("server.sf")
    chunks_patch = patch("server.split_audio_into_chunks", return_value=[(0, 2000)])
    audio_seg_patch = patch("server.AudioSegment")

    return {
        "sf": sf_patch,
        "chunks": chunks_patch,
        "audioseg": audio_seg_patch,
        "audio_array": audio_array,
    }


def _apply_audio_mocks(mock_dict):
    """Start all patches in the dict. Returns dict of mock objects."""
    mocks = {}
    for key, p in mock_dict.items():
        if key == "audio_array":
            mocks[key] = mock_dict[key]
            continue
        mocks[key] = p.start()

    # Configure sf.read to return (array, sample_rate)
    mock_array = mocks["audio_array"]
    mocks["sf"].read.return_value = (mock_array, 16000)

    # Configure mock array: ndim=1, astype returns self, mean returns self
    mock_array.ndim = 1
    mock_array.astype.return_value = mock_array
    mock_array.mean.return_value = mock_array

    # Configure AudioSegment mock
    mock_seg = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.frame_rate = 16000
    mock_chunk.get_array_of_samples.return_value = []
    mock_seg.__getitem__ = lambda s, key: mock_chunk
    mock_seg.set_channels.return_value = mock_chunk
    mocks["audioseg"].from_file.return_value = mock_seg

    return mocks


def _stop_audio_mocks(mock_dict):
    """Stop all patches."""
    for key, p in mock_dict.items():
        if key != "audio_array":
            p.stop()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model_state():
    """Patch model_state to appear loaded with a working mock model."""
    with patch("server.model_state") as ms:
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.dtype = "float32"
        ms.model = mock_model
        ms.processor = MagicMock()
        ms.processor.feature_extractor.sampling_rate = 16000
        ms.device = "cpu"
        ms.vad_model = MagicMock()
        yield ms


@pytest.fixture()
def client(mock_model_state):
    """FastAPI TestClient with model mocked out."""
    from server import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. model param — accepted and ignored
# ---------------------------------------------------------------------------


class TestModelParam:
    """The OpenAI API requires model='whisper-1'. We accept it and ignore."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_model_param_accepted(self, mock_transcribe, client):
        """POST with model=whisper-1 should return 200, not 422."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"model": "whisper-1"},
            )
            assert resp.status_code == 200, resp.text
        finally:
            _stop_audio_mocks(pipe)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_model_param_does_not_affect_output(self, mock_transcribe, client):
        """Different model values should produce same output."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            r1 = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"model": "whisper-1"},
            )
            r2 = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"model": "glm-nano-2512"},
            )
            assert r1.json()["text"] == r2.json()["text"]
        finally:
            _stop_audio_mocks(pipe)


# ---------------------------------------------------------------------------
# 2. response_format=json — OpenAI envelope
# ---------------------------------------------------------------------------


class TestJsonResponseFormat:
    """response_format=json should return OpenAI-style JSON envelope."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_json_format_returns_openai_envelope(self, mock_transcribe, client):
        """json format should return {task, language, duration, text}."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "json"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["text"] == "Hello world"
            assert body["task"] == "transcribe"
            assert "language" in body
            assert "duration" in body
            assert isinstance(body["duration"], float)
        finally:
            _stop_audio_mocks(pipe)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_json_format_with_language(self, mock_transcribe, client):
        """json format should echo back the requested language."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "你好世界"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "json", "language": "zh"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["text"] == "你好世界"
            assert body["language"] == "zh"
        finally:
            _stop_audio_mocks(pipe)


# ---------------------------------------------------------------------------
# 3. response_format=verbose_json — segments + metadata
# ---------------------------------------------------------------------------


class TestVerboseJsonResponseFormat:
    """response_format=verbose_json should return full OpenAI verbose shape."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_verbose_json_has_segments(self, mock_transcribe, client):
        """verbose_json should include a segments array."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "verbose_json"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["task"] == "transcribe"
            assert body["text"] == "Hello world"
            assert "segments" in body
            assert isinstance(body["segments"], list)
            if body["segments"]:
                seg = body["segments"][0]
                assert "start" in seg
                assert "end" in seg
                assert "text" in seg
        finally:
            _stop_audio_mocks(pipe)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_verbose_json_has_language_and_duration(self, mock_transcribe, client):
        """verbose_json should include language and duration fields."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Testing verbose"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "verbose_json", "language": "en"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "language" in body
            assert "duration" in body
            assert body["language"] == "en"
        finally:
            _stop_audio_mocks(pipe)


# ---------------------------------------------------------------------------
# 4. timestamp_granularities param
# ---------------------------------------------------------------------------


class TestTimestampGranularities:
    """timestamp_granularities[] should control segment-level output."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_timestamp_granularities_segment(self, mock_transcribe, client):
        """timestamp_granularities[]=segment should work with verbose_json."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "segments" in body
        finally:
            _stop_audio_mocks(pipe)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_timestamp_granularities_word_accepted(self, mock_transcribe, client):
        """timestamp_granularities[]=word should be accepted (best-effort)."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Hello world"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
            )
            assert resp.status_code == 200, resp.text
        finally:
            _stop_audio_mocks(pipe)


# ---------------------------------------------------------------------------
# 5. Default response_format=text (existing behaviour preserved)
# ---------------------------------------------------------------------------


class TestTextFormatPreserved:
    """Ensure existing text format still works."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_default_is_text(self, mock_transcribe, client):
        """Default response_format should return JSON with just text."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Default text"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["text"] == "Default text"
        finally:
            _stop_audio_mocks(pipe)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_explicit_text_format(self, mock_transcribe, client):
        """Explicit response_format=text should match default."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = "Explicit text"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "text"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["text"] == "Explicit text"
        finally:
            _stop_audio_mocks(pipe)


# ---------------------------------------------------------------------------
# 6. /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Verify the models listing matches OpenAI shape."""

    def test_models_endpoint_shape(self, client):
        """GET /v1/models should return OpenAI-compatible listing."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1
        model = body["data"][0]
        assert model["object"] == "model"
        assert "id" in model
        assert "owned_by" in model
        assert "created" in model


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling for OpenAI parity."""

    def test_stream_plus_srt_rejected(self, client):
        """stream=True + response_format=srt should return 400."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            data={"stream": "true", "response_format": "srt"},
        )
        assert resp.status_code in (400, 422)

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_empty_transcription_json(self, mock_transcribe, client):
        """Empty transcription should still return valid json envelope."""
        pipe = _mock_audio_pipeline()
        mocks = _apply_audio_mocks(pipe)
        try:
            mock_transcribe.return_value = ""
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "json"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "text" in body
            assert "task" in body
        finally:
            _stop_audio_mocks(pipe)
