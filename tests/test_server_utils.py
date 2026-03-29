"""Additional tests for server utility functions and edge cases.

Tests for:
- format_timestamp utility
- segments_to_srt utility
- SRT response format
- Health endpoint
- _fixed_duration_chunks fallback
- verbose_json with multiple chunks
- json format with auto language (should omit language field)
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
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _apply_audio_mocks():
    """Start audio pipeline mocks. Returns dict of mocks and patches for cleanup."""
    audio_array = MagicMock()
    audio_array.__len__ = lambda s: 32000
    audio_array.ndim = 1
    audio_array.astype.return_value = audio_array
    audio_array.mean.return_value = audio_array

    sf_patch = patch("server.sf")
    chunks_patch = patch("server.split_audio_into_chunks")
    audio_seg_patch = patch("server.AudioSegment")

    sf_mock = sf_patch.start()
    chunks_mock = chunks_patch.start()
    audio_seg_mock = audio_seg_patch.start()

    sf_mock.read.return_value = (audio_array, 16000)

    # Mock AudioSegment and chunk slicing
    mock_seg = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.frame_rate = 16000
    mock_chunk.get_array_of_samples.return_value = []
    mock_seg.__getitem__ = lambda s, key: mock_chunk
    audio_seg_mock.from_file.return_value = mock_seg

    return {
        "patches": [sf_patch, chunks_patch, audio_seg_patch],
        "chunks": chunks_mock,
        "audio_array": audio_array,
    }


def _stop_mocks(mock_info):
    """Stop all patches."""
    for p in mock_info["patches"]:
        p.stop()


@pytest.fixture()
def mock_model_state():
    """Patch model_state to appear loaded."""
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
    from server import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self):
        from server import format_timestamp

        assert format_timestamp(0) == "00:00:00,000"

    def test_one_second(self):
        from server import format_timestamp

        assert format_timestamp(1000) == "00:00:01,000"

    def test_complex(self):
        from server import format_timestamp

        # 1h 23m 45s 678ms
        ms = (1 * 3600 + 23 * 60 + 45) * 1000 + 678
        assert format_timestamp(ms) == "01:23:45,678"

    def test_padding(self):
        from server import format_timestamp

        assert format_timestamp(5) == "00:00:00,005"
        assert format_timestamp(50) == "00:00:00,050"


# ---------------------------------------------------------------------------
# segments_to_srt
# ---------------------------------------------------------------------------


class TestSegmentsToSrt:
    def test_single_segment(self):
        from server import segments_to_srt

        segs = [{"start_ms": 0, "end_ms": 1500, "text": "Hello world"}]
        result = segments_to_srt(segs)
        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:01,500" in result
        assert "Hello world" in result

    def test_multiple_segments(self):
        from server import segments_to_srt

        segs = [
            {"start_ms": 0, "end_ms": 2000, "text": "Hello"},
            {"start_ms": 2000, "end_ms": 4000, "text": "World"},
        ]
        result = segments_to_srt(segs)
        assert "1\n" in result
        assert "2\n" in result
        assert "Hello" in result
        assert "World" in result

    def test_empty_segments(self):
        from server import segments_to_srt

        assert segments_to_srt([]) == ""


# ---------------------------------------------------------------------------
# SRT response format
# ---------------------------------------------------------------------------


class TestSrtResponseFormat:
    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_srt_format_returns_text(self, mock_transcribe, client):
        """SRT format should return plain text SRT content."""
        mi = _apply_audio_mocks()
        try:
            mock_transcribe.return_value = "Hello world"
            mi["chunks"].return_value = [(0, 2000)]
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "srt"},
            )
            assert resp.status_code == 200, resp.text
            assert "00:00:00,000" in resp.text
        finally:
            _stop_mocks(mi)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_healthy(self, client):
        """Health check with model loaded should return healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["status"] == "healthy"


# ---------------------------------------------------------------------------
# verbose_json with multiple chunks
# ---------------------------------------------------------------------------


class TestVerboseJsonMultipleChunks:
    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_verbose_json_multi_chunk_segments(self, mock_transcribe, client):
        """Multiple chunks should produce multiple segments with correct timestamps."""
        mi = _apply_audio_mocks()
        try:
            mock_transcribe.return_value = "Part one"
            mi["chunks"].return_value = [(0, 5000), (5000, 10000)]
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "verbose_json"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert len(body["segments"]) == 2
            # First chunk: 0-5000ms → 0.0-5.0s
            assert body["segments"][0]["start"] == 0.0
            assert body["segments"][0]["end"] == 5.0
            assert body["segments"][0]["text"] == "Part one"
            # Second chunk: 5000-10000ms → 5.0-10.0s
            assert body["segments"][1]["start"] == 5.0
            assert body["segments"][1]["end"] == 10.0
        finally:
            _stop_mocks(mi)


# ---------------------------------------------------------------------------
# json with auto language
# ---------------------------------------------------------------------------


class TestJsonAutoLanguage:
    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_json_auto_language_omitted(self, mock_transcribe, client):
        """json format with language=auto should not set language field."""
        mi = _apply_audio_mocks()
        try:
            mock_transcribe.return_value = "Hello"
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"response_format": "json"},
            )
            assert resp.status_code == 200
            body = resp.json()
            # language should be null/None when "auto"
            assert body.get("language") is None
        finally:
            _stop_mocks(mi)


# ---------------------------------------------------------------------------
# _fixed_duration_chunks
# ---------------------------------------------------------------------------


class TestFixedDurationChunks:
    def test_short_audio_single_chunk(self):
        from server import _fixed_duration_chunks

        chunks = _fixed_duration_chunks(5000)
        assert len(chunks) == 1
        assert chunks[0] == (0, 5000)

    def test_long_audio_multiple_chunks(self):
        from server import _fixed_duration_chunks, CHUNK_DURATION_MS, CHUNK_OVERLAP_MS

        duration = 30000  # 30s
        chunks = _fixed_duration_chunks(duration)
        # Each chunk is up to CHUNK_DURATION_MS, stepping by (CHUNK_DURATION_MS - CHUNK_OVERLAP_MS)
        assert len(chunks) > 1
        # First chunk should start at 0
        assert chunks[0][0] == 0
        # Last chunk end should equal duration
        assert chunks[-1][1] == duration

    def test_zero_duration(self):
        from server import _fixed_duration_chunks

        chunks = _fixed_duration_chunks(0)
        assert chunks == [(0, 0)]


# ---------------------------------------------------------------------------
# Stream + verbose_json rejected
# ---------------------------------------------------------------------------


class TestStreamFormatRestrictions:
    def test_stream_plus_verbose_json_rejected(self, client):
        """stream=True + response_format=verbose_json should return 400."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            data={"stream": "true", "response_format": "verbose_json"},
        )
        assert resp.status_code in (400, 422)

    def test_stream_plus_srt_rejected(self, client):
        """stream=True + response_format=srt should return 400."""
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            data={"stream": "true", "response_format": "srt"},
        )
        assert resp.status_code in (400, 422)
