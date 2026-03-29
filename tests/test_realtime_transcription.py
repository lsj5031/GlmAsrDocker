"""Tests for the OpenAI Realtime API compatible transcription WebSocket endpoint.

These tests verify that the /v1/realtime endpoint matches the OpenAI Realtime
transcription API surface as closely as possible, using mocked model inference.

Covers:
- WebSocket connection and session.created event
- transcription_session.update client event
- input_audio_buffer.append with Base64 PCM16 audio
- input_audio_buffer.commit for manual buffer commit
- input_audio_buffer.clear
- Server event shapes: committed, delta, completed, cleared
- Error handling for invalid JSON
- Session config fields: format, language, turn_detection, noise_reduction, include
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use a simple list-based fake array that supports len()
class FakeArray:
    """Minimal array-like object for testing with proper len() support."""
    def __init__(self, data, sample_count=None):
        self._data = data
        self._sample_count = sample_count if sample_count is not None else len(data)
    def __len__(self):
        return self._sample_count
    def __truediv__(self, other):
        return self  # keep as array — we don't need real math in tests
    def __rtruediv__(self, other):
        return self
    def astype(self, dtype):
        return self  # just return self — the actual data doesn't matter in tests
    def __iter__(self):
        return iter(self._data)


def _make_pcm16_samples(duration_s: float = 1.0, sample_rate: int = 24000) -> bytes:
    """Generate PCM 16-bit silence bytes."""
    n_samples = int(sample_rate * duration_s)
    return b"\x00\x00" * n_samples


def _encode_pcm16_base64(duration_s: float = 1.0, sample_rate: int = 24000) -> str:
    """Generate Base64-encoded PCM16 audio."""
    raw = _make_pcm16_samples(duration_s, sample_rate)
    return base64.b64encode(raw).decode("utf-8")


def _make_event(event_type: str, **kwargs) -> str:
    """Build a client-sent JSON event string."""
    return json.dumps({"type": event_type, **kwargs})


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
def mock_numpy_realtime():
    """Patch numpy operations used by the realtime session to use FakeArray.

    This ensures that np.frombuffer, np.concatenate, and array operations
    produce objects with proper __len__ support.
    """
    import numpy as real_np

    def fake_frombuffer(buf, dtype=None):
        n = len(buf) // 2  # int16 = 2 bytes per sample
        return FakeArray([0.0] * n)

    def fake_concatenate(arrays):
        total = []
        for a in arrays:
            if isinstance(a, FakeArray):
                total.extend(a._data)
            elif hasattr(a, '_data'):
                total.extend(a._data)
            else:
                total.extend([0.0] * 4800)  # fallback: 0.2s at 24kHz
        return FakeArray(total)

    with patch("server.np.frombuffer", side_effect=fake_frombuffer) as mock_fb, \
         patch("server.np.concatenate", side_effect=fake_concatenate) as mock_cat:
        yield {"frombuffer": mock_fb, "concatenate": mock_cat}


@pytest.fixture()
def client(mock_model_state):
    """FastAPI TestClient with model mocked out."""
    from server import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Connection + session.created
# ---------------------------------------------------------------------------


class TestRealtimeConnection:
    """WebSocket handshake and initial session.created event."""

    def test_connect_receives_session_created(self, client):
        """Connecting to /v1/realtime should immediately send transcription_session.created."""
        with client.websocket_connect("/v1/realtime") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "transcription_session.created"
            assert "session" in msg
            session = msg["session"]
            assert session["object"] == "realtime.transcription_session"
            assert session["type"] == "transcription"
            assert "id" in session
            assert "input_audio_format" in session
            assert "input_audio_transcription" in session
            assert "turn_detection" in session

    def test_connect_with_intent_param(self, client):
        """Connecting with ?intent=transcription should work."""
        with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "transcription_session.created"

    def test_session_created_has_event_id(self, client):
        """All server events should include event_id."""
        with client.websocket_connect("/v1/realtime") as ws:
            msg = json.loads(ws.receive_text())
            assert "event_id" in msg
            assert msg["event_id"].startswith("event_")


# ---------------------------------------------------------------------------
# 2. transcription_session.update
# ---------------------------------------------------------------------------


class TestSessionUpdate:
    """Client can update session config via transcription_session.update."""

    def test_update_returns_session_updated(self, client):
        """transcription_session.update should respond with transcription_session.updated."""
        with client.websocket_connect("/v1/realtime") as ws:
            # Consume session.created
            ws.receive_text()

            ws.send_text(_make_event(
                "transcription_session.update",
                input_audio_format="pcm16",
                input_audio_transcription={
                    "model": "glm-nano-2512",
                    "language": "en",
                    "prompt": "test prompt",
                },
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.6,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 700,
                },
            ))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "transcription_session.updated"
            session = msg["session"]
            assert session["input_audio_format"] == "pcm16"
            assert session["input_audio_transcription"]["language"] == "en"
            assert session["turn_detection"]["type"] == "server_vad"
            assert session["turn_detection"]["threshold"] == 0.6
            assert session["turn_detection"]["prefix_padding_ms"] == 400
            assert session["turn_detection"]["silence_duration_ms"] == 700

    def test_update_disable_turn_detection(self, client):
        """Setting turn_detection=null disables VAD."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()  # session.created

            ws.send_text(_make_event(
                "transcription_session.update",
                turn_detection=None,
            ))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "transcription_session.updated"
            assert msg["session"]["turn_detection"] is None

    def test_update_noise_reduction(self, client):
        """Noise reduction can be configured."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event(
                "transcription_session.update",
                input_audio_noise_reduction={"type": "far_field"},
            ))
            msg = json.loads(ws.receive_text())
            session = msg["session"]
            assert session["input_audio_noise_reduction"]["type"] == "far_field"

    def test_update_noise_reduction_disabled(self, client):
        """Setting noise_reduction=null disables it."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event(
                "transcription_session.update",
                input_audio_noise_reduction=None,
            ))
            msg = json.loads(ws.receive_text())
            assert msg["session"]["input_audio_noise_reduction"] is None

    def test_update_include_logprobs(self, client):
        """include array can request logprobs."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event(
                "transcription_session.update",
                include=["item.input_audio_transcription.logprobs"],
            ))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "transcription_session.updated"
            assert "item.input_audio_transcription.logprobs" in msg["session"]["include"]


# ---------------------------------------------------------------------------
# 3. input_audio_buffer.commit (manual mode)
# ---------------------------------------------------------------------------


class TestManualBufferCommit:
    """Manual commit mode (turn_detection=null) transcribes on demand."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_commit_returns_committed_and_transcription(self, mock_transcribe, client, mock_numpy_realtime):
        """Commit should send committed event then transcription delta + completed."""
        mock_transcribe.return_value = "Hello world"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()  # session.created

            # Disable VAD for manual mode
            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()  # session.updated

            # Send some audio
            audio_b64 = _encode_pcm16_base64(duration_s=0.5, sample_rate=24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio_b64))

            # Commit
            ws.send_text(_make_event("input_audio_buffer.commit"))

            # Read events: committed → delta → completed
            committed = json.loads(ws.receive_text())
            assert committed["type"] == "input_audio_buffer.committed"
            assert "item_id" in committed
            assert "previous_item_id" in committed

            delta = json.loads(ws.receive_text())
            assert delta["type"] == "transcript.text.delta"
            assert delta["delta"] == "Hello world"
            assert delta["content_index"] == 0

            completed = json.loads(ws.receive_text())
            assert completed["type"] == "transcript.text.done"
            assert completed["text"] == "Hello world"
            assert "item_id" in completed

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_commit_empty_buffer_sends_nothing(self, mock_transcribe, client, mock_numpy_realtime):
        """Commit on empty buffer should not trigger transcription."""
        mock_transcribe.return_value = "unused"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()  # session.created

            # Disable VAD
            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            # Commit without any audio
            ws.send_text(_make_event("input_audio_buffer.commit"))

            # No events should be sent (next thing we send will be our own close)
            ws.send_text(_make_event("input_audio_buffer.clear"))

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_commit_preserves_item_ordering(self, mock_transcribe, client, mock_numpy_realtime):
        """Sequential commits should link via previous_item_id."""
        mock_transcribe.side_effect = ["First utterance", "Second utterance"]

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            # First commit
            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            committed1 = json.loads(ws.receive_text())
            item_id_1 = committed1["item_id"]
            ws.receive_text()  # delta
            ws.receive_text()  # completed

            # Second commit
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            committed2 = json.loads(ws.receive_text())
            assert committed2["previous_item_id"] == item_id_1


# ---------------------------------------------------------------------------
# 4. input_audio_buffer.clear
# ---------------------------------------------------------------------------


class TestBufferClear:
    """input_audio_buffer.clear should clear buffer and confirm."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_clear_sends_cleared_event(self, mock_transcribe, client, mock_numpy_realtime):
        """Clear should respond with input_audio_buffer.cleared."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            # Add audio then clear
            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.clear"))

            cleared = json.loads(ws.receive_text())
            assert cleared["type"] == "input_audio_buffer.cleared"

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_clear_then_commit_no_transcription(self, mock_transcribe, client, mock_numpy_realtime):
        """After clearing, commit should not produce transcription."""
        mock_transcribe.return_value = "should not appear"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.clear"))
            ws.receive_text()  # cleared

            ws.send_text(_make_event("input_audio_buffer.commit"))
            # Nothing should come back — we'll verify by sending another clear
            ws.send_text(_make_event("input_audio_buffer.clear"))
            cleared2 = json.loads(ws.receive_text())
            assert cleared2["type"] == "input_audio_buffer.cleared"


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Invalid input should produce error events, not crash."""

    def test_invalid_json_sends_error(self, client):
        """Non-JSON text should produce an error event."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()  # session.created

            ws.send_text("not json")
            error = json.loads(ws.receive_text())
            assert error["type"] == "error"
            assert "error" in error

    def test_unknown_event_type_ignored(self, client):
        """Unknown event types should not crash the server."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("some.unknown.event"))
            # No response expected — we verify by sending a valid event
            ws.send_text(_make_event("input_audio_buffer.clear"))
            cleared = json.loads(ws.receive_text())
            assert cleared["type"] == "input_audio_buffer.cleared"


# ---------------------------------------------------------------------------
# 6. Event shape validation
# ---------------------------------------------------------------------------


class TestEventShapes:
    """Verify server events match OpenAI Realtime API event shapes."""

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_committed_event_shape(self, mock_transcribe, client, mock_numpy_realtime):
        """input_audio_buffer.committed should match OpenAI shape."""
        mock_transcribe.return_value = "test"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            msg = json.loads(ws.receive_text())
            assert msg["type"] == "input_audio_buffer.committed"
            assert "event_id" in msg
            assert "item_id" in msg
            # previous_item_id should be null on first commit
            assert msg["previous_item_id"] is None

            # Consume remaining events
            ws.receive_text()  # delta
            ws.receive_text()  # completed

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_delta_event_shape(self, mock_transcribe, client, mock_numpy_realtime):
        """delta event should match OpenAI shape."""
        mock_transcribe.return_value = "Transcribed text"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            ws.receive_text()  # committed

            delta = json.loads(ws.receive_text())
            assert delta["type"] == "transcript.text.delta"
            assert "event_id" in delta
            assert "item_id" in delta
            assert delta["content_index"] == 0
            assert isinstance(delta["delta"], str)

            ws.receive_text()  # completed

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_completed_event_shape(self, mock_transcribe, client, mock_numpy_realtime):
        """completed event should match OpenAI shape."""
        mock_transcribe.return_value = "Final text"

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            ws.receive_text()  # committed
            ws.receive_text()  # delta

            completed = json.loads(ws.receive_text())
            assert completed["type"] == "transcript.text.done"
            assert "event_id" in completed
            assert "item_id" in completed
            assert completed["content_index"] == 0
            assert completed["text"] == "Final text"

    @patch("server.transcribe_audio_array", new_callable=AsyncMock)
    def test_transcription_error_returns_empty_completed(self, mock_transcribe, client, mock_numpy_realtime):
        """When transcription fails, should still send completed with empty transcript."""
        mock_transcribe.side_effect = RuntimeError("GPU OOM")

        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event("transcription_session.update", turn_detection=None))
            ws.receive_text()

            audio = _encode_pcm16_base64(0.3, 24000)
            ws.send_text(_make_event("input_audio_buffer.append", audio=audio))
            ws.send_text(_make_event("input_audio_buffer.commit"))

            ws.receive_text()  # committed

            completed = json.loads(ws.receive_text())
            assert completed["type"] == "transcript.text.done"
            assert completed["text"] == ""


# ---------------------------------------------------------------------------
# 7. Session dict validation
# ---------------------------------------------------------------------------


class TestSessionDict:
    """Verify the session configuration dict matches OpenAI shape."""

    def test_default_session_dict(self, client):
        """Default session should have all expected fields."""
        with client.websocket_connect("/v1/realtime") as ws:
            msg = json.loads(ws.receive_text())
            session = msg["session"]

            assert session["object"] == "realtime.transcription_session"
            assert session["type"] == "transcription"
            assert session["input_audio_format"] == "pcm16"
            assert session["input_audio_transcription"]["model"] == "glm-nano-2512"
            assert session["turn_detection"]["type"] == "server_vad"
            assert session["input_audio_noise_reduction"] is not None

    def test_updated_session_dict(self, client):
        """After update, session dict should reflect changes."""
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()

            ws.send_text(_make_event(
                "transcription_session.update",
                input_audio_transcription={"language": "zh", "prompt": "test"},
                turn_detection=None,
                input_audio_noise_reduction=None,
            ))
            msg = json.loads(ws.receive_text())
            session = msg["session"]

            assert session["input_audio_transcription"]["language"] == "zh"
            assert session["input_audio_transcription"]["prompt"] == "test"
            assert session["turn_detection"] is None
            assert session["input_audio_noise_reduction"] is None
