"""GLM-ASR FastAPI server for audio transcription."""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import anyio
import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from silero_vad import load_silero_vad, get_speech_timestamps
from sse_starlette.sse import EventSourceResponse
from pydub import AudioSegment
from transformers import AutoProcessor, GlmAsrForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-ASR-Nano-2512")
PORT = int(os.getenv("PORT", "8000"))
CHUNK_DURATION_MS = int(os.getenv("CHUNK_DURATION_MS", "15000"))  # 15 second chunks
CHUNK_OVERLAP_MS = int(os.getenv("CHUNK_OVERLAP_MS", "1000"))  # 1 second overlap
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "300"))
JSON_LOGGING = os.getenv("JSON_LOGGING", "false").lower() == "true"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))  # 10 minutes default
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "500"))  # 500MB max
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "false").lower() == "true"
WARMUP = os.getenv("WARMUP", "true").lower() == "true"


def setup_logging() -> None:
    """Configure logging with optional JSON format."""
    if JSON_LOGGING:

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


# Initialize logging at module load time
setup_logging()

logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint (text format)."""

    text: str


class TranscriptionJsonResponse(BaseModel):
    """OpenAI-compatible JSON response envelope."""

    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    text: str


class TranscriptionSegment(BaseModel):
    """A single segment in verbose_json output."""

    id: int
    start: float
    end: float
    text: str


class TranscriptionVerboseJsonResponse(BaseModel):
    """OpenAI-compatible verbose_json response envelope."""

    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    text: str
    segments: list[TranscriptionSegment] = []


class ModelState:
    """Container for model and component states."""

    def __init__(self) -> None:
        """Initialize model state with None values."""
        self.model: Optional[GlmAsrForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.device: Optional[torch.device] = None
        self.vad_model: Optional[Any] = None


model_state = ModelState()


def configure_torch_runtime() -> None:
    """Configure torch runtime settings for lower latency."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("Enabled TF32 matmul for CUDA")


def warmup_model() -> None:
    """Run a tiny warmup inference to reduce first-request latency."""
    if not WARMUP:
        return
    if model_state.model is None or model_state.processor is None:
        return

    try:
        target_sr = model_state.processor.feature_extractor.sampling_rate
        dummy_audio = np.zeros(int(target_sr * 0.25), dtype=np.float32)
        inputs = model_state.processor.apply_transcription_request(dummy_audio)
        inputs = inputs.to(model_state.model.device, dtype=model_state.model.dtype)
        with torch.no_grad():
            model_state.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
            )
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


async def load_model_fn() -> None:
    """Load model and components on startup."""
    model_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime()

    logger.info(f"Loading processor from {MODEL_ID}...")
    model_state.processor = AutoProcessor.from_pretrained(MODEL_ID)

    logger.info(f"Loading model from {MODEL_ID}...")
    model_state.model = GlmAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_state.model.eval()

    if TORCH_COMPILE:
        if hasattr(torch, "compile"):
            try:
                model_state.model = torch.compile(
                    model_state.model, mode="reduce-overhead"
                )
                logger.info("torch.compile enabled (reduce-overhead)")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without it: {e}")
        else:
            logger.warning("torch.compile not available in this torch version")

    logger.info(f"Model loaded on device: {model_state.device}")

    logger.info("Loading Silero VAD model...")
    model_state.vad_model = load_silero_vad()
    logger.info("VAD model loaded")

    warmup_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle (startup/shutdown)."""
    await load_model_fn()
    yield
    if model_state.model is not None:
        del model_state.model
        torch.cuda.empty_cache()
        logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for container orchestration."""
    try:
        if model_state.model is None or model_state.processor is None:
            return {"status": "loading", "model_loaded": False}
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(model_state.device),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/v1/models")
async def list_models() -> dict:
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "glm-nano-2512",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "zai-org",
            }
        ],
    }


def format_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Convert a list of timed segments to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start_ms"])
        end = format_timestamp(seg["end_ms"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(f"{seg['text']}")
        lines.append("")
    return "\n".join(lines) + "\n" if lines else ""


def split_audio_into_chunks(audio_path: str) -> list[tuple[int, int]]:
    """Load audio and return list of (start_ms, end_ms) tuples for chunks using VAD."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)

        # Prepare audio for VAD
        # Convert to 16kHz mono for VAD
        audio_vad = audio.set_frame_rate(16000).set_channels(1)
        # Convert to numpy/tensor
        wav_data = (
            np.array(audio_vad.get_array_of_samples()).astype(np.float32) / 32768.0
        )
        wav_tensor = torch.from_numpy(wav_data)

        # Get speech timestamps
        # speech_timestamps is list of {'start': int, 'end': int} in samples
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model_state.vad_model,
            sampling_rate=16000,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=250,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            return_seconds=False,
        )

        if not speech_timestamps:
            logger.info("[VAD] No speech detected, falling back to fixed chunking")
            return _fixed_duration_chunks(duration_ms)

        # Helper to convert sample index to ms
        def s2ms(samples):
            return samples / 16.0

        # First, expand any speech segments that are too long into multiple segments
        expanded_segments = []
        for seg in speech_timestamps:
            seg_start_ms = s2ms(seg["start"])
            seg_end_ms = s2ms(seg["end"])
            seg_duration = seg_end_ms - seg_start_ms

            if seg_duration <= CHUNK_DURATION_MS:
                expanded_segments.append((seg_start_ms, seg_end_ms))
            else:
                # Split long segment into CHUNK_DURATION_MS pieces
                pos = seg_start_ms
                while pos < seg_end_ms:
                    chunk_end = min(pos + CHUNK_DURATION_MS, seg_end_ms)
                    expanded_segments.append((pos, chunk_end))
                    pos = chunk_end

        # Now merge segments into chunks, respecting max duration
        chunks = []
        current_chunk_start = expanded_segments[0][0]
        current_chunk_end = expanded_segments[0][1]

        for i in range(1, len(expanded_segments)):
            seg_start_ms, seg_end_ms = expanded_segments[i]

            # Check if adding this segment would exceed max duration
            if (seg_end_ms - current_chunk_start) > CHUNK_DURATION_MS:
                # Close current chunk at the start of silence (current_chunk_end)
                chunks.append((int(current_chunk_start), int(current_chunk_end)))
                current_chunk_start = seg_start_ms
                current_chunk_end = seg_end_ms
            else:
                # Extend current chunk
                current_chunk_end = seg_end_ms

        # Add final chunk
        chunks.append((int(current_chunk_start), int(current_chunk_end)))

        logger.info(
            f"[AUDIO CHUNKING] VAD split {duration_ms}ms into {len(chunks)} chunks (max {CHUNK_DURATION_MS}ms each)"
        )
        for i, (start, end) in enumerate(chunks):
            logger.info(f"  Chunk {i + 1}: {start}ms - {end}ms ({end - start}ms)")
        return chunks

    except Exception as e:
        logger.error(f"[AUDIO CHUNKING FAILED] {str(e)}")
        return _fixed_duration_chunks(duration_ms if "duration_ms" in dir() else 0)


def _fixed_duration_chunks(duration_ms: int) -> list[tuple[int, int]]:
    """Fallback to fixed duration chunking."""
    if duration_ms <= 0:
        return [(0, 0)]
    chunks = []
    step = max(1, CHUNK_DURATION_MS - CHUNK_OVERLAP_MS)

    for i in range(0, duration_ms, step):
        chunk_end = min(i + CHUNK_DURATION_MS, duration_ms)
        chunks.append((i, chunk_end))
        if chunk_end >= duration_ms:
            break
    return chunks


def _transcribe_audio_array_sync(
    audio_array: np.ndarray, sampling_rate: int, language: str = "auto"
) -> str:
    """Synchronous core transcription function to be run in a thread."""
    assert model_state.model is not None
    assert model_state.processor is not None

    logger.info(
        f"[MODEL INFERENCE] Starting inference on {len(audio_array) / sampling_rate:.1f}s of audio..."
    )

    # Resample if needed
    target_sr = model_state.processor.feature_extractor.sampling_rate
    if sampling_rate != target_sr:
        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sr)
        audio_tensor = resampler(audio_tensor)
        audio_array = audio_tensor.squeeze(0).numpy()

    # Use the processor's apply_transcription_request method
    try:
        if language and language.lower() != "auto":
            inputs = model_state.processor.apply_transcription_request(
                audio_array, language=language
            )
        else:
            inputs = model_state.processor.apply_transcription_request(audio_array)
    except TypeError:
        inputs = model_state.processor.apply_transcription_request(audio_array)

    # Move to device and dtype
    inputs = inputs.to(model_state.model.device, dtype=model_state.model.dtype)

    # Generate
    logger.info("[MODEL INFERENCE] Running model.generate()...")
    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # Decode - skip input tokens
    prompt_len = inputs.input_ids.shape[1]
    transcript = model_state.processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    return transcript


async def transcribe_audio_array(
    audio_array: np.ndarray, sampling_rate: int, language: str = "auto"
) -> str:
    """Transcribe a single audio array using the new GLM-ASR API (Offloaded to worker thread)."""
    transcript = await anyio.to_thread.run_sync(
        _transcribe_audio_array_sync, audio_array, sampling_rate, language
    )

    logger.info(
        f"[TRANSCRIPTION RESULT] '{transcript[:100]}'"
        f"{'...' if len(transcript) > 100 else ''}"
    )
    return transcript


async def transcribe_stream_generator(
    audio_path: str,
    audio_array: np.ndarray,
    sr: int,
    chunks: list[tuple[int, int]],
    language: str = "auto",
):
    """Async generator that yields SSE events for each transcribed chunk."""
    if len(chunks) == 1:
        logger.info("[SSE STREAM] Processing as single chunk")
        try:
            transcript = await transcribe_audio_array(
                audio_array, sr, language=language
            )
            yield {"data": transcript or "[Empty transcription]"}
        except Exception as e:
            logger.error(f"[SSE STREAM] Single chunk failed: {str(e)}")
            yield {"data": f"[Error: {str(e)}]"}
    else:
        logger.info(f"[SSE STREAM] Processing {len(chunks)} chunks")
        audio_segment = AudioSegment.from_file(audio_path)

        for chunk_idx, (start_ms, end_ms) in enumerate(chunks, 1):
            logger.info(
                f"[SSE CHUNK {chunk_idx}/{len(chunks)}] Processing {start_ms}ms-{end_ms}ms..."
            )
            audio_chunk = audio_segment[start_ms:end_ms].set_channels(1)
            chunk_array = np.array(audio_chunk.get_array_of_samples()).astype(
                np.float32
            )
            chunk_array = chunk_array / 32768.0

            try:
                chunk_text = await transcribe_audio_array(
                    chunk_array, audio_chunk.frame_rate, language=language
                )
                if chunk_text:
                    yield {"data": chunk_text}
                    logger.info(
                        f"[SSE CHUNK {chunk_idx}] Streamed: '{chunk_text[:80]}'"
                    )
            except Exception as e:
                logger.error(f"[SSE CHUNK {chunk_idx}] Failed: {str(e)}")

    yield {"data": "[DONE]"}
    logger.info("[SSE STREAM] Streaming complete")


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form("auto"),
    stream: bool = Form(False),
    response_format: Literal["text", "json", "verbose_json", "srt"] = Form("text"),
    timestamp_granularities: Optional[str] = Form(None),
):
    """Transcribe audio file to text."""
    logger.info("=== TRANSCRIPTION REQUEST RECEIVED ===")

    if stream and response_format in ("srt", "verbose_json"):
        raise HTTPException(400, f"{response_format} format is not supported with streaming mode")

    if not model_state.model or not model_state.processor:
        logger.error("Model not loaded!")
        raise HTTPException(500, "Model not loaded")

    # Validate file
    if not file.filename:
        logger.error("No filename provided in request")
        raise HTTPException(400, "No filename provided")

    logger.info(f"[FRONTEND->BACKEND] File received: {file.filename}")
    logger.debug(f"  - Content-Type: {file.content_type}")
    logger.debug(f"  - Language: {language}")

    # Read file with size check
    try:
        logger.info("Reading file content from request...")
        content = await file.read()

        if len(content) == 0:
            logger.error("File content is empty")
            raise HTTPException(400, "Empty file received")

        # Check max file size
        max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            logger.error(f"File too large: {len(content) / (1024*1024):.1f}MB > {MAX_AUDIO_SIZE_MB}MB")
            raise HTTPException(
                413,
                f"File too large. Maximum size: {MAX_AUDIO_SIZE_MB}MB. "
                f"For longer audio, please split into smaller chunks on the client side."
            )

        # Check for valid audio file headers
        file_sig = content[:4].hex().lower()
        is_mp4 = (
            file_sig.startswith("00000020")
            or file_sig[4:8] == "6674"
            or content[4:8] == b"ftyp"
        )
        is_wav = content[:4] == b"RIFF"

        if not (is_mp4 or is_wav):
            logger.warning(f"[FILE VALIDATION] Unknown format signature: {file_sig}")

        logger.info(
            f"[FILE UPLOAD SUCCESS] Received {len(content)} bytes "
            f"({len(content) / (1024 * 1024):.2f} MB)"
        )
        logger.debug(f"  - First 20 bytes (hex): {content[:20].hex()}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[FILE UPLOAD FAILED] Error reading file: {str(e)}", exc_info=True
        )
        raise HTTPException(400, f"Failed to read uploaded file: {str(e)}")

    # Check if conversion is needed
    logger.info("[AUDIO PROCESSING] Starting audio processing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_audio")
        audio_path = os.path.join(tmpdir, "audio.wav")

        try:
            # Write uploaded content to temp file
            logger.debug(f"  - Writing {len(content)} bytes to temporary file...")
            with open(input_path, "wb") as f:
                f.write(content)

            # Check audio properties
            needs_conversion = False
            duration_seconds = 0
            try:
                # Try to load as-is to check properties
                if is_wav:
                    logger.debug("[AUDIO CHECK] File is WAV, probing properties...")
                    wav_data, sr = sf.read(input_path)
                    # sf.read returns (frames, channels) or (frames,)
                    # We just need duration
                    duration_seconds = len(wav_data) / sr
                    logger.debug(
                        f"  - Sample rate: {sr}, duration: {duration_seconds:.2f}s"
                    )
                    needs_conversion = sr != 16000
                    if needs_conversion:
                        logger.info(
                            f"[AUDIO CHECK] Conversion needed: sample rate {sr} != 16000"
                        )
                else:
                    logger.debug("[AUDIO CHECK] File is not WAV, conversion needed")
                    needs_conversion = True
            except Exception as e:
                logger.warning(f"[AUDIO CHECK] Could not probe, will convert: {str(e)}")
                needs_conversion = True

            # Check if audio is longer than 1 minute
            is_long_audio = duration_seconds > 60
            if is_long_audio:
                logger.info(
                    f"[AUDIO CHECK] Audio is long ({duration_seconds:.2f}s > 60s), "
                    "will convert for chunking"
                )
                needs_conversion = True

            # Convert if needed
            if needs_conversion:
                logger.info(
                    "[AUDIO CONVERSION] Starting ffmpeg conversion to WAV format..."
                )
                try:
                    logger.debug(
                        f"  - Input file size: {os.path.getsize(input_path)} bytes"
                    )
                    logger.debug(f"  - Output path: {audio_path}")
                    logger.debug(
                        "  - Ffmpeg parameters: format=wav, acodec=pcm_s16le, ar=16000"
                    )

                    input_stream = ffmpeg.input(input_path)
                    out_stream = ffmpeg.output(
                        input_stream,
                        audio_path,
                        format="wav",
                        acodec="pcm_s16le",
                        ar=16000,
                    )
                    await anyio.to_thread.run_sync(
                        lambda: ffmpeg.run(out_stream, overwrite_output=True)
                    )

                    converted_size = (
                        os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                    )
                    if converted_size < 1000:
                        logger.error(
                            f"[AUDIO CONVERSION FAILED] Output file too small: {converted_size} bytes"
                        )
                        raise HTTPException(
                            400,
                            "Audio conversion failed - output file is empty or too small. "
                            "Input file may be corrupted.",
                        )

                    logger.info(
                        f"[AUDIO CONVERSION SUCCESS] Converted to WAV: {converted_size} bytes "
                        f"({converted_size / (1024 * 1024):.2f} MB)"
                    )
                except ffmpeg.Error as e:
                    error_msg = e.stderr.decode() if e.stderr else str(e)
                    logger.error(
                        f"[FFMPEG CONVERSION FAILED] FFmpeg error: {error_msg}",
                        exc_info=True,
                    )
                    raise HTTPException(400, f"Audio conversion failed: {error_msg}")
                except HTTPException:
                    raise
            else:
                # No conversion needed, use original file
                logger.info(
                    "[AUDIO CHECK] No conversion needed, using original WAV file"
                )
                audio_path = input_path

        except Exception as e:
            logger.error(
                f"[AUDIO PROCESSING ERROR] Unexpected error: {str(e)}", exc_info=True
            )
            raise HTTPException(500, f"Audio processing error: {str(e)}")

        try:
            # Load the audio
            wav_data, sr = sf.read(audio_path)
            # Ensure float32
            wav_data = wav_data.astype(np.float32)

            # sf.read returns (frames, channels) or (frames,)
            # Convert to (channels, frames) for consistency if needed, but we pass to transcribe_audio_array as numpy
            if wav_data.ndim > 1:
                # Mix down to mono by averaging channels
                wav_data = wav_data.mean(axis=1)

            audio_array = wav_data

            # Split audio into chunks if needed
            chunks = split_audio_into_chunks(audio_path)

            # Handle SSE streaming mode
            if stream:
                logger.info("[SSE MODE] Preparing streaming response")
                # Copy audio to persistent temp file for streaming
                persistent_audio = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                shutil.copy2(audio_path, persistent_audio.name)
                persistent_audio.close()

                async def stream_with_cleanup():
                    try:
                        async for event in transcribe_stream_generator(
                            persistent_audio.name,
                            audio_array,
                            sr,
                            chunks,
                            language=language,
                        ):
                            yield event
                    finally:
                        try:
                            os.unlink(persistent_audio.name)
                            logger.debug(
                                f"[SSE CLEANUP] Removed temp file: {persistent_audio.name}"
                            )
                        except OSError:
                            pass

                return EventSourceResponse(
                    stream_with_cleanup(), media_type="text/event-stream"
                )

            # Non-streaming mode: process all chunks and return complete response
            segments = []
            audio_segment = AudioSegment.from_file(audio_path)

            for chunk_idx, (start_ms, end_ms) in enumerate(chunks, 1):
                logger.info(
                    f"[CHUNK {chunk_idx}/{len(chunks)}] Processing {start_ms}ms-{end_ms}ms..."
                )

                # Extract chunk and convert to mono
                audio_chunk = audio_segment[start_ms:end_ms].set_channels(1)

                # Convert to numpy array
                chunk_array = np.array(audio_chunk.get_array_of_samples()).astype(
                    np.float32
                )
                chunk_array = chunk_array / 32768.0  # Normalize int16 to float

                try:
                    chunk_text = await transcribe_audio_array(
                        chunk_array, audio_chunk.frame_rate, language=language
                    )

                    if chunk_text:
                        segments.append(
                            {"start_ms": start_ms, "end_ms": end_ms, "text": chunk_text}
                        )
                        logger.info(f"[CHUNK {chunk_idx}] Result: '{chunk_text[:80]}'")
                except Exception as e:
                    logger.error(f"[CHUNK {chunk_idx}] Failed: {str(e)}")

            if response_format == "srt":
                logger.info(
                    f"[SRT OUTPUT] Generating SRT from {len(segments)} segments"
                )
                srt_output = segments_to_srt(segments)
                logger.info(
                    f"[SRT OUTPUT] Generated {len(srt_output)} bytes of SRT content"
                )
                return Response(content=srt_output, media_type="text/plain")

            transcript = " ".join([seg["text"] for seg in segments])
            # Calculate audio duration from sample rate and array length
            audio_duration = round(len(audio_array) / sr, 2) if sr > 0 else 0.0
            effective_language = language if language and language.lower() != "auto" else None

            logger.info(
                f"[TRANSCRIPTION COMPLETE] Merged {len(segments)} chunks "
                f"into {len(transcript)} chars"
            )

            if response_format == "verbose_json":
                # OpenAI verbose_json: include segment-level timestamps
                verbose_segments = [
                    TranscriptionSegment(
                        id=i,
                        start=round(seg["start_ms"] / 1000.0, 3),
                        end=round(seg["end_ms"] / 1000.0, 3),
                        text=seg["text"],
                    )
                    for i, seg in enumerate(segments)
                ]
                return TranscriptionVerboseJsonResponse(
                    task="transcribe",
                    language=effective_language,
                    duration=audio_duration,
                    text=transcript or "",
                    segments=verbose_segments,
                )

            if response_format == "json":
                # OpenAI json envelope
                return TranscriptionJsonResponse(
                    task="transcribe",
                    language=effective_language,
                    duration=audio_duration,
                    text=transcript or "",
                )

            # Default text format (backward compatible)
            return TranscriptionResponse(text=transcript or "[Empty transcription]")

        except ValueError as e:
            logger.error(
                f"[AUDIO PROCESSING ERROR] ValueError: {str(e)}", exc_info=True
            )
            raise HTTPException(400, f"Audio processing error: {str(e)}")
        except RuntimeError as e:
            logger.error(
                f"[MODEL INFERENCE ERROR] RuntimeError: {str(e)}", exc_info=True
            )
            raise HTTPException(500, f"Inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"[INFERENCE ERROR] Unexpected error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Transcription error: {str(e)}")


# WebSocket streaming constants
WS_SAMPLE_RATE = 16000
WS_TRANSCRIBE_INTERVAL_SEC = (
    0.8  # Transcribe every 0.8s of new audio for faster feedback
)
WS_MAX_BUFFER_MS = 25000  # Force flush after 25s to prevent latency lag
WS_CONTEXT_WINDOW_MS = 3000  # Keep last 3s as context when flushing


@app.websocket("/v1/audio/transcriptions/stream")
async def websocket_transcribe(websocket: WebSocket, language: str = "auto"):
    """
    Simplified WebSocket endpoint for real-time audio transcription.

    Server is a "dumb pipe" - just accumulates audio and transcribes periodically.
    All VAD/silence detection is handled by the client.

    Client sends:
    - Binary messages: PCM 16-bit, 16kHz, mono audio chunks
    - JSON messages: {"action": "stop"} to signal end of recording

    Server sends:
    - JSON messages: {"text": "...", "final": false/true}
    """
    await websocket.accept()
    logger.info("[WS] Client connected")

    audio_buffer = np.array([], dtype=np.float32)
    last_transcription = ""
    last_inference_audio_len = 0  # Track how much audio we've processed

    try:
        while True:
            # Check connection state
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.info("[WS] Client disconnected (state check)")
                break

            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Handle binary audio data
            if "bytes" in message:
                audio_bytes = message["bytes"]
                # Convert PCM 16-bit to float32
                chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                chunk = chunk / 32768.0  # Normalize to [-1, 1]
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # Check how much new audio since last inference
                new_audio_sec = (
                    len(audio_buffer) - last_inference_audio_len
                ) / WS_SAMPLE_RATE

                # Calculate current buffer duration
                buffer_duration_ms = (len(audio_buffer) / WS_SAMPLE_RATE) * 1000

                # Transcribe if we have enough new audio
                if (
                    new_audio_sec >= WS_TRANSCRIBE_INTERVAL_SEC
                    or buffer_duration_ms >= WS_MAX_BUFFER_MS
                ):
                    try:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            break

                        transcript = await transcribe_audio_array(
                            audio_buffer, WS_SAMPLE_RATE, language=language
                        )

                        # Force flush if buffer is full (prevent infinite growth/latency lag)
                        is_force_flush = buffer_duration_ms >= WS_MAX_BUFFER_MS

                        if transcript and (
                            transcript != last_transcription or is_force_flush
                        ):
                            last_transcription = transcript
                            await websocket.send_json(
                                {
                                    "text": transcript,
                                    "final": is_force_flush,  # Send final=true to commit on client
                                }
                            )
                            logger.info(
                                f"[WS] {'Flush' if is_force_flush else 'Partial'}: {transcript[:50]}..."
                            )

                        if is_force_flush:
                            # Keep last N seconds as context to avoid cutting words
                            context_samples = int(WS_SAMPLE_RATE * (WS_CONTEXT_WINDOW_MS / 1000))
                            if len(audio_buffer) > context_samples:
                                audio_buffer = audio_buffer[-context_samples:]
                                last_inference_audio_len = len(audio_buffer)
                            else:
                                audio_buffer = np.array([], dtype=np.float32)
                                last_inference_audio_len = 0
                            last_transcription = ""
                            logger.info(f"[WS] Buffer flushed, kept {len(audio_buffer)/WS_SAMPLE_RATE:.1f}s context")
                        else:
                            last_inference_audio_len = len(audio_buffer)

                    except Exception as e:
                        logger.error(f"[WS] Transcription error: {e}")
                        break

            # Handle JSON control messages
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("action") == "stop":
                        logger.info("[WS] Stop signal received")
                        # Final transcription of all accumulated audio
                        if len(audio_buffer) >= WS_SAMPLE_RATE * 0.3:
                            try:
                                transcript = await transcribe_audio_array(
                                    audio_buffer, WS_SAMPLE_RATE, language=language
                                )
                                await websocket.send_json(
                                    {
                                        "text": transcript or "",
                                        "final": True,
                                    }
                                )
                                logger.info(
                                    f"[WS] Final: {transcript[:50] if transcript else '(empty)'}..."
                                )
                            except Exception as e:
                                logger.error(f"[WS] Final transcription error: {e}")
                                await websocket.send_json(
                                    {
                                        "text": last_transcription,
                                        "final": True,
                                        "error": str(e),
                                    }
                                )
                        else:
                            await websocket.send_json(
                                {
                                    "text": last_transcription,
                                    "final": True,
                                }
                            )
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}", exc_info=True)
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("[WS] Connection closed")


# ---------------------------------------------------------------------------
# OpenAI Realtime API – transcription-only WebSocket endpoint
# ---------------------------------------------------------------------------

import asyncio
import base64
import uuid

# Defaults for the Realtime transcription session
RT_DEFAULT_INPUT_FORMAT = "pcm16"  # pcm16 = 16-bit PCM, 24kHz (OpenAI default)
RT_DEFAULT_SAMPLE_RATE = 24000
RT_VAD_THRESHOLD = float(os.getenv("RT_VAD_THRESHOLD", "0.5"))
RT_VAD_PREFIX_PADDING_MS = int(os.getenv("RT_VAD_PREFIX_PADDING_MS", "300"))
RT_VAD_SILENCE_MS = int(os.getenv("RT_VAD_SILENCE_MS", "500"))
RT_VAD_MIN_SPEECH_MS = int(os.getenv("RT_VAD_MIN_SPEECH_MS", "250"))
RT_MAX_BUFFER_SEC = 30  # hard cap: force commit if buffer exceeds this


def _rt_event(event_type: str, **kwargs) -> dict:
    """Build a Realtime API server event dict with a unique event_id."""
    return {"event_id": f"event_{uuid.uuid4().hex[:8]}", "type": event_type, **kwargs}


class RealtimeTranscriptionSession:
    """Manages state for a single OpenAI-compatible Realtime transcription session."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.session_id = f"session_{uuid.uuid4().hex[:12]}"
        self.item_counter = 0

        # Session config (mutable via transcription_session.update)
        self.input_audio_format: str = RT_DEFAULT_INPUT_FORMAT
        self.sample_rate: int = RT_DEFAULT_SAMPLE_RATE
        self.language: str = ""
        self.prompt: str = ""
        self.turn_detection_type: str = "server_vad"
        self.vad_threshold: float = RT_VAD_THRESHOLD
        self.vad_prefix_padding_ms: int = RT_VAD_PREFIX_PADDING_MS
        self.vad_silence_duration_ms: int = RT_VAD_SILENCE_MS
        self.noise_reduction: Optional[str] = "near_field"
        self.include_logprobs: bool = False

        # Audio buffer for the current utterance (float32)
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_samples: int = 0

        # VAD ring buffer for prefix padding
        self.vad_ring: list[np.ndarray] = []
        self.vad_ring_samples: int = 0

    # ---- helpers ----

    def _next_item_id(self) -> str:
        self.item_counter += 1
        return f"item_{self.item_counter:03d}"

    def _get_buffer_array(self) -> Optional[np.ndarray]:
        if not self.audio_buffer:
            return None
        return np.concatenate(self.audio_buffer)

    def _clear_buffer(self) -> None:
        self.audio_buffer.clear()
        self.buffer_samples = 0

    # ---- session events ----

    async def send_created(self) -> None:
        await self.ws.send_json(
            _rt_event("transcription_session.created", session=self._session_dict())
        )

    async def send_updated(self) -> None:
        await self.ws.send_json(
            _rt_event("transcription_session.updated", session=self._session_dict())
        )

    async def send_error(self, message: str) -> None:
        await self.ws.send_json(
            _rt_event("error", error={"type": "invalid_request_error", "message": message})
        )

    # ---- audio handling ----

    def _pcm16_to_float32(self, raw_bytes: bytes) -> np.ndarray:
        """Decode PCM16 bytes → float32 array normalised to [-1, 1]."""
        import struct as _struct
        n_samples = len(raw_bytes) // 2
        if n_samples == 0:
            return np.array([], dtype=np.float32)
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _base64_to_audio(self, b64: str) -> np.ndarray:
        """Decode Base64 PCM16 → float32."""
        raw = base64.b64decode(b64)
        return self._pcm16_to_float32(raw)

    async def append_audio(self, b64_audio: str) -> None:
        """Append a Base64-encoded PCM16 audio chunk to the buffer."""
        chunk = self._base64_to_audio(b64_audio)
        n = len(chunk)
        self.audio_buffer.append(chunk)
        self.buffer_samples += n

        # Maintain VAD prefix padding ring
        self.vad_ring.append(chunk)
        self.vad_ring_samples += n
        max_ring = int(self.sample_rate * (self.vad_prefix_padding_ms / 1000.0))
        while self.vad_ring_samples > max_ring and len(self.vad_ring) > 1:
            removed = self.vad_ring.pop(0)
            self.vad_ring_samples -= len(removed)

        # If turn_detection is server_vad, run VAD on the accumulated buffer
        if self.turn_detection_type == "server_vad":
            await self._maybe_vad_commit()

    async def _maybe_vad_commit(self) -> None:
        """Run Silero VAD on the buffer and commit if speech has ended."""
        if model_state.vad_model is None:
            return

        arr = self._get_buffer_array()
        if arr is None:
            return
        try:
            arr_len = len(arr)
        except TypeError:
            arr_len = 0
        if arr_len < self.sample_rate * 0.3:
            return  # not enough audio yet

        # Hard cap: force commit if buffer exceeds max
        if self.buffer_samples > self.sample_rate * RT_MAX_BUFFER_SEC:
            await self._commit_buffer(reason="max_buffer")
            return

        # Run VAD
        try:
            speech_ts = await anyio.to_thread.run_sync(
                lambda: get_speech_timestamps(
                    arr,
                    model_state.vad_model,
                    sampling_rate=self.sample_rate,
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=RT_VAD_MIN_SPEECH_MS,
                    min_silence_duration_ms=self.vad_silence_duration_ms,
                    return_seconds=False,
                )
            )
        except Exception as e:
            logger.warning(f"[RT-VAD] VAD failed: {e}")
            return

        if not speech_ts:
            # No speech detected in current buffer; if we have a lot of audio, flush
            if self.buffer_samples > self.sample_rate * 2:
                # Check if there's a long trailing silence
                await self._commit_buffer(reason="silence")
            return

        # Check if the last speech segment ended well before the buffer end
        last_speech_end = speech_ts[-1]["end"]
        silence_after = len(arr) - last_speech_end
        silence_samples = int(self.sample_rate * (self.vad_silence_duration_ms / 1000.0))

        if silence_after >= silence_samples and last_speech_end > 0:
            # Speech has ended → commit the buffer up to the end of speech
            await self._commit_buffer(reason="vad_silence")

    async def _commit_buffer(self, reason: str = "manual") -> None:
        """Commit the current audio buffer, transcribe it, and send events."""
        arr = self._get_buffer_array()
        if arr is None:
            return
        try:
            arr_len = len(arr)
        except TypeError:
            arr_len = 0
        if arr_len < self.sample_rate * 0.1:
            return

        item_id = self._next_item_id()
        previous_item_id = getattr(self, "_last_item_id", None)

        # 1. Send input_audio_buffer.committed
        await self.ws.send_json(
            _rt_event(
                "input_audio_buffer.committed",
                item_id=item_id,
                previous_item_id=previous_item_id,
            )
        )

        self._last_item_id = item_id

        # 2. Transcribe
        try:
            lang = self.language if self.language else "auto"
            transcript = await transcribe_audio_array(arr, self.sample_rate, language=lang)
        except Exception as e:
            logger.error(f"[RT-TRANSCRIBE] Transcription failed: {e}")
            await self.ws.send_json(
                _rt_event(
                    "transcript.text.done",
                    item_id=item_id,
                    content_index=0,
                    text="",
                )
            )
            self._clear_buffer()
            return

        if not transcript:
            self._clear_buffer()
            return

        # 3. Send delta event (full transcript as one delta for simplicity —
        #    GLM-ASR doesn't stream tokens, so we send the whole thing)
        await self.ws.send_json(
            _rt_event(
                "transcript.text.delta",
                item_id=item_id,
                content_index=0,
                delta=transcript,
            )
        )

        # 4. Send completed event
        await self.ws.send_json(
            _rt_event(
                "transcript.text.done",
                item_id=item_id,
                content_index=0,
                text=transcript,
            )
        )

        logger.info(f"[RT] Committed ({reason}): '{transcript[:80]}'")
        self._clear_buffer()

    async def commit_buffer(self) -> None:
        """Public API: commit buffer on explicit client request."""
        await self._commit_buffer(reason="client_commit")

    # ---- session dict for responses ----

    def _session_dict(self) -> dict:
        td = None
        if self.turn_detection_type == "server_vad":
            td = {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
            }
        nr = None
        if self.noise_reduction:
            nr = {"type": self.noise_reduction}
        include = []
        if self.include_logprobs:
            include.append("item.input_audio_transcription.logprobs")
        return {
            "object": "realtime.transcription_session",
            "type": "transcription",
            "id": self.session_id,
            "input_audio_format": self.input_audio_format,
            "input_audio_transcription": {
                "model": "glm-nano-2512",
                "language": self.language or None,
                "prompt": self.prompt,
            },
            "turn_detection": td,
            "input_audio_noise_reduction": nr,
            "include": include,
        }

    # ---- handle client events ----

    async def handle_event(self, event: dict) -> None:
        """Route a single client-sent JSON event."""
        evt_type = event.get("type", "")

        if evt_type == "transcription_session.update":
            await self._handle_session_update(event)
        elif evt_type == "input_audio_buffer.append":
            audio_b64 = event.get("audio", "")
            if audio_b64:
                await self.append_audio(audio_b64)
        elif evt_type == "input_audio_buffer.commit":
            await self.commit_buffer()
        elif evt_type == "input_audio_buffer.clear":
            self._clear_buffer()
            await self.ws.send_json(_rt_event("input_audio_buffer.cleared"))
        else:
            logger.warning(f"[RT] Unknown client event type: {evt_type}")

    async def _handle_session_update(self, event: dict) -> None:
        """Handle transcription_session.update client event."""
        # Update input format
        fmt = event.get("input_audio_format")
        if fmt:
            self.input_audio_format = fmt
            if fmt == "pcm16":
                self.sample_rate = 24000
            elif fmt == "pcm8":
                self.sample_rate = 8000
            # g711 variants would need transcoding — accept but keep 24kHz

        # Update transcription config
        xcr = event.get("input_audio_transcription")
        if isinstance(xcr, dict):
            self.language = xcr.get("language", "")
            self.prompt = xcr.get("prompt", "")

        # Update turn detection
        td = event.get("turn_detection")
        if td is None:
            self.turn_detection_type = "none"
        elif isinstance(td, dict):
            self.turn_detection_type = td.get("type", "server_vad")
            if "threshold" in td:
                self.vad_threshold = float(td["threshold"])
            if "prefix_padding_ms" in td:
                self.vad_prefix_padding_ms = int(td["prefix_padding_ms"])
            if "silence_duration_ms" in td:
                self.vad_silence_duration_ms = int(td["silence_duration_ms"])

        # Update noise reduction
        nr = event.get("input_audio_noise_reduction")
        if nr is None:
            self.noise_reduction = None
        elif isinstance(nr, dict):
            self.noise_reduction = nr.get("type")

        # Update include
        inc = event.get("include")
        if isinstance(inc, list):
            self.include_logprobs = "item.input_audio_transcription.logprobs" in inc

        await self.send_updated()


@app.websocket("/v1/realtime")
async def realtime_transcription_ws(
    websocket: WebSocket,
    intent: Optional[str] = None,
):
    """
    OpenAI Realtime API compatible transcription WebSocket.

    Query params:
    - intent: "transcription" (recommended for clarity, but not required)

    Client-sent events (JSON):
    - transcription_session.update: configure the session
    - input_audio_buffer.append: send Base64-encoded PCM16 audio
    - input_audio_buffer.commit: manually commit buffer for transcription
    - input_audio_buffer.clear: clear the audio buffer

    Server-sent events (JSON):
    - transcription_session.created / .updated
    - input_audio_buffer.committed
    - input_audio_buffer.cleared
    - transcript.text.delta
    - transcript.text.done
    - error
    """
    await websocket.accept()
    logger.info(f"[RT-WS] Client connected (intent={intent})")

    session = RealtimeTranscriptionSession(websocket)

    # Send session.created on connect
    await session.send_created()

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await session.send_error("Invalid JSON")
                continue

            await session.handle_event(event)

    except WebSocketDisconnect:
        logger.info("[RT-WS] Client disconnected")
    except Exception as e:
        logger.error(f"[RT-WS] Error: {e}", exc_info=True)
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("[RT-WS] Connection closed")


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
