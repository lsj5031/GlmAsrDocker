"""GLM-ASR FastAPI server for audio transcription."""

import json
import logging
import os
import signal
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Optional

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pydub import AudioSegment
from transformers import AutoProcessor, GlmAsrForConditionalGeneration

logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""

    text: str


class ModelState:
    """Container for model and component states."""

    def __init__(self) -> None:
        """Initialize model state with None values."""
        self.model: Optional[GlmAsrForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.device: Optional[torch.device] = None


model_state = ModelState()

MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-ASR-Nano-2512")
PORT = int(os.getenv("PORT", "8000"))
CHUNK_DURATION_MS = 30 * 1000  # 30 second chunks
CHUNK_OVERLAP_MS = 2 * 1000  # 2 second overlap
JSON_LOGGING = os.getenv("JSON_LOGGING", "false").lower() == "true"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500"))


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


def handle_sigterm(signum: int, frame: Any) -> None:
    """Handle SIGTERM signal for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


async def load_model_fn() -> None:
    """Load model and components on startup."""
    signal.signal(signal.SIGTERM, handle_sigterm)

    model_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading processor from {MODEL_ID}...")
    model_state.processor = AutoProcessor.from_pretrained(MODEL_ID)

    logger.info(f"Loading model from {MODEL_ID}...")
    model_state.model = GlmAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_state.model.eval()
    logger.info(f"Model loaded on device: {model_state.device}")


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
    return {"data": [{"id": "glm-nano-2512", "object": "model"}]}


def split_audio_into_chunks(audio_path: str) -> list[tuple[int, int]]:
    """Load audio and return list of (start_ms, end_ms) tuples for chunks."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunks = []
        step = CHUNK_DURATION_MS - CHUNK_OVERLAP_MS

        for i in range(0, duration_ms, step):
            chunk_end = min(i + CHUNK_DURATION_MS, duration_ms)
            chunks.append((i, chunk_end))
            if chunk_end >= duration_ms:
                break

        logger.info(f"[AUDIO CHUNKING] Split {duration_ms}ms into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"[AUDIO CHUNKING FAILED] {str(e)}")
        # Return full duration as single chunk on error
        try:
            audio = AudioSegment.from_file(audio_path)
            return [(0, len(audio))]
        except Exception:
            return [(0, 0)]


def transcribe_audio_array(audio_array: np.ndarray, sampling_rate: int) -> str:
    """Transcribe a single audio array using the new GLM-ASR API."""
    assert model_state.model is not None
    assert model_state.processor is not None

    # Resample if needed
    target_sr = model_state.processor.feature_extractor.sampling_rate
    if sampling_rate != target_sr:
        logger.debug(f"Resampling from {sampling_rate} to {target_sr}")
        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sr)
        audio_tensor = resampler(audio_tensor)
        audio_array = audio_tensor.squeeze(0).numpy()

    # Use the processor's apply_transcription_request method
    inputs = model_state.processor.apply_transcription_request(audio_array)

    # Move to device and dtype
    inputs = inputs.to(model_state.model.device, dtype=model_state.model.dtype)

    # Generate
    logger.info("[INFERENCE START] Running model.generate()...")
    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    logger.info("[INFERENCE SUCCESS] Generation completed")

    # Decode - skip input tokens
    prompt_len = inputs.input_ids.shape[1]
    transcript = model_state.processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    logger.info(
        f"[TRANSCRIPTION RESULT] '{transcript[:100]}'"
        f"{'...' if len(transcript) > 100 else ''}"
    )
    return transcript


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...), language: Optional[str] = "auto"
) -> TranscriptionResponse:
    """Transcribe audio file to text."""
    logger.debug("=== TRANSCRIPTION REQUEST RECEIVED ===")

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
        logger.debug("Reading file content from request...")
        content = await file.read()

        if len(content) == 0:
            logger.error("File content is empty")
            raise HTTPException(400, "Empty file received")

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
    logger.debug("[AUDIO CHECK] Checking if conversion is needed...")
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
                    ffmpeg.run(out_stream, overwrite_output=True)

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

            if len(chunks) == 1:
                # Single chunk - process normally
                logger.info("[AUDIO CHUNKING] Processing as single chunk")
                transcript = transcribe_audio_array(audio_array, sr)
            else:
                # Multiple chunks - process each and merge
                logger.info(
                    f"[AUDIO CHUNKING] Processing {len(chunks)} chunks sequentially"
                )
                audio_segment = AudioSegment.from_file(audio_path)
                transcripts = []

                for chunk_idx, (start_ms, end_ms) in enumerate(chunks, 1):
                    logger.info(
                        f"[CHUNK {chunk_idx}/{len(chunks)}] Processing {start_ms}ms-{end_ms}ms..."
                    )

                    # Extract chunk
                    audio_chunk = audio_segment[start_ms:end_ms]

                    # Convert to numpy array
                    chunk_array = np.array(audio_chunk.get_array_of_samples()).astype(
                        np.float32
                    )
                    chunk_array = chunk_array / 32768.0  # Normalize int16 to float

                    try:
                        chunk_text = transcribe_audio_array(
                            chunk_array, audio_chunk.frame_rate
                        )

                        if chunk_text:
                            transcripts.append(chunk_text)
                            logger.info(
                                f"[CHUNK {chunk_idx}] Result: '{chunk_text[:80]}'"
                            )
                    except Exception as e:
                        logger.error(f"[CHUNK {chunk_idx}] Failed: {str(e)}")

                transcript = " ".join(transcripts)
                logger.info(
                    f"[TRANSCRIPTION COMPLETE] Merged {len(transcripts)} chunks "
                    f"into {len(transcript)} chars"
                )

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


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
