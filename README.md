# GLM-ASR

A production-ready FastAPI-based speech-to-text service powered by the GLM-ASR-Nano model. Transcribe audio and video files with ease using this OpenAI-compatible API.

Inspired by the architecture of [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) by Fedir Zadniprovskyi.

## ✨ Features

### Core Capabilities
- **🎯 OpenAI-compatible API**: Drop-in replacement for OpenAI's audio transcription endpoint
- **🎬 Multi-format support**: Handles 30+ audio and video formats via FFmpeg
- **⚡ GPU acceleration**: CUDA support for fast inference (~8x real-time speed)
- **📊 Built-in CLI**: Easy-to-use command-line interface for batch processing
- **🔄 Automatic chunking**: VAD-based smart audio segmentation for long files
- **📝 Multiple output formats**: Text and SRT subtitle formats
- **🌐 Multi-language support**: Auto-detection and manual language selection
- **💪 Production-ready**: Docker support, health checks, graceful shutdown

### Advanced Features
- **🎥 Video support**: Automatically extracts audio from video files
- **📈 Progress tracking**: Real-time progress bars for long transcription jobs
- **🏥 Health monitoring**: Built-in health check endpoint for orchestration
- **⚙️ Flexible configuration**: Environment variables for all settings
- **🔧 WebSocket streaming**: Real-time transcription support
- **🛡️ Non-root container**: Enhanced security with dedicated user

## 📋 Requirements

### Docker Deployment (Recommended)
- Docker & Docker Compose
- CUDA 12.8 + NVIDIA drivers (for GPU acceleration)
- 4GB+ GPU VRAM

### Local Development
- Python 3.10+
- CUDA 12.8 (optional, for GPU acceleration)
- FFmpeg
- libsndfile

## 🚀 Quick Start

### 1. Clone and Start (Easiest)

```bash
# Clone the repository
git clone https://github.com/lsj5031/glm-asr-docker.git
cd glm-asr-docker

# Start the service with GPU support
docker compose up -d

# Wait for the model to load (~30 seconds)
docker compose logs -f
```

The service will be available at `http://localhost:18000`

### 2. Transcribe Your First File

```bash
# Create data directory
mkdir -p data

# Copy your audio/video file
cp your-file.mp3 data/

# Transcribe using CLI
make transcribe INPUT=/app/data/your-file.mp3

# Or use docker compose directly
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/your-file.mp3
```

### 3. Check Server Health

```bash
make health
# or
curl http://localhost:18000/health
```

## 📦 Installation

### Using Docker Compose (Recommended)

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f glm-asr

# Stop the service
docker compose down
```

### Using Make Commands

The Makefile provides convenient shortcuts:

```bash
# Show all available commands
make help

# Container management
make up          # Start the service
make build       # Build Docker image
make logs        # View logs
make shell       # Open shell in container
make restart     # Restart the service
make clean       # Stop and remove containers

# CLI commands
make transcribe INPUT=/app/data/file.mp3
make health
make cli
```

### Local Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# Run the server
python server.py
```

## 🎯 CLI Usage (Recommended)

The built-in CLI provides the easiest way to transcribe files:

### Basic Usage

```bash
# Start the service first
make up

# Transcribe a single file
make transcribe INPUT=/app/data/podcast.mp3

# With custom output
make transcribe INPUT=/app/data/interview.wav OUTPUT=/app/data/transcript.txt

# With specific language
make transcribe INPUT=/app/data/speech.mp3 LANGUAGE=zh

# To SRT subtitle format
make transcribe INPUT=/app/data/video.mp4 FORMAT=srt
```

### Direct Docker Compose Usage

```bash
# Single file
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/audio.mp3

# With all options
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/audio.mp3 \
  -o /app/data/output.txt \
  -l en \
  -f srt \
  -c 10

# Batch process directory
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/

# Check health
docker compose exec glm-asr python glm_asr_cli.py health

# Show help
docker compose exec glm-asr python glm_asr_cli.py --help
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `input` | - | Audio file(s) or directory (required) | - |
| `--output` | `-o` | Output transcript path | Auto-generated |
| `--server-url` | `-s` | Server URL | http://localhost:8000 |
| `--language` | `-l` | Language code (ISO 639-1) | auto |
| `--format` | `-f` | Output format (text/srt/json/verbose_json) | text |

### Global CLI Wrapper

Use `glm-asr` from anywhere on your system:

```bash
glm-asr transcribe ~/Downloads/audio.mp3
glm-asr transcribe video.mp4 -l zh --format srt
glm-asr health
```

## 🎬 Supported Formats

The service supports **all audio and video formats** that FFmpeg can handle:

### Audio Formats (30+)
- **Common**: MP3, WAV, FLAC, M4A, AAC, OGG, WMA
- **Additional**: OPUS, AIFF, AU, RA, AMR, DSD, APE, WV, AC3, DTS

### Video Formats (audio extracted automatically)
- **Common**: MP4, MOV, AVI, MKV, WEBM, WMV, FLV
- **Additional**: MPG/MPEG, M4V, TS, MTS, VOB, RM, RMVB

### Examples

```bash
# All of these work:
make transcribe INPUT=/app/data/podcast.mp3
make transcribe INPUT=/app/data/interview.wav
make transcribe INPUT=/app/data/meeting.m4a
make transcribe INPUT=/app/data/lecture.mp4    # Video!
make transcribe INPUT=/app/data/webinar.mkv    # Video!

# Batch process mixed formats
cp ~/media/*.{mp3,wav,m4a,mp4,mov} data/
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/
```

## 🔌 API Usage

### Health Check

```bash
curl http://localhost:18000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Transcribe Audio

```bash
curl -X POST "http://localhost:18000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=auto" \
  -F "response_format=text"
```

Response:
```json
{
  "text": "Transcribed text here..."
}
```

### SRT Subtitles

```bash
curl -X POST "http://localhost:18000/v1/audio/transcriptions" \
  -F "file=@video.mp4" \
  -F "response_format=srt" \
  -o subtitles.srt
```

### List Models

```bash
curl http://localhost:18000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "glm-nano-2512",
      "object": "model",
      "created": 1234567890,
      "owned_by": "zai-org"
    }
  ]
}
```

### Interactive API Documentation

Visit `http://localhost:18000/docs` for interactive Swagger UI documentation.

## ⚙️ Configuration

Configure via environment variables in `docker-compose.yml`:

```yaml
environment:
  # Server
  - PORT=8000

  # Model
  - MODEL_ID=zai-org/GLM-ASR-Nano-2512

  # Chunking
  - CHUNK_DURATION_MS=15000
  - CHUNK_OVERLAP_MS=1000

  # VAD (Voice Activity Detection)
  - VAD_THRESHOLD=0.6
  - VAD_MIN_SILENCE_MS=300

  # Transcription
  - MAX_NEW_TOKENS=500
  - REQUEST_TIMEOUT=600
  - MAX_AUDIO_SIZE_MB=500

  # Performance
  - TORCH_COMPILE=false
  - WARMUP=true

  # Logging
  - JSON_LOGGING=false
```

## 📊 Performance

### Benchmarks (CUDA GPU)

| Audio Duration | Processing Time | Speed Ratio |
|----------------|-----------------|-------------|
| 1 minute | ~8 seconds | 7.5x |
| 10 minutes | ~75 seconds | 8x |
| 14 minutes (video) | ~110 seconds | 7.6x |

*Tested with WebM video, Chinese language, 5-minute chunks*

### Optimization Tips

1. **Enable torch.compile** for ~20% speed boost (may increase memory)
2. **Adjust chunk size** based on your audio length
3. **Use GPU** for 10-20x speedup vs CPU
4. **Batch process** multiple files for efficiency

## 🛠️ Development

### Running Tests

```bash
# Test CLI
./test_cli.sh

# Test with actual audio
make transcribe INPUT=/app/data/test-audio.mp3

# View logs
docker compose logs -f glm-asr
```

### Building for Development

```bash
# Build with no cache
docker compose build --no-cache

# Run with shell access
docker compose run --rm glm-asr /bin/bash

# Test Python syntax
python3 -m py_compile glm_asr_cli.py
```

## 🔍 Troubleshooting

### Container Issues

**Container won't start**
```bash
# Check logs
docker compose logs glm-asr

# Verify GPU
nvidia-smi

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

**Model loading is slow**
- First run downloads ~2GB model (normal)
- Model cached in `~/.cache/huggingface/`
- Subsequent runs load in ~10 seconds

### Transcription Issues

**Out of memory errors**
```bash
# Reduce chunk size in docker-compose.yml
- CHUNK_DURATION_MS=10000

# Monitor GPU
watch -n 1 nvidia-smi
```

**Inaccurate transcription**
- Ensure clear audio quality
- Try different language codes
- Check audio is 16kHz (server auto-converts)
- Longer audio is automatically chunked

### File Issues

**File not found**
```bash
# Ensure file is in data directory
ls -la data/

# Use correct path (/app/data/...)
make transcribe INPUT=/app/data/filename.mp3
```

**Unsupported format**
- The service supports all FFmpeg formats
- Test with: `ffmpeg -i yourfile.ext`
- Convert to common format if needed

## 📚 Documentation

- **[CLI_USAGE.md](docs/CLI_USAGE.md)** - Comprehensive CLI guide
- **[SUPPORTED_FORMATS.md](docs/SUPPORTED_FORMATS.md)** - All supported formats
- **[GLOBAL_CLI.md](docs/GLOBAL_CLI.md)** - Global CLI wrapper setup and usage


## 🌐 Frontend Options

For a complete speech-to-text experience:

- **[NeuralWhisper](https://github.com/lsj5031/NeuralWhisper)** - Modern web-based frontend with real-time capabilities
- **[WhisperSqueak](https://github.com/lsj5031/WhisperSqueak)** - Lightweight desktop application

Both frontends are fully compatible with this GLM-ASR server.

## 🤖 Model Information

Uses the [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) model from the [ZAI organization](https://huggingface.co/zai-org).

**Model Features:**
- 2512 parameters (Nano variant)
- Multimodal speech recognition
- State-of-the-art accuracy
- Efficient inference with bfloat16

## 🏆 Docker Best Practices

The Dockerfile implements:
- ✅ Multi-stage builds for smaller images
- ✅ Non-root user (appuser, UID 1000)
- ✅ Health checks for orchestration
- ✅ Official PyTorch base with CUDA/cuDNN
- ✅ Proper layer caching for fast rebuilds
- ✅ Minimal attack surface

## 🤝 Contributing

Contributions welcome! We especially value:
- Performance optimizations
- Documentation improvements
- Additional format support
- Testing enhancements
- Docker optimizations

## 📜 Acknowledgments

Built upon excellent work by:
- **GLM-ASR** - Speech recognition model by [ZAI Organization](https://huggingface.co/zai-org)
- **faster-whisper-server** - Architecture by [Fedir Zadniprovskyi](https://github.com/fedirz/faster-whisper-server)
- **FastAPI** - Python web framework
- **HuggingFace** - Transformers library and model hub
- **PyTorch** - Deep learning infrastructure

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 📖 Citation

```bibtex
@misc{glm-asr,
  title={GLM-ASR: Global Large-scale Multimodal Model for Automatic Speech Recognition},
  author={ZAI Organization},
  year={2024},
  url={https://huggingface.co/zai-org}
}
```

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐️

---

**Made with ❤️ by the GLM-ASR community**
