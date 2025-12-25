# GLM-ASR

A FastAPI-based speech-to-text service powered by the GLM-ASR-Nano model. Transcribe audio files with ease using this OpenAI-compatible API.

Inspired by the architecture of [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) by Fedir Zadniprovskyi.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's audio transcription endpoint
- **Multi-format support**: Handles various audio formats via FFmpeg
- **GPU acceleration**: CUDA support for fast inference
- **Streaming ready**: FastAPI-based architecture for easy extension
- **Docker support**: Production-ready containerized deployment
- **Health checks**: Built-in `/health` endpoint for container orchestration
- **Graceful shutdown**: Proper SIGTERM handling for clean container termination
- **Flexible configuration**: Environment variables for port, model selection, and logging

## Requirements

- Python 3.10+
- CUDA 12.8 (optional, for GPU acceleration)
- FFmpeg
- libsndfile (for audio processing)
- Docker & Docker Compose (for containerized setup)

## Installation

### Quick Start (Docker)

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/lsj5031/glm-asr-docker.git
cd glm-asr-docker

# Start the service with GPU support
docker-compose up
```

The API will be available at `http://localhost:8000`

### Using Make Commands

With the Makefile, you can use convenient commands:

```bash
# Build the Docker image
make build

# Run with docker-compose
make run

# View logs
make logs

# Open a shell in the container
make shell

# Clean up (stop and remove containers)
make clean

# Show all available commands
make help
```

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/lsj5031/glm-asr-docker.git
cd glm-asr-docker
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install FFmpeg and libsndfile:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile

# Windows (with conda)
conda install ffmpeg libsndfile
```

5. Run the server:
```bash
python server.py
```

The API will be available at `http://localhost:8000`

### Docker Setup

#### Using Pre-built Image

Pull the latest image from GitHub Container Registry:
```bash
docker pull ghcr.io/lsj5031/glm-asr-docker:latest
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  ghcr.io/lsj5031/glm-asr-docker:latest
```

#### Build Locally

Build and run with Docker:
```bash
docker build -t glm-asr .
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  glm-asr
```

#### Using Docker Compose

The simplest method with GPU support:
```bash
docker-compose up
```

## Configuration

Configure the service using environment variables. See `.env.example` for all options:

```env
# Server port (default: 8000)
PORT=8000

# Model ID from Hugging Face (default: zai-org/GLM-ASR-Nano-2512)
MODEL_ID=zai-org/GLM-ASR-Nano-2512

# Enable JSON logging (default: false)
JSON_LOGGING=false

# Max new tokens for generation (default: 500)
MAX_NEW_TOKENS=500
```

### Setting Environment Variables

#### With Docker
```bash
docker run --gpus all -p 8000:8000 \
  -e PORT=8000 \
  -e MODEL_ID=zai-org/GLM-ASR-Nano-2512 \
  -e JSON_LOGGING=true \
  glm-asr
```

#### With Docker Compose
Edit `docker-compose.yml` and update the `environment` section:
```yaml
environment:
  - PORT=8000
  - MODEL_ID=zai-org/GLM-ASR-Nano-2512
  - JSON_LOGGING=false
```

#### Running Locally
```bash
export PORT=8000
export MODEL_ID=zai-org/GLM-ASR-Nano-2512
export JSON_LOGGING=false
python server.py
```

## Usage

### Health Check

Check if the service is running and healthy:

```bash
curl http://localhost:8000/health
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
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3"
```

Response:
```json
{
  "text": "Hello, this is a test transcription."
}
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

Response:
```json
{
  "data": [
    {
      "id": "glm-nano-2512",
      "object": "model"
    }
  ]
}
```

## API Documentation

Interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

You can also access Redoc documentation at `http://localhost:8000/redoc`

## Frontend Options

For a complete speech-to-text experience, you can use these frontend applications that are compatible with this GLM-ASR server:

- **[NeuralWhisper](https://github.com/lsj5031/NeuralWhisper)** - A modern web-based frontend for speech transcription with real-time capabilities
- **[WhisperSqueak](https://github.com/lsj5031/WhisperSqueak)** - A lightweight desktop application for audio transcription

Both frontends are designed to work seamlessly with this GLM-ASR server's OpenAI-compatible API endpoints.

## Model

Uses the [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) model from the [ZAI organization](https://huggingface.co/zai-org), which provides efficient speech recognition with minimal computational overhead.

The GLM-ASR project is developed by the ZAI team and represents state-of-the-art multimodal speech recognition capabilities.

## Performance

- Input audio is resampled to 16kHz (optimal for the model)
- Supports up to 30-second chunks, automatically batched for longer audio
- Inference runs in bfloat16 precision for efficiency
- GPU acceleration significantly reduces inference time

### Typical Performance

- Short audio (< 30 seconds): ~2-5 seconds on GPU
- Long audio (> 1 minute): Processed in 30-second chunks sequentially
- Without GPU: ~10-30 seconds per chunk

## Troubleshooting

### Container won't start
- Check logs: `docker-compose logs glm-asr`
- Ensure GPU drivers are installed: `nvidia-smi`
- Verify Docker has GPU support: `docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi`

### Model loading is slow
- First run downloads the model (~2GB) - this is normal
- Model is cached in `~/.cache/huggingface/` for subsequent runs
- Mount this directory as a volume for persistent caching

### Out of memory errors
- Reduce batch size or use a smaller model
- Ensure sufficient GPU VRAM (minimum 4GB recommended)
- Monitor with: `nvidia-smi -l 1`

### Transcription is inaccurate
- Ensure audio is clear and at 16kHz sample rate
- Try with different audio preprocessing
- Audio longer than 60 seconds is chunked automatically

## Docker Best Practices

The Dockerfile uses:
- **Multi-stage builds**: Smaller final image size
- **Non-root user**: Enhanced security (appuser, UID 1000)
- **Health checks**: Automatic container health monitoring
- **Official PyTorch base**: Latest CUDA/cuDNN support
- **Proper layer caching**: Faster rebuilds

## Development

### Running Tests
```bash
# Run linting
make lint

# Run container smoke test
docker build -t glm-asr:test .
docker run --rm glm-asr:test python -c "from server import app; print('OK')"
```

### Building for Development
```bash
docker build -t glm-asr:dev .
docker run -it --rm --gpus all -p 8000:8000 glm-asr:dev
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

We especially welcome:
- Performance optimizations
- Documentation improvements
- Additional model support
- Testing enhancements
- Docker optimization ideas

## Acknowledgments

This project builds upon the excellent work of:

- **GLM-ASR** - The underlying speech recognition model by the ZAI organization ([zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512))
- **faster-whisper-server** - Inspired by [Fedir Zadniprovskyi's architecture](https://github.com/fedirz/faster-whisper-server) for OpenAI-compatible speech API servers
- **FastAPI** - For the excellent Python web framework
- **HuggingFace** - For the Transformers library and model hub
- **PyTorch** - For deep learning infrastructure

## License

MIT License - See LICENSE file for details

## Citation

If you use GLM-ASR in your research, please cite the original GLM-ASR model from ZAI organization:

```bibtex
@misc{glm-asr,
  title={GLM-ASR: Global Large-scale Multimodal Model for Automatic Speech Recognition},
  author={ZAI Organization},
  year={2024},
  url={https://huggingface.co/zai-org}
}
```
