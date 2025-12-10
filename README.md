# GLM-ASR

A FastAPI-based speech-to-text service powered by the GLM-ASR-Nano model. Transcribe audio files with ease using this OpenAI-compatible API.

Inspired by the architecture of [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) by Fedir Zadniprovskyi.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's audio transcription endpoint
- **Multi-format support**: Handles various audio formats via FFmpeg
- **GPU acceleration**: CUDA support for fast inference
- **Streaming ready**: FastAPI-based architecture for easy extension
- **Docker support**: Production-ready containerized deployment

## Requirements

- Python 3.10+
- CUDA 12.1 (optional, for GPU acceleration)
- FFmpeg

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/lsj5031/glm-asr-docker.git
cd glm-asr
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

4. Run the server:
```bash
python server.py
```

The API will be available at `http://localhost:8000`

### Docker Setup

#### Using Pre-built Image

Pull the latest image from GitHub Container Registry:
```bash
docker pull ghcr.io/lsj5031/glm-asr-docker:latest
docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface ghcr.io/lsj5031/glm-asr-docker:latest
```

#### Build Locally

Build and run with Docker:
```bash
docker build -t glm-asr .
docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface glm-asr
```

## Usage

### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3"
```

### List Available Models

```bash
curl "http://localhost:8000/v1/models"
```

Returns:
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

## Model

Uses the [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) model from the [ZAI organization](https://huggingface.co/zai-org), which provides efficient speech recognition with minimal computational overhead.

The GLM-ASR project is developed by the ZAI team and represents state-of-the-art multimodal speech recognition capabilities.

## Performance

- Input audio is resampled to 16kHz (optimal for the model)
- Supports up to 30-second chunks, automatically batched for longer audio
- Inference runs in bfloat16 precision for efficiency

## Acknowledgments

This project builds upon the excellent work of:

- **GLM-ASR** - The underlying speech recognition model by the ZAI organization ([zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512))
- **faster-whisper-server** - Inspired by [Fedir Zadniprovskyi's architecture](https://github.com/fedirz/faster-whisper-server) for OpenAI-compatible speech API servers
- **FastAPI** - For the excellent Python web framework
- **HuggingFace** - For the Transformers library and model hub

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome. Please feel free to submit a pull request.

We especially welcome enhancements to the Dockerfile to make it smaller and more modern. If you have ideas for optimizing the Docker image (multi-stage builds, better layer caching, Alpine Linux compatibility, etc.), we'd love to see your contributions.

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
