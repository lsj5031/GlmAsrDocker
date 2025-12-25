# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.1.0] - 2025-12-26

### Added
- Native implementation of `GlmAsrForConditionalGeneration` replacing generic AutoModel
- `AutoProcessor` integration for correct feature extraction
- `MAX_NEW_TOKENS` environment variable to control generation length
- `soundfile` and `numpy` based audio processing pipeline
- `libsndfile1` system dependency in Dockerfile

### Changed
- Refactored audio processing to be simpler and more robust
- Updated dependencies to use latest transformers from source
- Improved audio resampling logic
- Replaced `torchaudio` loading with `soundfile` for better compatibility

### Removed
- `WhisperFeatureExtractor` dependency (incorrect for GLM-ASR)
- Complex custom audio token length calculation logic
- Generic `AutoModelForCausalLM` usage

### Infrastructure

### Added
- Multi-stage Docker build for smaller image size
- PyTorch base image (2.8.0-cuda12.8-cudnn9-runtime)
- `HEALTHCHECK` directive in Dockerfile
- `/health` endpoint for container orchestration
- Graceful shutdown (SIGTERM) handling
- Structured JSON logging support via `JSON_LOGGING` environment variable
- `PORT` environment variable for configurable server port
- Makefile with development commands (`build`, `run`, `shell`, `logs`, `clean`)
- `docker-compose.yml` for easy one-command startup with GPU support
- `.dockerignore` for optimized Docker builds
- `.env.example` for configuration documentation
- CI/CD workflows:
  - `build.yml`: Build and push to GHCR on tags
  - `test.yml`: Lint, import tests, container smoke tests
- CHANGELOG.md for version history tracking

### Changed
- Upgraded PyTorch CUDA version from 12.1 to 12.8
- Updated Dockerfile to use official PyTorch base image
- Improved logging with optional JSON format
- Server now respects `PORT` environment variable
- Better health check mechanism in Docker

### Improved
- Docker image build performance with multi-stage builds
- Development workflow with Makefile
- Production deployment readiness
- Container monitoring and orchestration support

## [v1.0.0] - Initial Release

### Added
- FastAPI-based speech-to-text service
- GLM-ASR-Nano model integration
- OpenAI-compatible transcription API
- Multi-format audio support (MP3, WAV, etc.)
- GPU acceleration with CUDA
- Docker support
- Comprehensive logging
- Audio chunking for long files

### Features
- `/v1/audio/transcriptions` endpoint for audio transcription
- `/v1/models` endpoint to list available models
- Support for various audio formats via FFmpeg
- Automatic audio resampling to 16kHz
- Graceful handling of long audio files
