# GLM-ASR CLI Quick Reference Guide

## Quick Start

1. **Start the service:**
   ```bash
   docker compose up -d
   # or
   make up
   ```

2. **Place audio files in data directory:**
   ```bash
   mkdir -p data
   cp your-audio.mp3 data/
   # Or any other supported format:
   # cp your-audio.wav data/
   # cp your-audio.m4a data/
   # cp video.mp4 data/  # Yes, video files work too!
   ```

3. **Transcribe:**
   ```bash
   # Method 1: Using Makefile (easiest)
   make transcribe INPUT=/app/data/your-audio.mp3

   # Method 2: Using docker compose exec
   docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/your-audio.mp3

   # Method 3: Using global CLI wrapper
   glm-asr transcribe data/your-audio.mp3
   ```

## Supported Audio Formats

The service supports **all audio and video formats** that ffmpeg can handle, including:

**Audio Formats:**
- MP3, WAV, FLAC, OGG, WMA, AAC
- M4A, OPUS, AIFF, AU, RA, AMR
- And many more...

**Video Formats** (audio track is extracted):
- MP4, M4V, MOV, AVI, MKV, WEBM
- WMV, MPG, MPEG, FLV, 3GP
- And more...

The server uses **pydub** with **ffmpeg** for automatic format conversion, so virtually any audio/video file will work!

## Common Commands

### Basic Transcription
```bash
# Single file
make transcribe INPUT=/app/data/input.mp3

# With output file
make transcribe INPUT=/app/data/input.mp3 OUTPUT=/app/data/output.txt

# With specific language
make transcribe INPUT=/app/data/input.mp3 LANGUAGE=en

# To SRT format
make transcribe INPUT=/app/data/input.mp3 FORMAT=srt
```

### Batch Processing
```bash
# Process all files in directory
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/
```

### Server Management
```bash
# Check server health
make health
# or
docker compose exec glm-asr python glm_asr_cli.py health

# View logs
make logs

# Restart server
make restart

# Stop server
make clean
```

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output file path | Auto-generated |
| `--server-url` | `-s` | Server URL | http://localhost:8000 |
| `--language` | `-l` | Language code | auto |
| `--format` | `-f` | Output format (text/srt/json/verbose_json) | text |

## File Locations

| Location | Purpose |
|----------|---------|
| `./data/` | Place input audio/video files here |
| `/app/data/` | Path inside container |
| `./data/*.txt` | Output transcripts (auto-named) |
| `./data/*.srt` | Output SRT files |

## Examples

### Transcribe podcast episode
```bash
make transcribe INPUT=/app/data/podcast.mp3 FORMAT=srt
# or
make transcribe INPUT=/app/data/podcast.m4a FORMAT=srt
```

### Extract audio from video
```bash
# Video files work automatically - audio track is extracted
make transcribe INPUT=/app/data/interview.mp4
make transcribe INPUT=/app/data/lecture.mkv FORMAT=srt
```

### Batch process multiple files
```bash
# Place all files in data directory
cp ~/Downloads/*.mp3 data/
cp ~/Downloads/*.m4a data/
cp ~/Videos/*.mp4 data/

# Process all at once
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/
```

### Custom language and output
```bash
make transcribe INPUT=/app/data/speech.mp3 LANGUAGE=en OUTPUT=/app/data/transcript.txt
make transcribe INPUT=/app/data/speech.wav LANGUAGE=zh OUTPUT=/app/data/transcript.txt
```

### Long audio file (1 hour+)
```bash
# Use larger chunks for efficiency
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/long-audio.mp3 -c 10
docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/audiobook.m4b -c 15
```

## Troubleshooting

### Container not running
```bash
# Check status
docker compose ps

# Start if not running
docker compose up -d
```

### File not found
```bash
# Ensure file is in data directory
ls -la data/

# Use correct path (/app/data/...)
make transcribe INPUT=/app/data/filename.mp3
```

### Server not healthy
```bash
# Check health
make health

# View logs
make logs

# Restart
make restart
```

### Permission denied
```bash
# Ensure data directory exists and is writable
mkdir -p data
chmod 755 data
```

## Advanced Usage

### Direct API access
```bash
# Using curl
curl -X POST "http://localhost:18000/v1/audio/transcriptions" \
  -F "file=@data/input.mp3"

# Check API docs
open http://localhost:18000/docs
```

### WebSocket streaming
```bash
# For real-time transcription, use WebSocket endpoint
# See API documentation at http://localhost:18000/docs
```

## Tips

1. **File organization**: Create subdirectories in `data/` for organized output
   ```bash
   mkdir -p data/podcasts data/meetings
   ```

2. **Batch processing**: Use shell wildcards for multiple files
   ```bash
   cp ~/audio/*.mp3 data/
   docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/
   ```

3. **Output formats**: Specify SRT for subtitles
   ```bash
   make transcribe INPUT=/app/data/video.mp4 FORMAT=srt
   ```

4. **Language support**: Use ISO 639-1 codes
   ```bash
   make transcribe INPUT=/app/data/speech.mp3 LANGUAGE=zh  # Chinese
   make transcribe INPUT=/app/data/speech.mp3 LANGUAGE=es  # Spanish
   ```

## Support

- GitHub Issues: https://github.com/lsj5031/glm-asr-docker/issues
- API Documentation: http://localhost:18000/docs (when running)
- Server Health: http://localhost:18000/health
