#!/usr/bin/env python3
"""
Example client for GLM-ASR server.

Usage:
    python client_example.py input.mp3
    python client_example.py input.mp3 --output transcript.txt
    python client_example.py input.mp3 -l en
"""

import argparse
import sys
from pathlib import Path

import httpx


def transcribe_file(
    audio_path: str,
    server_url: str = "http://localhost:18000",
    language: str = "auto",
) -> str:
    """Send an audio file to the server and return the transcript."""
    with open(audio_path, "rb") as f:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{server_url}/v1/audio/transcriptions",
                files={"file": (Path(audio_path).name, f)},
                data={"language": language, "response_format": "text"},
            )
            response.raise_for_status()
            return response.json()["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using GLM-ASR server"
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-o", "--output", help="Output transcript file path")
    parser.add_argument("-s", "--server", default="http://localhost:18000")
    parser.add_argument("-l", "--language", default="auto")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    transcript = transcribe_file(args.input, args.server, args.language)

    if args.output:
        Path(args.output).write_text(transcript, encoding="utf-8")
        print(f"Saved to: {args.output}")
    else:
        print(transcript)


if __name__ == "__main__":
    main()
