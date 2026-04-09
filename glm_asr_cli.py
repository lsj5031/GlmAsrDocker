#!/usr/bin/env python3
"""GLM-ASR CLI - Thin client for the GLM-ASR transcription server."""

import argparse
import sys
from pathlib import Path

import httpx

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".wma", ".aac",
    ".opus", ".aiff", ".aif", ".au", ".amr", ".3gp", ".webm",
    ".mkv", ".avi", ".mov", ".wmv", ".mpg", ".mpeg", ".flv",
}


def transcribe_file(
    audio_path: str,
    output_path: str,
    server_url: str,
    language: str,
    response_format: str,
) -> str:
    """Send an audio file to the server and return the transcript."""
    print(f"Transcribing: {audio_path}")
    with open(audio_path, "rb") as f:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{server_url}/v1/audio/transcriptions",
                files={"file": (Path(audio_path).name, f)},
                data={"language": language, "response_format": response_format},
            )
            response.raise_for_status()

    if response_format in ("json", "verbose_json"):
        text = response.json()["text"]
    elif response_format == "srt":
        text = response.text
    else:
        text = response.json()["text"]

    Path(output_path).write_text(text, encoding="utf-8")
    print(f"Saved to: {output_path}")
    return text


def cmd_health(args: argparse.Namespace) -> int:
    """Check server health."""
    try:
        with httpx.Client(timeout=10.0) as client:
            data = client.get(f"{args.server_url}/health").raise_for_status().json()
        print(f"Status: {data.get('status', 'unknown')}")
        print(f"Model loaded: {data.get('model_loaded', False)}")
        print(f"Device: {data.get('device', 'unknown')}")
        return 0 if data.get("status") == "healthy" else 1
    except httpx.HTTPError as e:
        print(f"Error: Server is not reachable: {e}", file=sys.stderr)
        return 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio file(s)."""
    # Collect input paths
    input_paths = []
    for p in args.input:
        path = Path(p)
        if path.is_dir():
            input_paths.extend(path.iterdir())
        else:
            input_paths.append(path)

    input_paths = [p for p in input_paths if p.suffix.lower() in AUDIO_EXTENSIONS]
    if not input_paths:
        print("Error: No audio files found", file=sys.stderr)
        return 1

    print(f"Found {len(input_paths)} audio file(s)")

    ext = ".srt" if args.format == "srt" else ".txt"
    exit_code = 0
    for i, input_path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] {input_path}")
        output_path = args.output if args.output and len(input_paths) == 1 else str(input_path.with_suffix(ext))
        try:
            transcribe_file(str(input_path), output_path, args.server_url, args.language, args.format)
            print(f"✓ {input_path.name}")
        except Exception as e:
            print(f"✗ {input_path.name} - {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(prog="glm-asr", description="GLM-ASR CLI")
    sub = parser.add_subparsers(dest="command")

    hp = sub.add_parser("health", help="Check server health")
    hp.add_argument("--server-url", default="http://localhost:8000")
    hp.set_defaults(func=cmd_health)

    tp = sub.add_parser("transcribe", help="Transcribe audio file(s)")
    tp.add_argument("input", nargs="+", help="Audio file(s) or directory")
    tp.add_argument("-o", "--output", help="Output file path")
    tp.add_argument("-s", "--server-url", default="http://localhost:8000")
    tp.add_argument("-l", "--language", default="auto")
    tp.add_argument("-f", "--format", choices=["text", "srt", "json", "verbose_json"], default="text")
    tp.set_defaults(func=cmd_transcribe)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
