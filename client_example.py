#!/usr/bin/env python3
"""
Example client for GLM-ASR server with automatic chunking for long audio.

Usage:
    python client_example.py input.mp3 --output transcript.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import httpx
from pydub import AudioSegment
from tqdm import tqdm


def chunk_audio_file(
    audio_path: str, chunk_duration_minutes: int = 5
) -> list[AudioSegment]:
    """Split audio file into manageable chunks.
    
    Args:
        audio_path: Path to audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        
    Returns:
        List of AudioSegment chunks
    """
    audio = AudioSegment.from_file(audio_path)
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    
    chunks = []
    for start_ms in range(0, len(audio), chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, len(audio))
        chunks.append(audio[start_ms:end_ms])
    
    return chunks


def transcribe_chunk(
    chunk: AudioSegment, 
    server_url: str = "http://localhost:18000",
    language: str = "auto",
    timeout: float = 600.0,
) -> str:
    """Transcribe a single audio chunk.
    
    Args:
        chunk: AudioSegment to transcribe
        server_url: Base URL of GLM-ASR server
        language: Language code or 'auto'
        timeout: Request timeout in seconds
        
    Returns:
        Transcribed text
    """
    # Export chunk to WAV bytes
    from io import BytesIO
    buffer = BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    
    # Send to server
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{server_url}/v1/audio/transcriptions",
            files={"file": ("chunk.wav", buffer, "audio/wav")},
            data={"language": language, "response_format": "text"},
        )
        response.raise_for_status()
        return response.json()["text"]


def transcribe_file(
    audio_path: str,
    output_path: Optional[str] = None,
    server_url: str = "http://localhost:18000",
    language: str = "auto",
    chunk_duration_minutes: int = 5,
) -> str:
    """Transcribe audio file with automatic chunking.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save transcript
        server_url: Base URL of GLM-ASR server
        language: Language code or 'auto'
        chunk_duration_minutes: Size of chunks to split long audio
        
    Returns:
        Complete transcript
    """
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    duration_minutes = len(audio) / (1000 * 60)
    
    print(f"Duration: {duration_minutes:.2f} minutes")
    
    # Check if chunking is needed
    if duration_minutes <= chunk_duration_minutes:
        print("Processing as single file...")
        chunks = [audio]
    else:
        print(f"Splitting into {chunk_duration_minutes}-minute chunks...")
        chunks = chunk_audio_file(audio_path, chunk_duration_minutes)
        print(f"Created {len(chunks)} chunks")
    
    # Process chunks with progress bar
    transcripts = []
    with tqdm(total=len(chunks), desc="Transcribing", unit="chunk") as pbar:
        for i, chunk in enumerate(chunks, 1):
            chunk_duration = len(chunk) / 1000  # seconds
            pbar.set_postfix({"chunk": f"{i}/{len(chunks)}", "duration": f"{chunk_duration:.1f}s"})
            
            try:
                text = transcribe_chunk(chunk, server_url, language)
                transcripts.append(text)
                pbar.update(1)
            except httpx.HTTPError as e:
                print(f"\nError transcribing chunk {i}: {e}", file=sys.stderr)
                transcripts.append(f"[ERROR: Chunk {i} failed]")
                pbar.update(1)
    
    # Combine transcripts
    full_transcript = " ".join(transcripts)
    
    # Save if output path provided
    if output_path:
        Path(output_path).write_text(full_transcript, encoding="utf-8")
        print(f"\nTranscript saved to: {output_path}")
    
    return full_transcript


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using GLM-ASR server"
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-o", "--output", help="Output transcript file path")
    parser.add_argument(
        "-s", 
        "--server", 
        default="http://localhost:18000",
        help="Server URL (default: http://localhost:18000)"
    )
    parser.add_argument(
        "-l",
        "--language",
        default="auto",
        help="Language code (default: auto)"
    )
    parser.add_argument(
        "-c",
        "--chunk-minutes",
        type=int,
        default=5,
        help="Chunk duration in minutes for long audio (default: 5)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        transcript = transcribe_file(
            args.input,
            args.output,
            args.server,
            args.language,
            args.chunk_minutes,
        )
        
        if not args.output:
            print("\n=== TRANSCRIPT ===")
            print(transcript)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
