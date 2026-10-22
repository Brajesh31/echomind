"""EchoMind — Audio Ingestion & Extraction Processor.

Handles:
- YouTube audio extraction with SSRF protections and real metadata capture.
- System dependency validation (FFmpeg detection).
- Audio conversion to 16kHz mono WAV for Whisper.
- Audio splitting into manageable chunks with exact duration tracking.
- Safe temporary workspace isolation and cleanup.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from pydub import AudioSegment
import yt_dlp

from backend.config import settings
from backend.services.security import validate_video_url

logger = logging.getLogger("echomind.audio_processor")


def is_ffmpeg_available() -> bool:
    """Check whether ffmpeg executable is discovered in system PATH."""
    return shutil.which("ffmpeg") is not None


def ensure_ffmpeg():
    """Raise a descriptive error if ffmpeg is not available in the runtime environment."""
    if not is_ffmpeg_available():
        raise RuntimeError(
            "System dependency 'ffmpeg' is required for audio conversion and extraction, "
            "but was not found in PATH. Please install ffmpeg or run EchoMind via Docker."
        )


def get_download_dir() -> Path:
    """Return configured download directory and ensure it exists."""
    path = settings.absolute_data_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract standard 11-character YouTube video ID from various URL formats."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if "youtu.be" in hostname:
            return parsed.path.strip("/").split("/")[0]

        if "youtube.com" in hostname:
            if parsed.path == "/watch":
                qs = parse_qs(parsed.query)
                return qs.get("v", [None])[0]
            if parsed.path.startswith("/embed/"):
                return parsed.path.split("/")[2]
            if parsed.path.startswith("/v/"):
                return parsed.path.split("/")[2]
            if parsed.path.startswith("/shorts/"):
                return parsed.path.split("/")[2]
            # Pattern match within youtube.com URL
            match = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.debug("Failed parsing video ID from url '%s': %s", url, e)
        return None


def extract_video_metadata(url: str) -> Dict[str, Any]:
    """Fetch video metadata from YouTube without downloading media streams."""
    validated_url = validate_video_url(url)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(validated_url, download=False)
            if not info:
                raise ValueError("Could not retrieve video information from the provided URL.")

            duration = info.get("duration")
            return {
                "video_id": info.get("id") or extract_youtube_video_id(url) or uuid.uuid4().hex[:11],
                "source_url": validated_url,
                "title": info.get("title") or "Untitled Video",
                "duration": float(duration) if duration is not None else None,
                "thumbnail": info.get("thumbnail"),
                "author": info.get("uploader") or info.get("channel"),
                "upload_date": str(info.get("upload_date") or ""),
                "source_platform": "youtube",
            }
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).lower()
        if "private video" in msg:
            raise ValueError("The requested YouTube video is private.") from exc
        if "video unavailable" in msg or "deleted" in msg or "not available" in msg:
            raise ValueError("The requested YouTube video is unavailable or has been removed.") from exc
        if "sign in" in msg or "bot" in msg:
            raise ValueError("YouTube requested authentication or bot verification for this video.") from exc
        raise RuntimeError(f"Failed to fetch video information: {exc}") from exc
    except Exception as exc:
        logger.error("Error extracting video metadata for %s: %s", validated_url, exc)
        raise


def download_youtube_audio(url: str, session_dir: Path | str) -> Tuple[str, Dict[str, Any]]:
    """Download audio from a validated YouTube URL into the session directory.

    Returns:
        Tuple of (path_to_wav_file, video_metadata_dict)
    """
    ensure_ffmpeg()
    validated_url = validate_video_url(url)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    output_tmpl = str(session_dir / "audio_source.%(ext)s")
    expected_wav = str(session_dir / "audio_source.wav")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_tmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    logger.info("Downloading audio from YouTube: %s", validated_url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(validated_url, download=True)
            if not info:
                raise ValueError("Failed to retrieve audio stream from YouTube.")

            duration = info.get("duration")
            if duration and duration > (settings.max_video_duration_minutes * 60):
                raise ValueError(
                    f"Video duration ({duration / 60:.1f} minutes) exceeds the maximum allowed limit "
                    f"of {settings.max_video_duration_minutes} minutes."
                )

            metadata = {
                "video_id": info.get("id") or extract_youtube_video_id(url) or uuid.uuid4().hex[:11],
                "source_url": validated_url,
                "title": info.get("title") or "Untitled Video",
                "duration": float(duration) if duration is not None else None,
                "thumbnail": info.get("thumbnail"),
                "author": info.get("uploader") or info.get("channel"),
                "upload_date": str(info.get("upload_date") or ""),
                "source_platform": "youtube",
            }

            prep_file = ydl.prepare_filename(info)
            wav_file = os.path.splitext(prep_file)[0] + ".wav"
            if os.path.exists(wav_file):
                return wav_file, metadata
            if os.path.exists(expected_wav):
                return expected_wav, metadata
            for f in session_dir.glob("*.wav"):
                return str(f), metadata

            raise RuntimeError("Audio was downloaded but WAV file could not be located on disk.")
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).lower()
        if "private video" in msg:
            raise ValueError("The requested YouTube video is private.") from exc
        if "video unavailable" in msg or "deleted" in msg:
            raise ValueError("The requested YouTube video is unavailable or has been removed.") from exc
        raise RuntimeError(f"YouTube audio download failed: {exc}") from exc
    except Exception as exc:
        logger.error("YouTube download failed for %s: %s", validated_url, exc)
        raise


def convert_to_wav(input_path: str, out_dir: Path | str) -> Tuple[str, Dict[str, Any]]:
    """Convert an uploaded audio or video file to 16kHz mono WAV format."""
    ensure_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input media file not found: {input_path}")

    filename = os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    output_path = str(out_dir / filename)

    logger.info("Converting %s to 16kHz mono WAV at %s", input_path, output_path)
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        duration_seconds = len(audio) / 1000.0

        metadata = {
            "video_id": uuid.uuid4().hex[:11],
            "source_url": os.path.basename(input_path),
            "title": os.path.splitext(os.path.basename(input_path))[0],
            "duration": round(duration_seconds, 2),
            "thumbnail": None,
            "author": "Local Upload",
            "upload_date": "",
            "source_platform": "file",
        }
        return output_path, metadata
    except Exception as exc:
        logger.error("Audio conversion failed for %s: %s", input_path, exc)
        raise RuntimeError(f"Failed to process media file: {exc}") from exc


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> List[Tuple[str, float]]:
    """Split a WAV file into fixed-duration chunks with exact duration tracking.

    Returns:
        List of tuples: (chunk_file_path, duration_seconds)
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found for chunking: {wav_path}")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks: List[Tuple[str, float]] = []

    try:
        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start: start + chunk_ms]
            chunk_path = f"{wav_path}_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunks.append((chunk_path, len(chunk) / 1000.0))
    finally:
        # Guarantee removal of intermediate raw full WAV file
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError as e:
                logger.warning("Could not remove intermediate WAV file %s: %s", wav_path, e)

    return chunks


def process_input(source: str, session_id: Optional[str] = None) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Ingest media source, convert to WAV, split into chunks, and return (chunk_tuples, metadata)."""
    session_id = session_id or uuid.uuid4().hex
    session_dir = get_download_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Ingesting video URL for session %s", session_id)
        wav_path, metadata = download_youtube_audio(source, session_dir)
    else:
        logger.info("Ingesting local media file for session %s", session_id)
        wav_path, metadata = convert_to_wav(source, session_dir)

    logger.info("Chunking audio for transcription...")
    chunks = chunk_audio(wav_path)
    logger.info("Generated %d chunk(s) for transcription (video duration: %s s).", len(chunks), metadata.get("duration"))
    return chunks, metadata
