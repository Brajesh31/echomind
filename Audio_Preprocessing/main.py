"""EchoMind — End-to-End Media Processing Pipeline.

Coordinates:
- Ingestion & WAV conversion with real metadata extraction (Audio Processor)
- Whisper transcription & segment timing (Transcriber)
- Segment-aware vector indexing with timestamp preservation (Vector Store)
- Summarization, title, and chapter derivation (LLM Pipeline)
- Session persistence & idempotency (Session Store)
- Progress tracking for asynchronous background jobs
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.models.schemas import JobStatus
from backend.models.transcript import StructuredTranscript, TranscriptSegment
from backend.services.session_store import session_store
from .audio_processor import process_input
from .llm_pipeline import generate_chapters, summarize_title
from .transcriber import load_model, transcribe_all
from .vector_store import build_rag_chain, get_embeddings

logger = logging.getLogger("echomind.pipeline")


def warm_up():
    """Pre-warm heavy machine learning models at server startup."""
    logger.info("Warming up models...")
    try:
        load_model()
        get_embeddings()
        logger.info("Models ready.")
    except Exception as exc:
        logger.warning("Model warm-up deferred or partially completed: %s", exc)


def preprocessing(
    source: Optional[str] = None,
    transcript: Optional[str] = None,
    language: str = "english",
    progress_callback: Optional[Callable[[JobStatus, str, str, int], None]] = None,
) -> Tuple[str, List[TranscriptSegment], Dict[str, Any], str]:
    """Convert input into raw text, structured timestamped segments, and metadata."""
    if transcript is not None:
        structured = StructuredTranscript.from_raw_text(transcript, language=language)
        metadata = {
            "video_id": "text_input",
            "source_url": "plain_text",
            "title": "Uploaded Transcript",
            "duration": None,
            "thumbnail": None,
            "author": "Text Upload",
            "source_platform": "text",
        }
        return structured.full_text, structured.segments, metadata, language

    if source is None:
        raise ValueError("Either 'source' (URL or file) or 'transcript' must be provided.")

    if progress_callback:
        progress_callback(JobStatus.DOWNLOADING, "downloading", "Downloading media stream...", 15)

    chunks, metadata = process_input(source)
    session_dir = os.path.dirname(chunks[0][0]) if chunks else None

    if progress_callback:
        progress_callback(JobStatus.TRANSCRIBING, "transcribing", f"Transcribing {len(chunks)} audio chunk(s) with Whisper...", 40)

    try:
        full_text, segments, detected_language = transcribe_all(chunks, language=language)
        return full_text, segments, metadata, detected_language
    finally:
        # Guarantee removal of temporary session download directory
        if session_dir and os.path.isdir(session_dir):
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except OSError as e:
                logger.warning("Could not clean session directory %s: %s", session_dir, e)


def run_pipeline(
    source: Optional[str] = None,
    transcript: Optional[str] = None,
    user_id: str = "default",
    append: bool = False,
    language: str = "english",
    summary_type: str = "comprehensive",
    progress_callback: Optional[Callable[[JobStatus, str, str, int], None]] = None,
) -> Dict[str, Any]:
    """Execute complete ingestion, transcription, indexing, and summarization pipeline."""
    user_id = user_id or "default"
    existing_session = session_store.get(user_id)

    # Idempotent check: if session exists, has transcript, same source, and not append
    if existing_session and not append and source:
        existing_meta = existing_session.get("metadata", {})
        if existing_meta.get("source_url") == source and existing_session.get("transcript"):
            logger.info("Idempotent hit: returning existing session data for '%s' (source=%s)", user_id, source)
            if progress_callback:
                progress_callback(JobStatus.COMPLETED, "ready", "Loaded existing processed video session.", 100)
            return existing_session

    if append:
        if not existing_session:
            raise ValueError(f"Session '{user_id}' does not exist to append to.")

        new_text, new_segments, metadata, detected_lang = preprocessing(
            source=source,
            transcript=transcript,
            language=language,
            progress_callback=progress_callback,
        )
        old_text = existing_session.get("transcript", "")
        full_text = f"{old_text}\n\n{new_text}".strip()

        old_segments = existing_session.get("segments", [])
        combined_segments = old_segments + [s.model_dump() for s in new_segments]
    else:
        new_text, new_segments, metadata, detected_lang = preprocessing(
            source=source,
            transcript=transcript,
            language=language,
            progress_callback=progress_callback,
        )
        full_text = new_text
        combined_segments = [s.model_dump() for s in new_segments]

    if progress_callback:
        progress_callback(JobStatus.PROCESSING_TRANSCRIPT, "processing_transcript", "Generating structured summary & chapter markers...", 75)

    # Run summarization, chaptering, and RAG indexing in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        summary_future = executor.submit(summarize_title, full_text, summary_type)
        chapters_future = executor.submit(generate_chapters, combined_segments)
        rag_future = executor.submit(build_rag_chain, new_text, user_id, append, new_segments)

    title_summary = summary_future.result()
    chapters = chapters_future.result()
    rag_chain_ok = rag_future.result()

    if progress_callback:
        progress_callback(JobStatus.INDEXING, "indexing", "Finalizing session state and semantic index...", 90)

    # Extract clean title from title_summary
    derived_title = metadata.get("title")
    if not derived_title or derived_title == "Untitled Video":
        lines = [line.strip() for line in title_summary.split("\n") if line.strip()]
        if lines:
            derived_title = lines[0].replace("#", "").strip()

    # Save to atomic session store
    session_data = {
        "user_id": user_id,
        "title": derived_title,
        "summary": title_summary,
        "transcript": full_text,
        "segments": combined_segments,
        "metadata": metadata,
        "chapters": chapters,
        "language": detected_lang,
        "segment_count": len(combined_segments),
        "duration_seconds": metadata.get("duration"),
    }
    saved_session = session_store.save(user_id, session_data)

    if progress_callback:
        progress_callback(JobStatus.COMPLETED, "ready", "Video intelligence processing complete.", 100)

    return saved_session


# Expose user_ids dictionary proxy and save_session for backwards compatibility
user_ids = session_store.list_all()


def save_session():
    """Backwards compatibility helper."""
    pass