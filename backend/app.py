"""EchoMind — FastAPI Application.

Coordinates API endpoints, request validation, error handling,
background processing jobs, transcript search, and grounded RAG Q&A.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from Audio_Preprocessing.main import run_pipeline, warm_up
from Audio_Preprocessing.transcript_search import search_transcript
from Audio_Preprocessing.transcriber import _model, load_model
from Audio_Preprocessing.vector_store import (
    _embeddings,
    ask_questions,
    clear_session_cache,
    get_embeddings,
)
from backend.config import settings
from backend.models.schemas import (
    AskQuestionRequest,
    Chapter,
    EvidenceSource,
    HealthResponse,
    JobResponse,
    JobStatus,
    ProcessResponse,
    QuestionResponse,
    ReadyResponse,
    ResetSessionResponse,
    SessionInfo,
    SessionListResponse,
    SuggestedQuestionsResponse,
    TranscriptSearchResponse,
    TranscriptSearchResult,
    VideoMetadata,
    YoutubeProcessRequest,
)
from backend.services.job_manager import job_manager
from backend.services.security import safe_token, validate_video_url
from backend.services.session_store import session_store

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("echomind.api")

FRONTEND_INDEX = settings.base_dir / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for warm-up and graceful shutdown."""
    logger.info("Initializing %s v%s...", settings.app_name, settings.app_version)
    settings.ensure_directories()

    try:
        warm_up()
    except Exception as exc:
        logger.warning("Startup model warm-up encountered an issue: %s", exc)

    logger.info("%s ready to accept requests.", settings.app_name)
    yield
    logger.info("%s shutting down.", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Intelligent AI-Powered Video Assistant for Transcription, Summarization, and Grounded Q&A.",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add defense-in-depth HTTP security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    return response


# ── Global Exception Handlers ──────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Translate ValueErrors (e.g. invalid URLs, failed SSRF checks) to HTTP 400."""
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Translate RuntimeErrors (e.g. missing API keys or external pipeline failures)."""
    logger.error("Runtime error on %s: %s", request.url.path, exc)
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if "API key" in str(exc) or "unavailable" in str(exc).lower()
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler: logs the full exception internally, returns a clean message."""
    logger.exception("Unhandled server error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# ── Static / Root ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend application."""
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Frontend index.html not found."},
    )


# ── Health & Readiness ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Cheap and fast liveness check for Docker / load balancers."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        message=f"{settings.app_name} is operational.",
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """Readiness probe checking storage, model caches, and configuration."""
    whisper_ready = _model is not None
    embeddings_ready = _embeddings is not None
    mistral_ready = settings.is_mistral_configured
    storage_ready = (
        settings.absolute_data_path.exists()
        and settings.absolute_chroma_path.exists()
    )

    all_ready = whisper_ready and embeddings_ready and storage_ready
    return ReadyResponse(
        status="ready" if all_ready else "degraded",
        whisper_loaded=whisper_ready,
        embeddings_loaded=embeddings_ready,
        mistral_configured=mistral_ready,
        storage_ready=storage_ready,
        details={
            "whisper_model": settings.whisper_model,
            "embedding_model": settings.embedding_model,
            "environment": settings.environment,
        },
    )


# ── Asynchronous Background Jobs ───────────────────────────────────────────────
@app.post(
    "/api/jobs/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Jobs"],
    summary="Submit video processing job (Asynchronous)",
)
async def submit_video_job(data: YoutubeProcessRequest):
    """Submit a YouTube URL for asynchronous processing in the background."""
    validated_url = validate_video_url(data.url)
    session_id = safe_token(data.user_id)

    # Check for idempotent hit: if session already exists for this URL
    existing = session_store.get(session_id)
    if existing and not data.append:
        meta = existing.get("metadata", {})
        if meta.get("source_url") == validated_url and existing.get("transcript"):
            logger.info("Idempotent hit for URL %s in session %s", validated_url, session_id)
            job = job_manager.create_job(source=validated_url, session_id=session_id)
            job.complete(existing)
            return job.to_schema()

    return job_manager.submit_processing_task(
        source=validated_url,
        session_id=session_id,
        append=data.append,
    )


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Jobs"],
    summary="Get background processing job status",
)
async def get_job_status(job_id: str):
    """Check the real-time execution status and result of a background job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job.to_schema()


@app.get(
    "/api/jobs",
    response_model=List[JobResponse],
    tags=["Jobs"],
    summary="List all background jobs",
)
async def list_all_jobs():
    """List all tracked media processing jobs."""
    return [j.to_schema() for j in job_manager.list_jobs()]


# ── Media Processing Routes (Canonical & Synchronous Compatibility) ────────────
@app.post(
    "/youtube_process",
    response_model=Dict[str, Any],
    tags=["Processing"],
    summary="Process YouTube video (synchronous)",
)
@app.post(
    "/api/process/url",
    response_model=Dict[str, Any],
    tags=["Processing"],
    include_in_schema=False,
)
def process_youtube(data: YoutubeProcessRequest):
    """Ingest, transcribe, index, and summarize a YouTube video URL synchronously."""
    validated_url = validate_video_url(data.url)
    session_id = safe_token(data.user_id)

    logger.info("Starting synchronous YouTube processing for session '%s': %s", session_id, validated_url)
    session = run_pipeline(
        source=validated_url,
        transcript=None,
        user_id=session_id,
        append=data.append,
        language="english",
    )
    return {
        "summary": session["summary"],
        "session_id": session_id,
        "title": session.get("title"),
        "metadata": session.get("metadata"),
        "chapters": session.get("chapters", []),
    }


@app.post(
    "/process_text_audio",
    response_model=Dict[str, Any],
    tags=["Processing"],
    summary="Process uploaded media file (synchronous)",
)
@app.post(
    "/api/process/file",
    response_model=Dict[str, Any],
    tags=["Processing"],
    include_in_schema=False,
)
async def process_file(
    file: UploadFile = File(...),
    append_previous: bool = Form(False),
    user_id: str = Form("default"),
):
    """Process an uploaded audio/video file or plain-text transcript."""
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in settings.allowed_av_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(sorted(settings.allowed_av_extensions))}",
        )

    session_id = safe_token(user_id)

    # Fast path for raw transcript text files (.txt, .md)
    if ext in settings.transcript_extensions:
        raw_bytes = await file.read()
        text = raw_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded transcript file is empty.",
            )

        session = run_pipeline(
            source=None,
            transcript=text,
            user_id=session_id,
            append=append_previous,
            language="english",
        )
        return {
            "summary": session["summary"],
            "session_id": session_id,
            "title": session.get("title"),
            "chapters": session.get("chapters", []),
        }

    # Audio / Video upload path
    safe_name = safe_token(file.filename or "media")
    tmp_path = Path(tempfile.gettempdir()) / f"echomind_{session_id}_{uuid.uuid4().hex}_{safe_name}{ext}"

    logger.info("Receiving media upload '%s' for session '%s'", file.filename, session_id)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        session = run_pipeline(
            source=str(tmp_path),
            transcript=None,
            user_id=session_id,
            append=append_previous,
            language="english",
        )
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as e:
                logger.warning("Could not delete temporary upload file %s: %s", tmp_path, e)

    return {
        "summary": session["summary"],
        "session_id": session_id,
        "title": session.get("title"),
        "metadata": session.get("metadata"),
        "chapters": session.get("chapters", []),
    }


# ── Grounded Q&A Route ─────────────────────────────────────────────────────────
@app.post(
    "/question",
    response_model=QuestionResponse,
    tags=["Q&A"],
    summary="Ask a question against a video transcript with citations",
)
@app.post(
    "/api/question",
    response_model=QuestionResponse,
    tags=["Q&A"],
    include_in_schema=False,
)
async def ask_video_question(data: AskQuestionRequest):
    """Perform grounded semantic Q&A with timestamped evidence sources."""
    session_id = safe_token(data.user_id)
    cleaned_question = data.question.strip()

    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    session = session_store.get(session_id)
    if not session or not session.get("transcript"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Please ingest a video first.",
        )

    history = session.get("history", [])

    try:
        result = ask_questions(session_id, cleaned_question, history=history)

        # Update session conversation history (keep last 5 turns)
        history.append({"role": "user", "text": cleaned_question})
        history.append({"role": "assistant", "text": result["answer"]})
        session["history"] = history[-10:]
        session_store.save(session_id, session)

        raw_sources = result.get("sources", [])
        sources_out = [
            s if isinstance(s, EvidenceSource) else EvidenceSource(**s)
            for s in raw_sources
        ]

        return QuestionResponse(
            question=cleaned_question,
            answer=result["answer"],
            session_id=session_id,
            sources=sources_out,
            has_evidence=result.get("has_evidence", True),
        )
    except RuntimeError as exc:
        if "Mistral API key" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {exc}",
        )


# ── Transcript Search Route ───────────────────────────────────────────────────
@app.get(
    "/api/transcript/search",
    response_model=TranscriptSearchResponse,
    tags=["Transcript"],
    summary="Search for exact keyword occurrences with timestamps",
)
async def search_in_transcript(
    query: str = Query(..., min_length=1, description="Keyword to search in transcript"),
    session_id: str = Query(..., min_length=1, description="Session ID"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Perform exact keyword search across timestamped transcript segments."""
    sid = safe_token(session_id)
    session = session_store.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{sid}' does not exist.",
        )

    # Segments may be directly on session or nested under transcript_data
    segments = session.get("segments")
    if not segments and isinstance(session.get("transcript_data"), dict):
        segments = session["transcript_data"].get("segments", [])
    elif not segments and hasattr(session.get("transcript_data"), "segments"):
        segments = getattr(session["transcript_data"], "segments", [])
    if not segments:
        segments = []

    matches = search_transcript(segments, query, max_results=limit)

    return TranscriptSearchResponse(
        query=query,
        session_id=sid,
        total_matches=len(matches),
        results=matches,
        matches=matches,
    )


# ── Session & Chapters Routes ─────────────────────────────────────────────────
@app.get(
    "/api/sessions",
    response_model=SessionListResponse,
    tags=["Sessions"],
    summary="List all processed video sessions with metadata",
)
async def list_sessions():
    """List all active sessions and their metadata for the video library."""
    all_sessions = session_store.list_all()
    session_items = []
    for sid, data in all_sessions.items():
        summary_raw = data.get("summary", "")
        preview = summary_raw[:120] + "..." if len(summary_raw) > 120 else summary_raw
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        source_url = meta.get("source_url") or data.get("source_url")
        platform = meta.get("source_platform") or ("youtube" if source_url and "youtu" in source_url.lower() else "file")

        session_items.append(
            SessionInfo(
                user_id=sid,
                has_transcript=bool(data.get("transcript")),
                segment_count=len(data.get("segments", [])),
                summary_preview=preview,
                title=data.get("title") or meta.get("title") or "Processed Video",
                duration_seconds=data.get("duration_seconds") or meta.get("duration"),
                thumbnail=meta.get("thumbnail"),
                author=meta.get("author"),
                source_url=source_url,
                source_platform=platform,
                created_at=data.get("created_at") or data.get("updated_at"),
                updated_at=data.get("updated_at"),
            )
        )
    return SessionListResponse(count=len(session_items), sessions=session_items)


@app.get(
    "/api/sessions/{session_id}",
    tags=["Sessions"],
    summary="Retrieve full transcript, summary, chapters, and metadata for a video",
)
async def get_session(session_id: str):
    """Retrieve full transcript, summary, and metadata for a session."""
    sid = safe_token(session_id)
    session = session_store.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{sid}' does not exist.",
        )
    return session


@app.get(
    "/api/sessions/{session_id}/chapters",
    response_model=List[Chapter],
    tags=["Sessions"],
    summary="Get derived video chapters with timestamps",
)
async def get_session_chapters(session_id: str):
    """Retrieve the timestamp-grounded chapters for a session."""
    sid = safe_token(session_id)
    session = session_store.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{sid}' does not exist.",
        )
    raw_chapters = session.get("chapters", [])
    chapters_out: List[Chapter] = []
    for i, c in enumerate(raw_chapters):
        if isinstance(c, Chapter):
            chapters_out.append(c)
        elif isinstance(c, dict):
            cid = int(c.get("chapter_id", i + 1))
            title = str(c.get("title", f"Chapter {cid}"))
            st = float(c.get("start_time", 0.0))
            et = float(c.get("end_time", 0.0))
            st_s = int(st)
            et_s = int(et)
            fmt_ts = c.get("formatted_timestamp") or f"[{st_s//60:02d}:{st_s%60:02d} - {et_s//60:02d}:{et_s%60:02d}]"
            summary = str(c.get("summary", ""))
            chapters_out.append(Chapter(
                chapter_id=cid,
                title=title,
                start_time=st,
                end_time=et,
                formatted_timestamp=fmt_ts,
                summary=summary,
            ))
    return chapters_out


def format_seconds_to_srt(seconds: float) -> str:
    """Format seconds into HH:MM:SS,mmm for SubRip subtitles."""
    s = max(0.0, seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int(round((s - int(s)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_seconds_to_vtt(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm for WebVTT subtitles."""
    s = max(0.0, seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int(round((s - int(s)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


@app.get(
    "/api/sessions/{session_id}/export",
    tags=["Sessions"],
    summary="Export session transcript or summary in TXT, MD, SRT, or VTT format",
)
async def export_session(
    session_id: str,
    format: str = Query("txt", pattern="^(txt|md|srt|vtt)$", description="Format: txt, md, srt, vtt"),
):
    """Generate real formatted exports backed by actual transcript segments and timestamps."""
    sid = safe_token(session_id)
    session = session_store.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{sid}' does not exist.",
        )

    title = session.get("title") or "EchoMind Video Transcript"
    segments = session.get("segments", [])
    summary = session.get("summary", "")

    if format == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            st = float(seg.get("start_time", seg.get("start", 0.0)))
            et = float(seg.get("end_time", seg.get("end", 0.0)))
            text = str(seg.get("text", "")).strip()
            srt_lines.append(f"{i}\n{format_seconds_to_srt(st)} --> {format_seconds_to_srt(et)}\n{text}\n")
        content = "\n".join(srt_lines) if srt_lines else "1\n00:00:00,000 --> 00:00:01,000\n[No transcript segments available]\n"
        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{sid}_transcript.srt"'},
        )

    elif format == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for i, seg in enumerate(segments, start=1):
            st = float(seg.get("start_time", seg.get("start", 0.0)))
            et = float(seg.get("end_time", seg.get("end", 0.0)))
            text = str(seg.get("text", "")).strip()
            vtt_lines.append(f"{i}\n{format_seconds_to_vtt(st)} --> {format_seconds_to_vtt(et)}\n{text}\n")
        content = "\n".join(vtt_lines)
        return Response(
            content=content,
            media_type="text/vtt; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{sid}_transcript.vtt"'},
        )

    elif format == "md":
        lines = [
            f"# {title}",
            "",
            f"- **Session ID:** `{sid}`",
        ]
        duration = session.get("duration_seconds")
        if duration:
            lines.append(f"- **Duration:** {int(duration)//60}m {int(duration)%60}s")
        meta = session.get("metadata")
        if isinstance(meta, dict) and meta.get("source_url"):
            lines.append(f"- **Source:** {meta['source_url']}")
        lines.extend(["", "---", "", "## Video Summary", "", summary or "_No summary available._", "", "---", "", "## Grounded Transcript"])
        for seg in segments:
            st = float(seg.get("start_time", seg.get("start", 0.0)))
            et = float(seg.get("end_time", seg.get("end", 0.0)))
            ts = f"[{int(st)//60:02d}:{int(st)%60:02d} - {int(et)//60:02d}:{int(et)%60:02d}]"
            lines.append(f"**{ts}** {seg.get('text', '').strip()}\n")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{sid}_report.md"'},
        )

    else:  # txt
        lines = [f"=== {title} ===\n", f"SUMMARY:\n{summary}\n\nTRANSCRIPT:"]
        for seg in segments:
            st = float(seg.get("start_time", seg.get("start", 0.0)))
            et = float(seg.get("end_time", seg.get("end", 0.0)))
            ts = f"[{int(st)//60:02d}:{int(st)%60:02d} - {int(et)//60:02d}:{int(et)%60:02d}]"
            lines.append(f"{ts} {seg.get('text', '').strip()}")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{sid}_transcript.txt"'},
        )


@app.get(
    "/api/sessions/{session_id}/suggested-questions",
    response_model=SuggestedQuestionsResponse,
    tags=["Q&A"],
    summary="Get suggested starter questions tailored to the video content",
)
async def get_suggested_questions(session_id: str):
    """Derive relevant questions from the video title, summary, and chapters."""
    sid = safe_token(session_id)
    session = session_store.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{sid}' does not exist.",
        )

    title = session.get("title") or "the video"
    chapters = session.get("chapters", [])
    questions = [
        f"What are the main topics and key arguments covered in {title}?",
        "What are the most important takeaways and actionable recommendations?",
        "What specific examples or case studies were mentioned?",
    ]
    if chapters and len(chapters) > 1:
        c1 = chapters[0].get("title") if isinstance(chapters[0], dict) else getattr(chapters[0], "title", "")
        c2 = chapters[1].get("title") if isinstance(chapters[1], dict) else getattr(chapters[1], "title", "")
        if c1:
            questions.append(f"Can you explain what was discussed in '{c1}'?")
        if c2:
            questions.append(f"What key points were made during '{c2}'?")
    else:
        questions.append("What problems or challenges does the speaker identify?")
        questions.append("Summarize the speaker's final conclusions.")

    return SuggestedQuestionsResponse(session_id=sid, questions=questions[:5])


@app.delete(
    "/api/sessions/{session_id}",
    response_model=ResetSessionResponse,
    tags=["Sessions"],
    summary="Delete a session and its vector collection from library",
)
async def delete_session_by_id(session_id: str):
    """Delete a session permanently from storage, Chroma vector index, and cache."""
    sid = safe_token(session_id)
    existed = session_store.delete(sid)
    clear_session_cache(sid)
    logger.info("Deleted session '%s' (existed=%s)", sid, existed)
    return ResetSessionResponse(
        message=f"Session '{sid}' deleted." if existed else f"Session '{sid}' not found.",
        session_id=sid,
        existed=existed,
    )


@app.delete(
    "/reset",
    response_model=ResetSessionResponse,
    tags=["Sessions"],
    summary="Reset a session (legacy route)",
)
async def reset_session(user: str = Query(..., alias="user")):
    """Reset session state, delete ChromaDB collection, and invalidate RAG caches."""
    sid = safe_token(user)
    existed = session_store.delete(sid)
    clear_session_cache(sid)

    message = (
        f"Session '{sid}' and its vector collection have been successfully reset."
        if existed
        else f"No active session found for '{sid}'."
    )
    logger.info("Reset session '%s' (existed=%s)", sid, existed)
    return ResetSessionResponse(
        message=message,
        session_id=sid,
        existed=existed,
    )