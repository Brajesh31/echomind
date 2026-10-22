"""EchoMind — API Request & Response Schemas.

Pydantic models for request validation, typed responses, evidence sources,
job states, video metadata, and transcript search.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for background media processing jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    PROCESSING_TRANSCRIPT = "processing_transcript"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    """Structured metadata captured from the media source."""
    video_id: str = Field(..., description="Unique video identifier or source hash")
    source_url: str = Field(..., description="Canonical media URL or file reference")
    title: Optional[str] = Field(default=None, description="Video title")
    duration: Optional[float] = Field(default=None, description="Duration in seconds")
    thumbnail: Optional[str] = Field(default=None, description="Thumbnail URL if available")
    author: Optional[str] = Field(default=None, description="Channel, creator, or author name")
    upload_date: Optional[str] = Field(default=None, description="Upload date (YYYYMMDD) if available")
    source_platform: str = Field(default="youtube", description="Origin platform: youtube, file, etc.")


class EvidenceSource(BaseModel):
    """A specific transcript excerpt used as evidence for an answer."""
    chunk_id: str = Field(..., description="Unique ID of the retrieved chunk")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    formatted_timestamp: str = Field(..., description="Readable timestamp (e.g. [01:25 - 01:40])")
    text: str = Field(..., description="Transcript excerpt text")


class Chapter(BaseModel):
    """A topic-segmented chapter derived from the structured transcript."""
    chapter_id: int
    title: str
    start_time: float
    end_time: float
    formatted_timestamp: str
    summary: str


class YoutubeProcessRequest(BaseModel):
    """Payload for submitting a YouTube video URL for transcription & RAG indexing."""
    url: str = Field(
        ...,
        description="Public YouTube video URL",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    user_id: str = Field(
        default="default",
        max_length=64,
        description="Session or user identifier",
    )
    append: bool = Field(
        default=False,
        description="Whether to append to the existing session history",
    )


class AskQuestionRequest(BaseModel):
    """Payload for asking a question against an indexed session."""
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Session or user identifier",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the video transcript",
    )


class QuestionResponse(BaseModel):
    """Grounded question response containing answer, session reference, and evidence sources."""
    question: str
    answer: str
    session_id: str
    sources: List[EvidenceSource] = Field(default_factory=list, description="Retrieved timestamped evidence")
    has_evidence: bool = Field(default=True, description="False if answer was not found in transcript")


class ProcessResponse(BaseModel):
    """Response returned upon completion of video / audio processing."""
    session_id: str
    summary: str
    title: Optional[str] = None
    segment_count: int = 0
    duration_seconds: Optional[float] = None
    metadata: Optional[VideoMetadata] = None
    chapters: List[Chapter] = Field(default_factory=list)


class JobResponse(BaseModel):
    """Status information for an asynchronous media processing job."""
    job_id: str
    session_id: str
    status: JobStatus
    stage: str
    progress_percent: Optional[int] = None
    message: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TranscriptSearchResult(BaseModel):
    """A single segment match from keyword searching the transcript."""
    segment_id: int
    start_time: float
    end_time: float
    formatted_timestamp: str
    text: str
    match_snippet: str


class TranscriptSearchResponse(BaseModel):
    """Results returned from searching within a video transcript."""
    query: str
    session_id: str
    total_matches: int
    results: List[TranscriptSearchResult] = Field(default_factory=list)
    matches: List[TranscriptSearchResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Basic health check response."""
    status: str = "ok"
    app_name: str
    version: str
    message: str = "EchoMind is operational."


class ReadyResponse(BaseModel):
    """Readiness probe response detailing system dependencies."""
    status: str
    whisper_loaded: bool
    embeddings_loaded: bool
    mistral_configured: bool
    storage_ready: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    """Summary information for an active session in the video library."""
    user_id: str
    has_transcript: bool
    segment_count: int = 0
    summary_preview: str = ""
    title: Optional[str] = None
    duration_seconds: Optional[float] = None
    thumbnail: Optional[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    source_platform: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionListResponse(BaseModel):
    """List of active sessions."""
    count: int
    sessions: List[SessionInfo]


class SuggestedQuestionsResponse(BaseModel):
    """Suggested grounded questions for a video session."""
    session_id: str
    questions: List[str]


class ResetSessionResponse(BaseModel):
    """Response returned upon session deletion or reset."""
    message: str
    session_id: str
    existed: bool
