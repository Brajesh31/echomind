"""EchoMind — Structured Transcript Data Models.

Defines the core data structures for timestamped transcript segments,
full transcripts, and formatting utilities for RAG grounding and citations.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Represents a single continuous segment of transcribed speech with timing."""

    segment_id: int = Field(
        ..., description="Zero-based or model-assigned segment index"
    )
    start_time: float = Field(
        ..., ge=0.0, description="Start timestamp in seconds from beginning of audio"
    )
    end_time: float = Field(
        ..., ge=0.0, description="End timestamp in seconds from beginning of audio"
    )
    text: str = Field(..., description="Transcribed text for this temporal segment")

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return max(0.0, self.end_time - self.start_time)

    def formatted_timestamp(self) -> str:
        """Return HH:MM:SS or MM:SS formatted string for citations."""
        def _fmt(seconds: float) -> str:
            mins, secs = divmod(int(seconds), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                return f"{hours:02d}:{mins:02d}:{secs:02d}"
            return f"{mins:02d}:{secs:02d}"

        return f"[{_fmt(self.start_time)} - {_fmt(self.end_time)}]"


class StructuredTranscript(BaseModel):
    """Container for the complete structured transcript of a media source."""

    segments: List[TranscriptSegment] = Field(
        default_factory=list, description="Ordered list of timestamped segments"
    )
    full_text: str = Field(
        default="", description="Complete concatenated transcript text"
    )
    language: str = Field(
        default="english", description="Detected or specified speech language"
    )
    duration_seconds: Optional[float] = Field(
        default=None, description="Total audio/video duration in seconds if known"
    )

    @classmethod
    def from_segments(
        cls,
        segments: List[TranscriptSegment],
        language: str = "english",
        duration: Optional[float] = None,
    ) -> StructuredTranscript:
        """Construct a structured transcript from a list of segments."""
        full_text = " ".join(s.text.strip() for s in segments if s.text.strip())
        calculated_duration = (
            duration
            if duration is not None
            else (segments[-1].end_time if segments else 0.0)
        )
        return cls(
            segments=segments,
            full_text=full_text,
            language=language,
            duration_seconds=calculated_duration,
        )

    @classmethod
    def from_raw_text(
        cls, raw_text: str, language: str = "english"
    ) -> StructuredTranscript:
        """Construct from raw text when timestamps are not available (e.g. uploaded .txt file).

        Does not invent fake timestamps.
        """
        cleaned = raw_text.strip()
        segments = [
            TranscriptSegment(
                segment_id=0,
                start_time=0.0,
                end_time=0.0,
                text=cleaned,
            )
        ] if cleaned else []
        return cls(
            segments=segments,
            full_text=cleaned,
            language=language,
            duration_seconds=None,
        )

    def to_dict_list(self) -> List[dict]:
        """Serialize segments to a list of dicts for JSON storage."""
        return [s.model_dump() for s in self.segments]
