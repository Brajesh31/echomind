"""Tests for Transcript Data Structures."""

from backend.models.transcript import StructuredTranscript, TranscriptSegment


def test_transcript_segment_creation():
    """Verify segment attributes, duration, and timestamp formatting."""
    seg = TranscriptSegment(
        segment_id=1,
        start_time=65.5,
        end_time=72.0,
        text="This is an important segment.",
    )

    assert seg.segment_id == 1
    assert seg.start_time == 65.5
    assert seg.end_time == 72.0
    assert seg.duration == 6.5
    assert seg.formatted_timestamp() == "[01:05 - 01:12]"


def test_transcript_segment_hour_formatting():
    """Verify timestamp formatting for long videos with hour counts."""
    seg = TranscriptSegment(
        segment_id=42,
        start_time=3665.0,  # 1h 1m 5s
        end_time=3700.0,    # 1h 1m 40s
        text="Deep into the long lecture.",
    )
    assert seg.formatted_timestamp() == "[01:01:05 - 01:01:40]"


def test_structured_transcript_from_segments():
    """Verify StructuredTranscript construction from multiple segments."""
    segments = [
        TranscriptSegment(segment_id=0, start_time=0.0, end_time=4.5, text="Hello world."),
        TranscriptSegment(segment_id=1, start_time=4.5, end_time=9.2, text="Welcome to EchoMind."),
    ]
    transcript = StructuredTranscript.from_segments(segments, language="english")

    assert transcript.full_text == "Hello world. Welcome to EchoMind."
    assert len(transcript.segments) == 2
    assert transcript.duration_seconds == 9.2
    assert transcript.language == "english"

    dict_list = transcript.to_dict_list()
    assert len(dict_list) == 2
    assert dict_list[0]["text"] == "Hello world."


def test_structured_transcript_from_raw_text():
    """Verify StructuredTranscript construction from raw text without timestamps."""
    raw = "This is a pre-existing plain text transcript without timestamps."
    transcript = StructuredTranscript.from_raw_text(raw)

    assert transcript.full_text == raw
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start_time == 0.0
    assert transcript.segments[0].end_time == 0.0
    assert transcript.duration_seconds is None
