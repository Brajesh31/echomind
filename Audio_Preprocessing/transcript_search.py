"""EchoMind — Exact Transcript Keyword & Semantic Search.

Provides fast, deterministic keyword search across structured transcript segments
with accurate timestamps and snippet context (no LLM hallucination).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from backend.models.schemas import TranscriptSearchResult
from backend.models.transcript import TranscriptSegment


def format_seconds(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    secs = int(max(0.0, seconds))
    mins, s = divmod(secs, 60)
    hours, m = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def search_transcript(
    segments: List[Union[Dict[str, Any], TranscriptSegment]],
    query: str,
    max_results: int = 50,
) -> List[TranscriptSearchResult]:
    """Search for keyword occurrences within structured transcript segments.

    Args:
        segments: List of TranscriptSegment objects or dictionaries with start_time, end_time, text.
        query: User search term.
        max_results: Maximum matches to return.

    Returns:
        List of TranscriptSearchResult instances with exact timestamps and context snippets.
    """
    if not query or not query.strip():
        return []

    # If a StructuredTranscript instance was passed, extract its segments list
    if hasattr(segments, "segments"):
        segments = getattr(segments, "segments")

    cleaned_query = query.strip()
    pattern = re.compile(re.escape(cleaned_query), re.IGNORECASE)
    results: List[TranscriptSearchResult] = []

    for seg in segments:
        if isinstance(seg, dict):
            seg_id = int(seg.get("segment_id", seg.get("id", 0)))
            start_t = float(seg.get("start_time", seg.get("start", 0.0)))
            end_t = float(seg.get("end_time", seg.get("end", 0.0)))
            text = str(seg.get("text", "")).strip()
        else:
            seg_id = seg.segment_id
            start_t = seg.start_time
            end_t = seg.end_time
            text = seg.text.strip()

        if not text:
            continue

        match = pattern.search(text)
        if match:
            # Generate snippet with surrounding context
            start_idx = max(0, match.start() - 40)
            end_idx = min(len(text), match.end() + 40)
            snippet = ("..." if start_idx > 0 else "") + text[start_idx:end_idx] + ("..." if end_idx < len(text) else "")

            ts_str = f"[{format_seconds(start_t)} - {format_seconds(end_t)}]"

            results.append(
                TranscriptSearchResult(
                    segment_id=seg_id,
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    formatted_timestamp=ts_str,
                    text=text,
                    match_snippet=snippet,
                )
            )

            if len(results) >= max_results:
                break

    return results
