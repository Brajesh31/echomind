# EchoMind — Architecture & Codebase Audit Report

## 1. Executive Summary

**Project Name:** EchoMind (formerly Audio Transcriber / Ai-Powered-Video-Assistant)  
**Audit Date:** October 2026  
**Status:** Foundation Audit & Rebranding (Prompt 1 of 4)

EchoMind is an AI-powered media assistant that transcribes audio/video content (specifically YouTube URLs and uploaded audio/video files), generates structured summaries, and allows users to ask grounded questions against the content using Retrieval-Augmented Generation (RAG).

This audit documents the state of the codebase prior to and during the architectural refactoring, identifying working components, broken configurations, security vulnerabilities, and technical debt, while laying out the roadmap for subsequent prompts.

---

## 2. High-Level Current Flow

```text
User (Browser)
   │
   ▼
Frontend (Vanilla HTML/CSS/JS served via Nginx)
   │
   ▼  [REST / JSON & Multipart]
FastAPI API Layer (backend/app.py)
   │
   ├─► Video / Audio Ingestion (yt-dlp / pydub)
   │     └─► Temp audio chunks (WAV, 16kHz mono)
   │
   ├─► Speech Transcription (faster-whisper)
   │     └─► Text & Segment extraction
   │
   ├─► Vector Store & Embeddings (HuggingFace sentence-transformers + ChromaDB)
   │     └─► Segment embeddings persisted per session
   │
   ├─► LLM Processing (LangChain + Mistral AI)
   │     ├─► Map-Reduce Summarization & Title Generation
   │     └─► Retrieval-Augmented Generation (Q&A Chain)
   │
   └─► Session & Metadata Storage (sessions.json)
```

---

## 3. Component-by-Component Assessment

### 3.1 Frontend (`frontend/index.html`)
- **State:** Working UI, but with legacy branding ("Audio Transcriber"), simulated progress timers that do not match backend progress, and potential XSS vectors in DOM manipulation.
- **Tech Stack:** Vanilla HTML5, CSS3 with CSS variables and keyframe animations, Vanilla JavaScript.
- **Decision:** **REFACTOR**. Keep the aesthetic base and reactive single-page feel, rebrand to EchoMind, fix DOM security (safe text escaping / element construction), and prepare honest progress state handling.

### 3.2 Backend API (`backend/app.py`)
- **State:** Monolithic route file coordinating storage, audio processing, transcription, vector store, and LLM calls directly.
- **Tech Stack:** FastAPI, Pydantic, Uvicorn, Starlette.
- **Issues:**
  - Hardcoded `ALLOWED_ORIGINS = ["http://localhost:5500", "http://127.0.0.1:5500"]` (missing standard Vite/Nginx port 80/8000/3000 origins).
  - Routes lacked strict request validation (e.g. YouTube URL format, SSRF check).
  - Missing `/ready` endpoint for readiness checking.
  - Route names (`/process_text_audio`, `/youtube_process`) differed from README documentation (`/process`, `/ask`).
- **Decision:** **REFACTOR**. Separate routing from configuration, session management, and service orchestration.

### 3.3 Audio Processing & Ingestion (`Audio_Preprocessing/audio_processor.py`)
- **State:** Fragile and security-sensitive.
- **Issues:**
  - Critical SSRF vulnerability: accepts any URL without validation, allowing internal network / cloud metadata scanning (`http://169.254.169.254`, `http://localhost`).
  - Output filename template `%(title)s.%(ext)s` vulnerable to filesystem issues with special characters.
  - Hardcoded local author path left in `__main__` block (`C:\Users\Deepanshu sharma\...`).
- **Decision:** **REFACTOR**. Add URL domain allowlist (YouTube domains only), SSRF protection against private IPs, deterministic sanitized paths, and robust cleanup in `try/finally`.

### 3.4 Transcription (`Audio_Preprocessing/transcriber.py`)
- **State:** Functional using `faster-whisper`, but discards timestamps.
- **Issues:**
  - Only extracts `segment.text`, discarding `segment.start` and `segment.end`.
  - Discarding timestamps makes citations, transcript search, and grounded timestamp evidence impossible.
  - Model device selection hardcoded to CPU with no configurable fallback.
- **Decision:** **REFACTOR**. Introduce a structured `TranscriptSegment` model (`segment_id`, `start_time`, `end_time`, `text`) to preserve temporal data while retaining backward-compatible full text.

### 3.5 Vector Store & Embeddings (`Audio_Preprocessing/vector_store.py`)
- **State:** Functional, but fragile collection naming and session cleanup.
- **Issues:**
  - Collection name in Chroma uses raw `user_id`, which may violate Chroma naming constraints (3-63 chars, alphanumeric, underscores/hyphens).
  - In-memory `_chain_cache` and `_vs_cache` are unbounded and not invalidated when sessions are reset or deleted.
  - Hardcoded "Meeting Assistant" prompts rather than video assistant prompts.
- **Decision:** **REFACTOR**. Sanitize collection names, add proper cache invalidation, and isolate session collections safely.

### 3.6 LLM Pipeline (`Audio_Preprocessing/llm_pipeline.py`)
- **State:** Hardcoded to Mistral AI via `langchain-mistralai`.
- **Issues:**
  - Missing centralized error handling for rate limits, network timeouts, or missing API keys.
  - Prompt injection risk: raw transcript chunks interpolated directly into human prompts without clear boundary markers or system instructions treating transcript as untrusted data.
- **Decision:** **REFACTOR**. Encapsulate LLM calls, handle provider errors gracefully, and enforce prompt delimiters treating transcripts as untrusted data.

### 3.7 Session & JSON Storage (`Audio_Preprocessing/main.py`)
- **State:** Reads and writes `sessions.json` directly.
- **Issues:**
  - Non-atomic file writes (`with open(...) as f: json.dump(...)`) risk data corruption if interrupted mid-write.
  - Malformed JSON crashes startup or session access.
- **Decision:** **REFACTOR**. Create an atomic file persistence abstraction (`SessionStore`) with thread locks, safe parsing, and fallback defaults.

### 3.8 Docker & Deployment Configuration
- **State:** **BROKEN**.
- **Issues:**
  - `dockerfile.frontend` contained a duplicated copy of `docker-compose.yml` instead of a valid Dockerfile!
  - `docker-compose.yml` used obsolete `version: '3.9'` tag.
  - Missing `.env` / `.env.example`, causing `docker compose config` to error out immediately.
  - Nginx configuration proxies only specific old endpoints.
- **Decision:** **REPLACE/FIX**. Write a valid `dockerfile.frontend` with Nginx Alpine serving `frontend/` and proxying API calls; fix `docker-compose.yml` and provide `.env.example`.

---

## 4. Security Findings & Mitigations

| Vulnerability / Risk | Severity | Root Cause | Mitigation Implemented |
|---|---|---|---|
| **SSRF via URL Ingestion** | HIGH | `audio_processor.py` passed arbitrary user-supplied URLs to `yt-dlp` | Strict URL validation enforcing YouTube hostnames (`youtube.com`, `youtu.be`) and blocking private/loopback/cloud IP addresses |
| **Path Traversal / Injection** | MEDIUM | Filenames and user IDs used in temp directories | `_safe_token` regex sanitization and deterministic UUID paths |
| **Prompt Injection** | MEDIUM | Transcript text treated as trusted text in LLM prompts | System prompts updated with explicit untrusted data delimiters and instructions |
| **Data Corruption on Write** | MEDIUM | Non-atomic write to `sessions.json` | Atomic write pattern (write to temporary file then atomic rename/replace) |
| **Sensitive Data Exposure** | LOW | Raw exceptions with tracebacks could be returned to client | Structured HTTP exception handlers returning clean user-facing error messages |

---

## 5. Decision Matrix

| Component | Decision | Rationale |
|---|---|---|
| Frontend (`frontend/index.html`) | **REFACTOR** | Preserve smooth dark UI; rebrand to EchoMind; safe DOM rendering; align API calls |
| Docker Frontend (`dockerfile.frontend`) | **REPLACE** | Previous file was corrupt (contained docker-compose.yml) |
| Docker Compose (`docker-compose.yml`) | **REFACTOR** | Remove obsolete version tag, fix healthchecks and volumes |
| Config System | **REPLACE** | Create centralized `backend/config.py` with validated environment variables |
| Session Storage | **REFACTOR** | Create atomic `SessionStore` with thread safety and corruption recovery |
| Transcription (`transcriber.py`) | **REFACTOR** | Add structured transcript segments with timestamps (`start_time`, `end_time`) |
| Vector Store (`vector_store.py`) | **REFACTOR** | Sanitize Chroma collections, add cache eviction, fix prompt wording |
| LLM Pipeline (`llm_pipeline.py`) | **REFACTOR** | Add error handling, untrusted context boundaries, clean abstraction |
| Test Suite | **NEW** | Add comprehensive unit and integration tests under `tests/` |

---

## 6. Recommendations & Roadmap for Prompts 2, 3, and 4

- **Prompt 2 (Deep Processing & RAG Engine - COMPLETED):**
  - Implement full timestamped transcript explorer models.
  - Support long-video processing with robust chunking and progress reporting.
  - Enhance Q&A with grounded evidence citations citing specific timestamps.
  - Implement chapters and key highlights extraction.
  - Background asynchronous job manager with granular state tracking and idempotency.
  - Transcript keyword search with millisecond-exact timestamps.
- **Prompt 3 (Product UI & Workspace):**
  - Build responsive video library, transcript explorer with clickable timestamps, video player seeking, search, and export options (TXT, Markdown, SRT/VTT).
- **Prompt 4 (QA, Performance & Release Readiness):**
  - End-to-end load testing, Docker multi-platform verification, production security hardening, and final release packaging.

---

## 7. Prompt 2 Implementation Audit & Verification

During Prompt 2, the following engineering enhancements were audited, implemented, and verified:

1. **Media Ingestion & Audio Normalization (`Audio_Preprocessing/audio_processor.py`):**
   - FFmpeg availability probe (`ensure_ffmpeg()`) with helpful actionable error messages.
   - Deterministic 16kHz mono WAV conversion with temporary workspace cleanup (`try...finally`).
   - YouTube metadata extraction (title, duration, channel, upload date) with duration limits.
   - Audio chunking preserving absolute timestamps across chunks: $\text{chunk\_start} + \Delta t$.

2. **Transcription Engine (`Audio_Preprocessing/transcriber.py`):**
   - CUDA detection with automatic fallback to CPU `int8`.
   - Whisper model lazy singleton with thread synchronization.
   - Text normalization and consecutive identical segment deduplication.
   - Structured transcript output preserving float timestamps.

3. **Segment-Aware Semantic Chunking (`Audio_Preprocessing/vector_store.py`):**
   - Chunks constructed over contiguous `TranscriptSegment` objects instead of arbitrary character slices.
   - Preserves exact temporal boundaries `[min(start), max(end)]` in Chroma document metadata.
   - 1-segment overlap preserves conversational narrative continuity.

4. **Vector Store & Isolation:**
   - Isolated Chroma collections per session with sanitized collection names (`col_{hash}`).
   - HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-d).

5. **Grounded Q&A & Anti-Hallucination (`Audio_Preprocessing/vector_store.py`):**
   - Prompt-injection resistance via XML/data boundaries `<transcript_evidence>` and explicit system instructions.
   - Anti-hallucination fallback returning `"⚠️ This information was not found in the video transcript."` and `has_evidence: false`.
   - Verbatim evidence citations returned as `EvidenceSource` objects with readable timestamps.

6. **Asynchronous Background Processing (`backend/services/job_manager.py`):**
   - In-process `ThreadPoolExecutor` tracking `JobStatus` lifecycle (`queued` -> `validating` -> `downloading` -> `transcribing` -> `processing_transcript` -> `indexing` -> `completed`/`failed`).
   - Idempotency: Duplicate submissions for identical active sources reuse existing in-flight jobs.

7. **Exact Transcript Keyword Search (`Audio_Preprocessing/transcript_search.py`):**
   - Sub-millisecond substring search across transcript segments returning exact timestamps and context snippets.

8. **Verification & Test Suite:**
   - 53 automated unit and integration tests passing (`py -3.13 -m pytest -v`).
   - Complete byte-compilation (`compileall`) with 0 errors.
   - Valid Docker Compose configuration verified (`docker compose config`).
