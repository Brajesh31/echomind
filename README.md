# 🎙️ EchoMind — AI-Powered Video Assistant

**Transform any YouTube video or audio recording into structured summaries, timestamped transcripts, and grounded conversational Q&A.**

---

## 📌 Overview

**EchoMind** is an intelligent media assistant that ingests video and audio content, transcribes spoken dialogue into timestamped transcript segments, builds an isolated semantic vector index, and enables users to ask questions grounded directly in the video content.

EchoMind combines:
- **Audio Ingestion & Extraction**: Automatic YouTube audio extraction via `yt-dlp` and 16kHz mono WAV conversion via `pydub`.
- **SSRF Protection**: Strict URL and host validation blocking loopback, internal, and cloud-metadata addresses.
- **Fast Speech Transcription**: Local, high-performance speech-to-text using `faster-whisper` (CTranslate2).
- **Structured Transcript Modeling**: Temporal segments (`segment_id`, `start_time`, `end_time`, `text`) for timestamp citations and search.
- **Isolated Vector Retrieval (RAG)**: Multi-collection session isolation in `ChromaDB` with `all-MiniLM-L6-v2` embeddings.
- **Prompt-Injection Defense**: Delimited untrusted context wrappers and strict system instructions.
- **Grounded Q&A & Summarization**: Multi-stage map-reduce summarization and conversational answering via Mistral AI.
- **Robust Session Storage**: Atomic, thread-safe session persistence with automatic corruption recovery.
- **Containerized Architecture**: Multi-container Docker Compose deployment (FastAPI backend + Nginx reverse proxy).

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       Browser Client                        │
│                    http://localhost:80                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Nginx Proxy (:80)                           │
│     Serves static UI and reverse-proxies /api/*, /health     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Reverse Proxy
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (:8000)                     │
│                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌────────────┐ │
│  │ Security & SSRF │   │ Ingestion Layer │   │ Session    │ │
│  │ Host Validation │──▶│ yt-dlp + pydub  │──▶│ Store      │ │
│  └─────────────────┘   └────────┬────────┘   └────────────┘ │
│                                 │                           │
│                                 ▼                           │
│                        ┌─────────────────┐                  │
│                        │ faster-whisper  │                  │
│                        │ (CTranslate2)   │                  │
│                        └────────┬────────┘                  │
│                                 │                           │
│                    ┌────────────┴────────────┐              │
│                    ▼                         ▼              │
│          ┌───────────────────┐     ┌──────────────────┐     │
│          │ ChromaDB Vector   │     │ LangChain /      │     │
│          │ Store (Isolated)  │     │ Mistral AI       │     │
│          └───────────────────┘     └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
.
├── backend/
│   ├── app.py                  # FastAPI application and route definitions
│   ├── config.py               # Centralized Pydantic configuration & validation
│   ├── models/
│   │   ├── schemas.py          # Request & response Pydantic models
│   │   └── transcript.py       # Structured transcript & segment models
│   └── services/
│       ├── security.py         # SSRF protection, URL & token sanitization
│       └── session_store.py    # Atomic, thread-safe session persistence
├── Audio_Preprocessing/
│   ├── audio_processor.py      # Audio extraction, chunking, and conversion
│   ├── llm_pipeline.py         # Mistral summarization & prompt defense
│   ├── main.py                 # Pipeline coordination
│   ├── transcriber.py          # Whisper model loader & segment transcription
│   └── vector_store.py         # Chroma collection manager & MMR retrieval
├── frontend/
│   └── index.html              # Single-page UI with safe DOM rendering
├── docs/
│   └── architecture-audit.md   # Architectural audit & decision matrix
├── tests/
│   ├── test_api.py             # Integration tests for API endpoints
│   ├── test_config.py          # Configuration tests
│   ├── test_security.py        # SSRF and path sanitization tests
│   ├── test_session_store.py   # Atomic file storage tests
│   └── test_transcript.py      # Structured transcript model tests
├── docker-compose.yml          # Backend + Nginx multi-container compose
├── dockerfile.backend          # Python 3.11 Slim with ffmpeg and curl
├── dockerfile.frontend         # Nginx Alpine serving frontend & proxy
├── nginx.conf                  # Nginx proxy & static configuration
├── requirements.txt            # Production and test dependencies
├── .env.example                # Safe environment configuration template
└── README.md                   # Product documentation
```

---

## 🧰 Technology Stack

| Layer | Component | Version / Details |
|---|---|---|
| **API Framework** | FastAPI | >= 0.115.0 |
| **ASGI Server** | Uvicorn | >= 0.32.0 |
| **Audio Ingestion** | yt-dlp + pydub | Standardized 16kHz mono WAV |
| **Audio Processing** | FFmpeg | System runtime / Docker |
| **Speech-to-Text** | faster-whisper | CTranslate2 (CPU int8 / CUDA) |
| **Vector Database** | ChromaDB | Isolated per-session collections |
| **Embeddings** | sentence-transformers | `all-MiniLM-L6-v2` (384-d) |
| **LLM Provider** | Mistral AI | `mistral-small-latest` |
| **Orchestration** | LangChain (LCEL) | >= 0.3.0 |
| **Background Jobs** | ThreadPoolExecutor | Granular lifecycle & idempotency |
| **Frontend** | Vanilla HTML5 / CSS3 / ES6 | Reactive Video Workspace & Library |
| **Reverse Proxy** | Nginx | Alpine image |
| **Test Suite** | pytest + pytest-asyncio | 56 verified automated tests |

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)
- A **Mistral AI API Key** — obtain one at [console.mistral.ai](https://console.mistral.ai/)

### 1. Environment Setup

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set your Mistral API key:

```env
MISTRAL_API_KEY=your_actual_mistral_api_key
WHISPER_MODEL=base
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

- **Frontend Workspace:** Open `http://localhost`
- **Backend API Docs:** Open `http://localhost:8000/docs` (Swagger UI)
- **Health Check:** `http://localhost:8000/health`
- **Readiness Probe:** `http://localhost:8000/ready`

---

## 🏃 Running Locally (Without Docker)

> Requires Python 3.11+, ffmpeg, and git.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
# Ensure .env exists with MISTRAL_API_KEY

# 3. Start the FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` in your browser or visit `http://localhost:8000/`.

---

## 🧪 Running the Test Suite

EchoMind includes unit and integration tests covering SSRF protection, URL validation, configuration, atomic session persistence, structured transcript segments, segment-aware chunking, session isolation, prompt-injection resistance, anti-hallucination guardrails, background jobs, subtitle exporters (`.srt`, `.vtt`, `.md`, `.txt`), suggested questions, and session deletion.

```bash
# Run the complete test suite
python -m pytest -v

# Run with concise summary
python -m pytest -q
```

Sample output:
```text
56 passed in 1.77s
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Cheap liveness check returning operational status |
| `GET` | `/ready` | Readiness probe verifying model caches & storage |
| `POST` | `/api/jobs/process` | Asynchronously submit video URL for background processing |
| `GET` | `/api/jobs/{job_id}` | Poll background processing job status and progress |
| `GET` | `/api/jobs` | List all tracked processing jobs |
| `POST` | `/youtube_process` | Synchronous YouTube ingestion & summarization (legacy/fallback) |
| `POST` | `/process_text_audio` | Ingest an uploaded audio/video file or `.txt` transcript |
| `POST` | `/question` | Grounded Q&A returning verbatim evidence sources and timestamps |
| `GET` | `/api/transcript/search` | Fast exact keyword search across transcript segments |
| `GET` | `/api/sessions` | List active sessions with rich metadata (duration, author, thumbnail) |
| `GET` | `/api/sessions/{session_id}` | Retrieve session transcript, summary, chapters, and metadata |
| `GET` | `/api/sessions/{session_id}/chapters` | Retrieve derived timestamp-grounded chapters |
| `GET` | `/api/sessions/{session_id}/suggested-questions` | Dynamically derived contextual questions for the video |
| `GET` | `/api/sessions/{session_id}/export` | Export subtitles (`.srt`, `.vtt`) or reports (`.md`, `.txt`) |
| `DELETE` | `/api/sessions/{session_id}` | Permanently delete session, isolated vector store, and cache |
| `DELETE` | `/reset?user={session_id}` | Reset session and vector collection (legacy compatibility) |

Interactive Swagger documentation is accessible at `/docs`.

---

## 🔒 Security Measures

1. **SSRF Mitigation**:
   - Host validation against supported YouTube domains (`youtube.com`, `youtu.be`).
   - Rejection of loopback addresses (`127.0.0.1`, `localhost`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and cloud metadata (`169.254.169.254`).
2. **Safe Filesystem Usage**:
   - Input sanitization (`_safe_token`) strips path separators (`../`, `..\`) and special characters.
   - All temporary audio and chunk files are cleaned up in `try/finally` blocks.
3. **Session Store Integrity**:
   - Atomic writes via temporary files and `os.replace` prevent file corruption during sudden shutdown.
   - Corrupted JSON states are backed up automatically and cleanly reset.
4. **Prompt Injection Defense**:
   - Transcripts are enclosed in explicit `=== BEGIN UNTRUSTED VIDEO TRANSCRIPT ===` delimiters.
   - System prompts explicitly direct the model to treat transcript content strictly as reference data and never execute instructions found within.
5. **Frontend Security**:
   - Safe DOM manipulation avoids unsafe `innerHTML` interpolation of user and bot messages.
   - Event delegation replaces inline string-interpolated event handlers.
6. **HTTP Security Headers**:
   - Enforces `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: strict-origin-when-cross-origin`, and `Permissions-Policy`.

---

## 📜 License & Attribution

EchoMind is open-source software licensed under the [MIT License](LICENSE). Third-party open-source components and their respective licenses are documented in [LICENSE](LICENSE).

---

## ⚠️ Known Limitations

- **First Model Boot**: On the initial startup, faster-whisper and HuggingFace download model weights, which may take 1-2 minutes depending on connection speed. Subsequent boots use cached volumes.
- **Processing Time**: Very long videos (>1 hour) require several minutes to transcribe on CPU. GPU acceleration is supported via `WHISPER_DEVICE=cuda`.
- **API Key**: Q&A and summary features require a valid Mistral AI API key in `.env`.

