"""Integration & API Tests for EchoMind FastAPI endpoints."""

from unittest.mock import patch
import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.services.session_store import session_store


@pytest.fixture(autouse=True)
def clean_test_sessions():
    """Ensure clean session state for tests."""
    yield
    session_store.clear()


@pytest.fixture
def client():
    """Create test client with lifespan suppressed or mocked."""
    with patch("backend.app.warm_up"):
        with TestClient(app) as test_client:
            yield test_client


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns EchoMind status, metadata, and security headers."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "EchoMind"
    assert data["version"] == "1.0.0"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"
    assert "strict-origin-when-cross-origin" in response.headers.get("referrer-policy", "")


def test_readiness_endpoint(client: TestClient):
    """Verify GET /ready returns readiness information."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "whisper_loaded" in data
    assert "embeddings_loaded" in data
    assert "storage_ready" in data


def test_root_serves_frontend(client: TestClient):
    """Verify GET / serves HTML index."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "EchoMind" in response.text


def test_list_sessions_endpoint(client: TestClient):
    """Verify GET /api/sessions returns active sessions."""
    # Pre-populate a session
    session_store.save("s_demo", {"summary": "Demo summary", "transcript": "Demo text"})

    response = client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    session_ids = [s["user_id"] for s in data["sessions"]]
    assert "s_demo" in session_ids


def test_youtube_ssrf_rejected(client: TestClient):
    """Verify POST /youtube_process blocks internal/private URLs."""
    payload = {
        "url": "http://127.0.0.1:8000/internal",
        "user_id": "test_user",
        "append": False,
    }
    response = client.post("/youtube_process", json=payload)
    assert response.status_code == 400
    assert "SSRF" in response.json()["detail"] or "permitted" in response.json()["detail"]


def test_youtube_unsupported_domain_rejected(client: TestClient):
    """Verify POST /youtube_process rejects non-YouTube video hosts."""
    payload = {
        "url": "https://vimeo.com/12345678",
        "user_id": "test_user",
        "append": False,
    }
    response = client.post("/youtube_process", json=payload)
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_process_file_unsupported_extension(client: TestClient):
    """Verify POST /process_text_audio rejects unsupported files."""
    files = {"file": ("malicious.exe", b"MZ...", "application/octet-stream")}
    data = {"user_id": "u1", "append_previous": "false"}

    response = client.post("/process_text_audio", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_question_missing_session(client: TestClient):
    """Verify POST /question returns 404 if session does not exist."""
    payload = {
        "user_id": "non_existent_session_123",
        "question": "What happened in the video?",
    }
    response = client.post("/question", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_question_empty_validation(client: TestClient):
    """Verify POST /question rejects empty question."""
    payload = {
        "user_id": "some_user",
        "question": "",
    }
    response = client.post("/question", json=payload)
    assert response.status_code == 422  # Pydantic min_length=1 validation error


def test_reset_session_endpoint(client: TestClient):
    """Verify DELETE /reset deletes session and returns confirmation."""
    session_store.save("session_to_reset", {"summary": "tmp", "transcript": "tmp"})
    assert session_store.exists("session_to_reset")

    response = client.delete("/reset?user=session_to_reset")
    assert response.status_code == 200
    data = response.json()
    assert data["existed"] is True
    assert not session_store.exists("session_to_reset")


@patch("backend.app.run_pipeline")
def test_youtube_process_success(mock_run, client: TestClient):
    """Verify POST /youtube_process with mocked pipeline."""
    mock_run.return_value = {
        "user_id": "valid_user",
        "summary": "This is a great tutorial on Python.",
        "transcript": "Hello everyone...",
    }

    payload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "user_id": "valid_user",
        "append": False,
    }
    response = client.post("/youtube_process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "Python" in data["summary"]


def test_get_session_by_id(client: TestClient):
    """Verify GET /api/sessions/{session_id} returns session details."""
    session_store.save("s_detail", {"summary": "Detail summary", "transcript": "Some transcript"})

    response = client.get("/api/sessions/s_detail")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "s_detail"
    assert data["summary"] == "Detail summary"


def test_get_session_not_found(client: TestClient):
    """Verify GET /api/sessions/{session_id} returns 404 for unknown session."""
    response = client.get("/api/sessions/does_not_exist")
    assert response.status_code == 404


@patch("backend.app.run_pipeline")
def test_process_text_file_success(mock_run, client: TestClient):
    """Verify POST /process_text_audio handles .txt transcript upload."""
    mock_run.return_value = {
        "user_id": "txt_user",
        "summary": "Transcript summarized successfully.",
        "transcript": "Line 1.\nLine 2.",
    }

    files = {"file": ("notes.txt", b"Line 1.\nLine 2.", "text/plain")}
    data = {"user_id": "txt_user", "append_previous": "false"}

    response = client.post("/process_text_audio", files=files, data=data)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "Transcript summarized" in data["summary"]


def test_api_jobs_process_and_poll(client: TestClient):
    """Verify POST /api/jobs/process returns 202 and GET /api/jobs/{id} returns job status."""
    payload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "user_id": "job_user_1",
        "append": False,
    }
    with patch("backend.services.job_manager.job_manager.submit_processing_task") as mock_submit:
        from backend.models.schemas import JobResponse, JobStatus
        mock_submit.return_value = JobResponse(
            job_id="job_test_12345",
            session_id="job_user_1",
            status=JobStatus.QUEUED,
            stage="queued",
            progress_percent=0,
            message="Job queued for processing.",
            created_at="2026-10-22T00:00:00Z",
            updated_at="2026-10-22T00:00:00Z",
        )
        response = client.post("/api/jobs/process", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "job_test_12345"
        assert data["status"] == "queued"


def test_api_jobs_list(client: TestClient):
    """Verify GET /api/jobs returns job listing."""
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_transcript_search(client: TestClient):
    """Verify GET /api/transcript/search returns matched segments."""
    # Seed session with structured segments in transcript_data
    session_store.save("search_sess", {
        "summary": "AI Lecture",
        "transcript": "In this tutorial we discuss PyTorch.",
        "transcript_data": {
            "segments": [
                {"segment_id": 0, "start_time": 12.0, "end_time": 18.0, "text": "In this tutorial we discuss PyTorch."}
            ],
            "language": "en",
        }
    })

    response = client.get("/api/transcript/search?query=PyTorch&session_id=search_sess")
    assert response.status_code == 200
    data = response.json()
    assert data["total_matches"] == 1
    assert data["matches"][0]["segment_id"] == 0
    assert "[00:12 - 00:18]" in data["matches"][0]["formatted_timestamp"]


def test_api_get_chapters(client: TestClient):
    """Verify GET /api/sessions/{session_id}/chapters returns stored chapters."""
    session_store.save("chap_sess", {
        "summary": "AI Lecture",
        "transcript": "Full text...",
        "chapters": [
            {"title": "Introduction", "start_time": 0.0, "end_time": 60.0, "summary": "Intro overview"}
        ]
    })

    response = client.get("/api/sessions/chap_sess/chapters")
    assert response.status_code == 200
    chapters = response.json()
    assert len(chapters) == 1
    assert chapters[0]["title"] == "Introduction"


@patch("backend.app.ask_questions")
def test_question_with_grounded_evidence(mock_ask, client: TestClient):
    """Verify POST /question returns answer, sources, and has_evidence flag."""
    session_store.save("q_sess", {"summary": "Overview", "transcript": "Text content"})
    from backend.models.schemas import EvidenceSource

    mock_ask.return_value = {
        "answer": "The speaker discussed neural networks in detail.",
        "sources": [
            EvidenceSource(
                chunk_id="chunk_1",
                start_time=45.0,
                end_time=60.0,
                formatted_timestamp="[00:45 - 01:00]",
                text="The speaker discussed neural networks.",
            )
        ],
        "has_evidence": True,
    }

    payload = {
        "user_id": "q_sess",
        "question": "What did the speaker discuss?",
    }
    response = client.post("/question", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "neural networks" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["formatted_timestamp"] == "[00:45 - 01:00]"
    assert data["has_evidence"] is True


def test_export_session_formats(client: TestClient):
    """Verify GET /api/sessions/{id}/export generates valid SRT, VTT, MD, and TXT outputs."""
    session_store.save("exp_sess", {
        "title": "Deep Learning Tutorial",
        "summary": "Covers gradient descent.",
        "duration_seconds": 125.0,
        "segments": [
            {"segment_id": 0, "start_time": 10.5, "end_time": 25.2, "text": "Welcome to neural networks."},
            {"segment_id": 1, "start_time": 25.2, "end_time": 45.0, "text": "Backpropagation calculates errors."},
        ]
    })

    # 1. SRT
    r_srt = client.get("/api/sessions/exp_sess/export?format=srt")
    assert r_srt.status_code == 200
    assert "00:00:10,500 --> 00:00:25,200" in r_srt.text
    assert "Welcome to neural networks." in r_srt.text
    assert "attachment" in r_srt.headers.get("content-disposition", "")

    # 2. VTT
    r_vtt = client.get("/api/sessions/exp_sess/export?format=vtt")
    assert r_vtt.status_code == 200
    assert "WEBVTT" in r_vtt.text
    assert "00:00:10.500 --> 00:00:25.200" in r_vtt.text

    # 3. Markdown
    r_md = client.get("/api/sessions/exp_sess/export?format=md")
    assert r_md.status_code == 200
    assert "# Deep Learning Tutorial" in r_md.text
    assert "## Video Summary" in r_md.text
    assert "[00:10 - 00:25]" in r_md.text

    # 4. Plain Text
    r_txt = client.get("/api/sessions/exp_sess/export?format=txt")
    assert r_txt.status_code == 200
    assert "Welcome to neural networks." in r_txt.text


def test_suggested_questions(client: TestClient):
    """Verify GET /api/sessions/{id}/suggested-questions derives contextual questions."""
    session_store.save("sug_sess", {
        "title": "Transformers Architecture Explained",
        "summary": "Attention mechanism is key.",
        "chapters": [
            {"chapter_id": 1, "title": "Self-Attention Mechanism", "start_time": 0.0, "end_time": 60.0, "summary": "Core formula"}
        ]
    })

    response = client.get("/api/sessions/sug_sess/suggested-questions")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sug_sess"
    assert len(data["questions"]) >= 3
    assert any("Transformers Architecture" in q or "Self-Attention" in q for q in data["questions"])


def test_delete_session_by_id(client: TestClient):
    """Verify DELETE /api/sessions/{session_id} deletes session and index."""
    session_store.save("del_sess", {"summary": "To delete", "transcript": "Some text"})
    assert session_store.exists("del_sess")

    response = client.delete("/api/sessions/del_sess")
    assert response.status_code == 200
    data = response.json()
    assert data["existed"] is True
    assert not session_store.exists("del_sess")



