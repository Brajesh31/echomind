"""Tests for SessionStore: atomic writes, error recovery, and persistence."""

import json
from pathlib import Path
from backend.services.session_store import SessionStore


def test_session_store_crud(tmp_path: Path):
    """Verify session save, get, exists, list, and delete."""
    store_file = tmp_path / "test_sessions.json"
    store = SessionStore(store_file)

    # 1. Non-existent session
    assert store.get("session_1") is None
    assert not store.exists("session_1")

    # 2. Save session
    data = {
        "summary": "Key points of video",
        "transcript": "Hello and welcome.",
        "segments": [{"segment_id": 0, "start_time": 0.0, "end_time": 3.0, "text": "Hello"}],
    }
    saved = store.save("session_1", data)
    assert saved["user_id"] == "session_1"
    assert saved["summary"] == "Key points of video"
    assert store.exists("session_1")

    # 3. Read back from disk
    assert store_file.exists()
    content = json.loads(store_file.read_text(encoding="utf-8"))
    assert "session_1" in content
    assert content["session_1"]["summary"] == "Key points of video"

    # 4. List all
    all_sessions = store.list_all()
    assert "session_1" in all_sessions

    # 5. Delete session
    assert store.delete("session_1")
    assert not store.exists("session_1")
    assert not store.delete("session_1")  # Second delete returns False


def test_session_store_corrupted_file_recovery(tmp_path: Path):
    """Verify recovery when session file contains invalid JSON."""
    store_file = tmp_path / "corrupt_sessions.json"
    store_file.write_text("{this is not valid json!@@#", encoding="utf-8")

    # Loading corrupted file should not raise an unhandled exception
    store = SessionStore(store_file)
    assert store.list_all() == {}

    # Corrupted backup should be created
    backup = tmp_path / "corrupt_sessions.corrupt.bak"
    assert backup.exists()


def test_session_store_empty_file(tmp_path: Path):
    """Verify empty file is safely initialized without error."""
    store_file = tmp_path / "empty_sessions.json"
    store_file.write_text("", encoding="utf-8")

    store = SessionStore(store_file)
    assert store.list_all() == {}

    # Saving a new session writes valid JSON
    store.save("s_new", {"summary": "test", "transcript": "text"})
    assert store.get("s_new")["summary"] == "test"
