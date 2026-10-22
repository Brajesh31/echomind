"""EchoMind — Robust Session Persistence Layer.

Provides thread-safe, atomic read/write operations for session metadata,
handling missing files, corruption recovery, and concurrency without
scattering raw open() calls across the application.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("echomind.session_store")


class SessionStore:
    """Thread-safe, atomic session storage backed by a JSON file."""

    def __init__(self, file_path: Path | str = "sessions.json"):
        self.file_path = Path(file_path).resolve()
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load session data from disk into memory with fallback recovery."""
        with self._lock:
            if not self.file_path.exists():
                logger.info("Session file '%s' does not exist; starting with empty state.", self.file_path)
                self._cache = {}
                return

            try:
                content = self.file_path.read_text(encoding="utf-8").strip()
                if not content:
                    logger.warning("Session file '%s' was empty; resetting to empty state.", self.file_path)
                    self._cache = {}
                    return

                data = json.loads(content)
                if isinstance(data, dict):
                    self._cache = data
                else:
                    logger.warning("Session file '%s' did not contain a JSON dict; resetting.", self.file_path)
                    self._cache = {}
            except (json.JSONDecodeError, OSError) as exc:
                logger.error(
                    "Error parsing session file '%s': %s. Creating backup and initializing clean store.",
                    self.file_path,
                    exc,
                )
                self._backup_corrupted()
                self._cache = {}

    def _backup_corrupted(self) -> None:
        """Move corrupted session file to a timestamped backup before resetting."""
        try:
            backup_path = self.file_path.with_suffix(".corrupt.bak")
            shutil.copyfile(self.file_path, backup_path)
            logger.info("Backed up corrupted session file to '%s'", backup_path)
        except OSError as e:
            logger.warning("Failed to backup corrupted session file: %s", e)

    def _persist(self) -> None:
        """Atomically persist in-memory session cache to disk using replace."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in the same directory (so os.replace is atomic across filesystems)
        temp_dir = self.file_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=temp_dir,
            delete=False,
            encoding="utf-8",
            prefix=".sessions_tmp_",
        ) as tmp_file:
            json.dump(self._cache, tmp_file, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp_file.name)

        try:
            # Atomic rename / replace
            os.replace(tmp_path, self.file_path)
        except OSError as exc:
            logger.error("Failed to replace sessions file with tmp file: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        with self._lock:
            return self._cache.get(session_id)

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return session_id in self._cache

    def save(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save or update session data and atomically persist to disk."""
        with self._lock:
            # Retain structured intelligence fields
            segments = data.get("segments")
            if not segments and isinstance(data.get("transcript_data"), dict):
                segments = data["transcript_data"].get("segments", [])

            stored = {
                "user_id": session_id,
                "summary": data.get("summary", ""),
                "transcript": data.get("transcript", ""),
                "segments": segments or [],
                "chapters": data.get("chapters", []),
                "metadata": data.get("metadata"),
                "duration_seconds": data.get("duration_seconds"),
                "language": data.get("language"),
                "title": data.get("title") or (data.get("metadata", {}).get("title") if isinstance(data.get("metadata"), dict) else ""),
                "history": data.get("history", []),
                "updated_at": data.get("updated_at"),
            }
            self._cache[session_id] = stored
            self._persist()
            return stored

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID and update disk."""
        with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                self._persist()
                return True
            return False

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of all active sessions."""
        with self._lock:
            return dict(self._cache)

    def clear(self) -> None:
        """Clear all sessions."""
        with self._lock:
            self._cache.clear()
            self._persist()


# Global default instance configured from settings
from backend.config import settings

session_store = SessionStore(settings.absolute_sessions_path)
