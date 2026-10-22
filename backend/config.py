"""EchoMind — Centralized Application Configuration.

Loads environment variables with safe defaults and validates configuration
at startup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Set

from pydantic import BaseModel, Field

_BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / ".env")
except ImportError:
    _env_file = _BASE_DIR / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())




class Settings(BaseModel):
    """Central configuration for EchoMind."""

    # --- Application Metadata ---
    app_name: str = Field(default="EchoMind", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- AI & Transcription ---
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # --- Paths & Storage ---
    base_dir: Path = Field(default=_BASE_DIR)
    chroma_path: str = Field(default="vector_database", alias="CHROMA_PATH")
    data_path: str = Field(default="downloads", alias="DATA_PATH")
    sessions_file: str = Field(default="sessions.json", alias="SESSIONS_FILE")

    # --- Security & Limits ---
    cors_origins: str = Field(
        default="http://localhost,http://localhost:80,http://localhost:8000,http://localhost:5500,http://127.0.0.1:5500,http://127.0.0.1:8000",
        alias="CORS_ORIGINS",
    )
    max_upload_size_mb: int = Field(default=500, alias="MAX_UPLOAD_SIZE_MB")
    max_video_duration_minutes: int = Field(
        default=180, alias="MAX_VIDEO_DURATION_MINUTES"
    )

    allowed_av_extensions: Set[str] = {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".txt",
        ".md",
    }
    transcript_extensions: Set[str] = {".txt", ".md"}

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list of cleaned strings."""
        if not self.cors_origins:
            return ["*"]
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def absolute_chroma_path(self) -> Path:
        p = Path(self.chroma_path)
        return p if p.is_absolute() else self.base_dir / p

    @property
    def absolute_data_path(self) -> Path:
        p = Path(self.data_path)
        return p if p.is_absolute() else self.base_dir / p

    @property
    def absolute_sessions_path(self) -> Path:
        p = Path(self.sessions_file)
        return p if p.is_absolute() else self.base_dir / p

    @property
    def is_mistral_configured(self) -> bool:
        """Check if Mistral API key is set and not a placeholder."""
        key = (self.mistral_api_key or "").strip()
        return bool(key) and key != "your_mistral_api_key_here"

    def ensure_directories(self) -> None:
        """Ensure required runtime directories exist."""
        self.absolute_data_path.mkdir(parents=True, exist_ok=True)
        self.absolute_chroma_path.mkdir(parents=True, exist_ok=True)

    model_config = {"populate_by_name": True}


@lru_cache()
def get_settings() -> Settings:
    """Read environment variables and construct the singleton Settings instance."""
    env_vars = {
        "APP_NAME": os.getenv("APP_NAME", "EchoMind"),
        "APP_VERSION": os.getenv("APP_VERSION", "1.0.0"),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": int(os.getenv("PORT", "8000")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", ""),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "base"),
        "WHISPER_DEVICE": os.getenv("WHISPER_DEVICE", "cpu"),
        "WHISPER_COMPUTE_TYPE": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CHROMA_PATH": os.getenv("CHROMA_PATH", "vector_database"),
        "DATA_PATH": os.getenv("DATA_PATH", "downloads"),
        "SESSIONS_FILE": os.getenv("SESSIONS_FILE", "sessions.json"),
        "CORS_ORIGINS": os.getenv(
            "CORS_ORIGINS",
            "http://localhost,http://localhost:80,http://localhost:8000,http://localhost:5500,http://127.0.0.1:5500,http://127.0.0.1:8000",
        ),
        "MAX_UPLOAD_SIZE_MB": int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")),
        "MAX_VIDEO_DURATION_MINUTES": int(
            os.getenv("MAX_VIDEO_DURATION_MINUTES", "180")
        ),
    }
    return Settings(**env_vars)


# Export singleton
settings = get_settings()
