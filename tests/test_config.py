"""Tests for EchoMind Centralized Configuration."""

from pathlib import Path
from backend.config import Settings, get_settings


def test_default_configuration():
    """Verify default configuration attributes and types."""
    settings = get_settings()

    assert settings.app_name == "EchoMind"
    assert settings.app_version == "1.0.0"
    assert settings.port == 8000
    assert "youtube.com" not in settings.cors_origins_list  # CORS should be local origins
    assert "http://localhost" in settings.cors_origins_list
    assert ".mp4" in settings.allowed_av_extensions
    assert ".wav" in settings.allowed_av_extensions
    assert ".txt" in settings.transcript_extensions


def test_cors_origins_parsing():
    """Verify CORS comma-separated string parsing."""
    s = Settings(CORS_ORIGINS="http://example.com, https://app.echomind.ai , http://localhost:3000")
    origins = s.cors_origins_list

    assert len(origins) == 3
    assert "http://example.com" in origins
    assert "https://app.echomind.ai" in origins
    assert "http://localhost:3000" in origins


def test_path_resolution():
    """Verify absolute path resolution."""
    settings = get_settings()

    assert isinstance(settings.absolute_chroma_path, Path)
    assert isinstance(settings.absolute_data_path, Path)
    assert isinstance(settings.absolute_sessions_path, Path)
    assert settings.absolute_chroma_path.is_absolute()
    assert settings.absolute_data_path.is_absolute()


def test_mistral_configured_check():
    """Verify is_mistral_configured correctly detects missing or placeholder keys."""
    empty_settings = Settings(MISTRAL_API_KEY="")
    assert not empty_settings.is_mistral_configured

    placeholder_settings = Settings(MISTRAL_API_KEY="your_mistral_api_key_here")
    assert not placeholder_settings.is_mistral_configured

    valid_settings = Settings(MISTRAL_API_KEY="test_mock_valid_key_12345")
    assert valid_settings.is_mistral_configured
