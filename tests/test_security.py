"""Tests for Security Services: SSRF protection, URL validation, and token sanitization."""

import pytest
from backend.services.security import (
    is_ip_private_or_restricted,
    safe_token,
    sanitize_collection_name,
    validate_video_url,
)


class TestSSRFAndUrlValidation:
    """Validate that EchoMind strictly protects against SSRF and unauthorized URLs."""

    def test_valid_youtube_urls_allowed(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        ]
        for url in valid_urls:
            assert validate_video_url(url) == url

    def test_ssrf_rejects_localhost(self):
        with pytest.raises(ValueError, match="not permitted"):
            validate_video_url("http://localhost:8000/internal")

        with pytest.raises(ValueError, match="not permitted"):
            validate_video_url("http://127.0.0.1:80/admin")

        with pytest.raises(ValueError, match="not permitted"):
            validate_video_url("http://0.0.0.0:8000")

    def test_ssrf_rejects_private_ips(self):
        private_ips = [
            "http://10.0.0.1/video",
            "http://192.168.1.1/secret",
            "http://172.16.0.5/api",
            "http://169.254.169.254/latest/meta-data/",  # Cloud metadata
        ]
        for url in private_ips:
            with pytest.raises(ValueError):
                validate_video_url(url)

    def test_rejects_unsupported_domains(self):
        unsupported = [
            "https://vimeo.com/123456",
            "https://dailymotion.com/video/x123",
            "https://attacker.com/malicious.mp4",
            "https://google.com",
        ]
        for url in unsupported:
            with pytest.raises(ValueError, match="not supported"):
                validate_video_url(url)

    def test_rejects_non_http_schemes(self):
        invalid_schemes = [
            "file:///etc/passwd",
            "ftp://ftp.example.com/audio.wav",
            "javascript:alert(1)",
            "data:text/plain;base64,SGVsbG8=",
        ]
        for url in invalid_schemes:
            with pytest.raises(ValueError, match="scheme"):
                validate_video_url(url)

    def test_rejects_empty_or_malformed(self):
        with pytest.raises(ValueError):
            validate_video_url("")
        with pytest.raises(ValueError):
            validate_video_url("not a url")


class TestSafeToken:
    """Validate path sanitization to prevent path traversal."""

    def test_path_traversal_stripped(self):
        assert safe_token("../../etc/passwd") == "passwd"
        assert safe_token("..\\..\\windows\\system32") == "system32"

    def test_special_characters_replaced(self):
        assert safe_token("session; rm -rf /") == "session__rm_-rf"
        assert safe_token("user<alert(1)>") == "user_alert_1"

    def test_empty_value_fallback(self):
        assert safe_token("") == "default"
        assert safe_token("   ") == "default"
        assert safe_token("///") == "default"


class TestSanitizeCollectionName:
    """Validate Chroma collection name rules."""

    def test_valid_collection_names(self):
        col = sanitize_collection_name("user_123")
        assert 3 <= len(col) <= 63
        assert col[0].isalnum()
        assert col[-1].isalnum()

    def test_short_collection_names_padded(self):
        col = sanitize_collection_name("a")
        assert len(col) >= 3
        assert col[0].isalnum()
        assert col[-1].isalnum()

    def test_special_characters_sanitized(self):
        col = sanitize_collection_name("user@domain.com")
        assert "@" not in col
        assert "." not in col
