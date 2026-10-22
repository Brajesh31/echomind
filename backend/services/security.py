"""EchoMind — Security & Validation Services.

Provides protections against:
- SSRF (Server-Side Request Forgery) on video ingestion URLs.
- Path traversal and malicious filename injection.
- Invalid ChromaDB collection identifiers.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

# Supported video hostnames
ALLOWED_YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}

# Blocked private and internal IP networks
DISALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local / Cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),        # Current network
    ipaddress.ip_network("100.64.0.0/10"),    # Carrier-grade NAT
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]


def is_ip_private_or_restricted(ip_str: str) -> bool:
    """Return True if the IP address belongs to a private, loopback, or cloud-metadata range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or any(ip in net for net in DISALLOWED_NETWORKS)
        )
    except ValueError:
        return False


def validate_video_url(url: str) -> str:
    """Validate a video URL against SSRF and provider allowlist.

    Raises:
        ValueError: If the URL is malformed, uses an unsupported scheme, targets
                    an internal/private host, or is not an allowed video provider.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    cleaned_url = url.strip()

    try:
        parsed = urlparse(cleaned_url)
    except Exception as e:
        raise ValueError(f"Malformed URL: {e}") from e

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL is missing a valid hostname.")

    # 1. Block direct localhost / internal loopbacks
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"):
        raise ValueError(f"Target host '{hostname}' is not permitted (SSRF protection).")

    # 2. Check if hostname is an IP address
    if is_ip_private_or_restricted(hostname):
        raise ValueError(
            f"Access to private/restricted IP address '{hostname}' is forbidden."
        )

    # 3. Domain allowlist check: Must match supported YouTube domains
    domain_allowed = any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in ALLOWED_YOUTUBE_DOMAINS
    )
    if not domain_allowed:
        raise ValueError(
            f"Host '{hostname}' is not supported. EchoMind currently supports YouTube URLs "
            f"({', '.join(sorted(ALLOWED_YOUTUBE_DOMAINS))})."
        )

    # 4. Optional DNS resolution check to prevent DNS rebinding to local IPs
    try:
        addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in addr_info:
            resolved_ip = sockaddr[0]
            if is_ip_private_or_restricted(resolved_ip):
                raise ValueError(
                    f"Resolved IP '{resolved_ip}' for host '{hostname}' is in a private network."
                )
    except socket.gaierror:
        # Host could not be resolved; yt-dlp will fail gracefully if offline
        pass

    return cleaned_url


def safe_token(value: str, max_len: int = 64) -> str:
    """Sanitize user-supplied identifiers (session IDs, filenames) against path traversal.

    Replaces path separators and disallowed characters with an underscore.
    """
    if not value:
        return "default"
    # Strip trailing slashes first so basename doesn't return empty string
    raw = str(value).strip().rstrip("/\\")
    clean = os.path.basename(raw)
    # Allow alphanumeric, underscore, and hyphen only
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", clean)
    clean = clean.strip("._-")
    return clean[:max_len] or "default"


def sanitize_collection_name(raw_name: str) -> str:
    """Sanitize a session ID to comply with Chroma collection naming rules:

    - Length between 3 and 63 characters
    - Starts and ends with an alphanumeric character
    - Contains only alphanumeric characters, underscores, or hyphens
    - No two consecutive periods
    """
    token = safe_token(raw_name, max_len=60)
    # Ensure starts and ends with alphanumeric
    token = re.sub(r"^[^A-Za-z0-9]+", "", token)
    token = re.sub(r"[^A-Za-z0-9]+$", "", token)

    if len(token) < 3:
        token = f"col_{token}".ljust(3, "0")
    return token[:63]
