"""
Security module — Input validation, sanitization, and rate limiting.
"""

import re
import html
import time
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    MAX_TEXT_LENGTH: int = 1000
    MIN_TEXT_LENGTH: int = 3
    MAX_BATCH_SIZE: int = 20
    RATE_LIMIT_WINDOW: int = 60          # seconds
    RATE_LIMIT_MAX_REQUESTS: int = 20    # max requests per window
    BLOCKED_PATTERNS: tuple = (
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    )


CONFIG = SecurityConfig()


def sanitize_text(text: str) -> str:
    """Remove dangerous HTML/script content and normalize whitespace."""
    if not isinstance(text, str):
        return ""

    # HTML-escape to neutralize any injection
    text = html.escape(text, quote=True)

    # Remove any residual script-like patterns
    for pattern in CONFIG.BLOCKED_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_text_input(text: str) -> tuple[bool, str]:
    """
    Validate text input. Returns (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze."

    if len(text.strip()) < CONFIG.MIN_TEXT_LENGTH:
        return False, f"Text must be at least {CONFIG.MIN_TEXT_LENGTH} characters."

    if len(text) > CONFIG.MAX_TEXT_LENGTH:
        return False, f"Text exceeds maximum length of {CONFIG.MAX_TEXT_LENGTH} characters. Current: {len(text)}."

    return True, ""


def check_rate_limit(session_state) -> tuple[bool, str]:
    """
    Session-based rate limiting. Returns (is_allowed, message).
    """
    now = time.time()

    if "rate_limit_requests" not in session_state:
        session_state.rate_limit_requests = []

    # Prune old requests outside the window
    session_state.rate_limit_requests = [
        t for t in session_state.rate_limit_requests
        if now - t < CONFIG.RATE_LIMIT_WINDOW
    ]

    if len(session_state.rate_limit_requests) >= CONFIG.RATE_LIMIT_MAX_REQUESTS:
        remaining = int(
            CONFIG.RATE_LIMIT_WINDOW
            - (now - session_state.rate_limit_requests[0])
        )
        return False, f"Rate limit reached. Please wait {remaining}s before trying again."

    # Record this request
    session_state.rate_limit_requests.append(now)
    return True, ""


def validate_batch_input(texts: list[str]) -> tuple[bool, str]:
    """Validate batch input."""
    if not texts:
        return False, "No texts provided."

    if len(texts) > CONFIG.MAX_BATCH_SIZE:
        return False, f"Batch size exceeds maximum of {CONFIG.MAX_BATCH_SIZE}. Got {len(texts)}."

    for i, text in enumerate(texts):
        is_valid, msg = validate_text_input(text)
        if not is_valid:
            return False, f"Line {i + 1}: {msg}"

    return True, ""
