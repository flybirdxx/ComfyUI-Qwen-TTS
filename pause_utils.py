import re
from typing import Tuple


_LEADING_BREAK_PATTERN = re.compile(
    r"^\s*\[break=([0-9]+(?:\.[0-9]+)?)\]\s*",
    re.IGNORECASE,
)


def extract_leading_pause(text: str) -> Tuple[str, float]:
    """Remove leading break tags and return their combined duration."""
    pause_seconds = 0.0

    while match := _LEADING_BREAK_PATTERN.match(text):
        pause_seconds += float(match.group(1))
        text = text[match.end():]

    return text, pause_seconds
