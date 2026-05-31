"""Registry of all known site-specific parsers."""

from __future__ import annotations

from ..models import KnownSource
from .base import FreelanceParser
from .youdo import YouDoParser

REGISTRY: list[FreelanceParser] = [
    YouDoParser(),
]


def find_parser(from_address: str) -> FreelanceParser | None:
    """Return the parser whose `sender_match` matches the From: header."""
    for parser in REGISTRY:
        if parser.matches(from_address):
            return parser
    return None


def list_known_sources() -> list[KnownSource]:
    return [parser.describe() for parser in REGISTRY]
