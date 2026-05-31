"""Per-site email parsers."""

from .base import EmailContent, FreelanceParser
from .registry import REGISTRY, find_parser, list_known_sources
from .youdo import YouDoParser

__all__ = [
    "REGISTRY",
    "EmailContent",
    "FreelanceParser",
    "YouDoParser",
    "find_parser",
    "list_known_sources",
]
