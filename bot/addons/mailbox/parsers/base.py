"""Parser interface and shared HTML→text helpers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser

from ..models import FreelanceTask, KnownSource


@dataclass
class EmailContent:
    """Container of useful pieces extracted from an IMAP message before parsing."""

    uid: str
    from_address: str
    subject: str
    received_at: object | None
    text: str
    html: str
    links: list[str]
    # Source IMAP folder (e.g. "INBOX" or "Рассылки"); used so
    # ``get_task_details`` can re-fetch the message from the right place.
    folder: str = ""


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script", "head"):
            self._skip += 1
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._links.append(v)
        if tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("style", "script", "head"):
            self._skip = max(0, self._skip - 1)
        if tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        out = "".join(self._chunks)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n\s*\n+", "\n\n", out)
        return out.strip()

    @property
    def links(self) -> list[str]:
        return self._links


def html_to_text(html: str) -> tuple[str, list[str]]:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.text(), parser.links


class FreelanceParser(ABC):
    """Base class for site-specific parsers.

    Each parser declares its `id`, the substrings it recognises in the From: header,
    and a `parse()` method that receives a normalised :class:`EmailContent` and
    returns a :class:`FreelanceTask`.
    """

    id: str = ""
    display_name: str = ""
    sender_match: tuple[str, ...] = ()
    notes: str = ""

    def matches(self, from_address: str) -> bool:
        if not from_address:
            return False
        lowered = from_address.lower()
        return any(s.lower() in lowered for s in self.sender_match)

    @abstractmethod
    def parse(self, content: EmailContent) -> FreelanceTask:
        """Build a :class:`FreelanceTask` from a normalised email."""

    def describe(self) -> KnownSource:
        return KnownSource(
            id=self.id,
            display_name=self.display_name,
            sender_match=list(self.sender_match),
            notes=self.notes,
        )
