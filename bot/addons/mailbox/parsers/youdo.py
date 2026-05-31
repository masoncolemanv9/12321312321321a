"""Parser for YouDo task-notification emails."""

from __future__ import annotations

import re

from ..models import FreelanceTask
from .base import EmailContent, FreelanceParser

PRICE_RE = re.compile(
    r"(до\s*[\d\s]+|от\s*[\d\s]+|[\d\s]+)\s*(₽|руб(?:\.|лей)?|RUB|USD|\$|€|EUR)",
    re.IGNORECASE,
)
CATEGORY_LINE_RE = re.compile(
    r"^[А-ЯЁA-Z][А-Яа-яЁёA-Za-z\- ]{2,60}\s*/\s*[А-ЯЁA-Z][А-Яа-яЁёA-Za-z\- ]{2,60}$",
    re.MULTILINE,
)
CREATED_RE = re.compile(r"Создано\s+([0-9]{1,2}\s+\S+)")
NOISE_LINES = (
    "youdo",
    "новое задание",
    "подобрали",
    "спешите",
    "исполнител",
    "откликн",
    "бюджет",
    "хочу обсудить",
    "посмотреть на сайте",
)


class YouDoParser(FreelanceParser):
    id = "youdo"
    display_name = "YouDo"
    sender_match = ("youdo.com", "@youdo")
    notes = "Notification emails from YouDo with task title, description, category and budget."

    def parse(self, content: EmailContent) -> FreelanceTask:
        text = content.text or ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = "\n".join(lines)

        # Subject pattern in YouDo emails is essentially "<task title> - <price>",
        # e.g. "Нарисовать инфографику - до 5 000 ₽". Splitting it out gives
        # us a clean title even when the HTML body is mostly chrome/whitespace.
        subject_title, subject_price = self._split_subject(content.subject)

        title = subject_title or self._extract_title(lines)
        description = self._extract_description(lines, title) if title else None
        category = self._extract_category(blob)
        # Prefer body-level price (richer match), fall back to subject price.
        price_raw, price_amount, currency = self._extract_price(blob)
        if not price_raw and subject_price:
            price_raw, price_amount, currency = self._extract_price(subject_price)
        created_label = self._extract_created(blob)
        apply_link = self._extract_apply_link(content.links)

        return FreelanceTask(
            uid=content.uid,
            source=self.id,
            received_at=content.received_at,
            from_address=content.from_address,
            subject=content.subject,
            title=title or content.subject or None,
            description=description,
            category=category,
            price=price_raw,
            price_amount=price_amount,
            currency=currency,
            created_label=created_label,
            apply_link=apply_link,
            raw_text_excerpt=text[:1500],
        )

    @staticmethod
    def _split_subject(subject: str) -> tuple[str | None, str | None]:
        """Split a YouDo subject like ``Title - до 5 000 ₽`` into (title, price)."""
        if not subject:
            return None, None
        s = subject.strip()
        # Locate the price tail (last "до/от/N+ currency" chunk in the subject).
        match = PRICE_RE.search(s)
        if not match:
            return s or None, None
        title_part = s[: match.start()].rstrip(" -–—\t").strip()
        price_part = s[match.start() :].strip()
        return (title_part or None), (price_part or None)

    @staticmethod
    def _extract_title(lines: list[str]) -> str | None:
        for i, ln in enumerate(lines):
            ln_low = ln.lower()
            if "новое задание" in ln_low or "подобрали" in ln_low:
                for cand in lines[i + 1 : i + 8]:
                    cand_low = cand.lower()
                    if any(noise in cand_low for noise in NOISE_LINES):
                        continue
                    if PRICE_RE.search(cand) or CATEGORY_LINE_RE.match(cand):
                        continue
                    if 5 < len(cand) < 250:
                        return cand
                break

        for ln in lines:
            ln_low = ln.lower()
            if any(noise in ln_low for noise in NOISE_LINES):
                continue
            if PRICE_RE.search(ln) or CATEGORY_LINE_RE.match(ln):
                continue
            if 10 < len(ln) < 250:
                return ln
        return None

    @staticmethod
    def _extract_description(lines: list[str], title: str) -> str | None:
        try:
            idx = lines.index(title)
        except ValueError:
            return None
        for cand in lines[idx + 1 : idx + 5]:
            cand_low = cand.lower()
            if cand == title or any(noise in cand_low for noise in NOISE_LINES):
                continue
            if PRICE_RE.search(cand) or CATEGORY_LINE_RE.match(cand):
                continue
            if "создано" in cand_low:
                continue
            if 5 < len(cand) < 800:
                return cand
        return None

    @staticmethod
    def _extract_category(blob: str) -> str | None:
        match = CATEGORY_LINE_RE.search(blob)
        if not match:
            return None
        return re.sub(r"\s+/\s+", " / ", match.group(0).strip())

    @staticmethod
    def _extract_price(blob: str) -> tuple[str | None, float | None, str | None]:
        match = PRICE_RE.search(blob)
        if not match:
            return None, None, None
        raw = re.sub(r"\s+", " ", match.group(0).strip())
        digits = re.findall(r"\d[\d\s]*", match.group(1))
        amount: float | None = None
        if digits:
            try:
                amount = float(digits[0].replace(" ", ""))
            except ValueError:
                amount = None
        currency_token = match.group(2).lower()
        currency = "RUB"
        if "$" in currency_token or "usd" in currency_token:
            currency = "USD"
        elif "€" in currency_token or "eur" in currency_token:
            currency = "EUR"
        return raw, amount, currency

    @staticmethod
    def _extract_created(blob: str) -> str | None:
        match = CREATED_RE.search(blob)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_apply_link(links: list[str]) -> str | None:
        candidates = [link for link in links if "youdo." in link.lower()]
        if not candidates:
            return None
        for link in candidates:
            low = link.lower()
            if "/t/" in low or "/tasks" in low or "task" in low:
                return link
        return candidates[0]
