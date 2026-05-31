"""Tests for bot/tts.py — focused on the sanitizer (no model required).

The Supertonic synthesis path is not unit-tested here because it would
download a 99M-parameter model from Hugging Face on first run; we only
verify the text-transformation contract, which is what the user can
actually inspect from a Telegram voice message.
"""

from __future__ import annotations

import pytest

from bot.tts import sanitize_for_tts


@pytest.mark.parametrize(
    "raw,expected_pieces",
    [
        # Fenced code blocks vanish completely.
        (
            "Сначала обычный текст.\n```bash\npip install foo\nrun --bar\n```\n"
            "А потом ещё текст.",
            ["Сначала обычный текст", "А потом ещё текст"],
        ),
        # Inline code (backticks) vanishes too.
        (
            "Используй `subprocess.run` чтобы вызвать процесс.",
            ["Используй", "чтобы вызвать процесс"],
        ),
        # HTML <pre>...</pre> blocks vanish.
        (
            "Лог:<pre>err: not found\nexit 1</pre>Конец.",
            ["Лог:", "Конец"],
        ),
        # GitHub URL becomes "репо от создателей <name>".
        (
            "Смотри https://github.com/apify/agent-skills для подробностей.",
            ["Смотри", "репо от создателей agent-skills", "для подробностей"],
        ),
        # Non-github URL → bare domain.
        (
            "Сходи на https://duckduckgo.com/?q=test и проверь.",
            ["Сходи на", "duckduckgo.com", "и проверь"],
        ),
        # `www.` prefix is stripped.
        (
            "Открой https://www.example.org/path",
            ["Открой", "example.org"],
        ),
    ],
)
def test_sanitize_strips_code_and_shortens_urls(
    raw: str, expected_pieces: list[str]
) -> None:
    out = sanitize_for_tts(raw)
    for piece in expected_pieces:
        assert piece in out, f"expected '{piece}' in {out!r}"
    # Nothing that looks like a raw URL should leak through.
    assert "http" not in out, f"raw URL leaked: {out!r}"
    assert "```" not in out
    assert "<pre>" not in out
    assert "<code>" not in out


def test_sanitize_returns_empty_when_only_code() -> None:
    raw = "```\nonly code, nothing else\n```"
    assert sanitize_for_tts(raw) == ""


def test_sanitize_handles_empty_and_whitespace() -> None:
    assert sanitize_for_tts("") == ""
    assert sanitize_for_tts("   \n\n\t  ") == ""


def test_sanitize_truncates_long_text() -> None:
    raw = "слово " * 2000
    out = sanitize_for_tts(raw)
    assert 0 < len(out) <= 2000


def test_sanitize_decodes_basic_html_entities() -> None:
    raw = "5 &gt; 3 &amp; 1 &lt; 2"
    out = sanitize_for_tts(raw)
    assert out == "5 > 3 & 1 < 2"


def test_sanitize_github_url_without_repo_name() -> None:
    # Just the user — no repo path; should still produce something.
    out = sanitize_for_tts("Профиль: https://github.com/octocat")
    assert "github" not in out  # no raw domain
    assert "репо" in out or "octocat" in out


# ---- RAM guard ------------------------------------------------------------
# Supertonic peaks ~470 MB during synthesis. On a 512 MB Render Free
# box that OOM-kills the bot mid-reply. The guard reads cgroup
# accounting and skips synthesis with a clear log line instead.


def test_ram_guard_blocks_when_low_memory(monkeypatch) -> None:
    """`synthesize_voice_ogg` returns None and never touches the model
    when ``_available_ram_mb()`` reports less than the configured
    minimum. The bot stays alive instead of OOM-restarting."""
    import asyncio

    from bot import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_available_ram_mb", lambda: 300)
    monkeypatch.setattr(tts_mod, "_TTS_MIN_AVAILABLE_MB", 450)
    called = {"load": False}

    async def _fail_load() -> None:
        called["load"] = True
        raise AssertionError("must not even attempt to load the model")

    monkeypatch.setattr(tts_mod, "_ensure_loaded", _fail_load)

    out = asyncio.run(tts_mod.synthesize_voice_ogg("привет"))
    assert out is None
    assert called["load"] is False
    # And the reason surfaces through the settings screen helper.
    reason = tts_mod.get_unavailable_reason()
    assert reason is not None
    assert "RAM" in reason


def test_ram_guard_passes_when_plenty(monkeypatch) -> None:
    """No RAM block when the host has headroom; the original load path
    runs normally (we don't simulate full Supertonic init here)."""
    from bot import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_available_ram_mb", lambda: 2048)
    monkeypatch.setattr(tts_mod, "_TTS_MIN_AVAILABLE_MB", 450)
    # Persistent reason should also be clear for this test.
    monkeypatch.setattr(tts_mod, "_tts_unavailable_reason", None)
    assert tts_mod._ram_blocks_tts() is None
    assert tts_mod.get_unavailable_reason() is None


def test_available_ram_handles_missing_files(monkeypatch) -> None:
    """`_available_ram_mb()` returns None gracefully when neither
    ``/proc/meminfo`` nor cgroup files are readable (non-Linux dev box,
    very locked-down container)."""
    from bot import tts as tts_mod

    def _no_open(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", _no_open)
    assert tts_mod._available_ram_mb() is None
