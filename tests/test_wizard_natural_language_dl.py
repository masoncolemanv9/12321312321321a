"""Tests for the natural-language «скачай <URL>» dispatcher and the
«📝 Markitdown» settings screen plumbing.

We test the helpers and the regex matchers directly — not the live
aiogram routing — so the tests stay deterministic and fast.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot import wizard
from bot.workers import downloader as downloader_mod
from bot.workers.downloader import DownloaderWorker


# ---- regex matchers -----------------------------------------------------


@pytest.mark.parametrize(
    "text, url",
    [
        ("скачай https://www.instagram.com/reel/abc", "https://www.instagram.com/reel/abc"),
        ("Скачать https://www.youtube.com/shorts/xyz", "https://www.youtube.com/shorts/xyz"),
        ("  скачай  https://example.com/video.mp4  ", "https://example.com/video.mp4"),
        ("СКАЧАЙ https://t.co/foo", "https://t.co/foo"),
    ],
)
def test_download_phrase_regex_matches(text: str, url: str) -> None:
    m = wizard._DOWNLOAD_PHRASE_RE.match(text)
    assert m is not None
    assert m.group(1) == url


@pytest.mark.parametrize(
    "text",
    [
        "скачать-видео https://example.com",  # hyphen breaks the verb
        "Просто текст без URL",
        "/dl https://example.com",  # slash command, not natural lang
        "https://example.com",  # bare URL, no verb
        "загрузи https://example.com",  # different verb
    ],
)
def test_download_phrase_regex_no_match(text: str) -> None:
    assert wizard._DOWNLOAD_PHRASE_RE.match(text) is None


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.instagram.com/reel/abc", True),
        ("https://www.youtube.com/shorts/xyz", True),
        ("https://example.com/video.mp4", True),
        ("https://github.com/foo/bar", False),
        ("https://gitlab.com/foo/bar", False),
        ("https://bitbucket.org/foo/bar", False),
        ("ftp://example.com/v.mp4", False),  # not http(s)
    ],
)
def test_looks_like_video_url(url: str, expected: bool) -> None:
    assert wizard._looks_like_video_url(url) is expected


# ---- markitdown command regexes -----------------------------------------


def test_md_dump_regex_matches() -> None:
    text = "дай данные https://www.youtube.com/shorts/abc"
    m = wizard._MD_DUMP_RE.match(text)
    assert m is not None
    assert m.group(1) == "https://www.youtube.com/shorts/abc"


def test_md_summary_regex_matches_long_form() -> None:
    text = "посмотри и расскажи о чём там https://www.youtube.com/shorts/abc"
    m = wizard._MD_SUMMARY_RE.match(text)
    assert m is not None
    assert m.group(1) == "https://www.youtube.com/shorts/abc"


def test_md_summary_regex_matches_short_form() -> None:
    # the «о чём там» tail is optional — short version should also match
    text = "посмотри и расскажи https://example.com/foo"
    m = wizard._MD_SUMMARY_RE.match(text)
    assert m is not None
    assert m.group(1) == "https://example.com/foo"


# ---- choice persistence -------------------------------------------------


def test_downloader_choice_roundtrip(monkeypatch) -> None:
    """Setter writes the value, getter reads it back."""
    from bot.addons import state as addon_state

    # Snapshot + restore so test pollution doesn't leak
    original = addon_state.get("downloader", "choice")
    try:
        wizard._set_downloader_choice("dl")
        assert wizard._downloader_choice() == "dl"
        wizard._set_downloader_choice("bash2mp4")
        assert wizard._downloader_choice() == "bash2mp4"
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_downloader_choice_unknown_value_returns_none() -> None:
    from bot.addons import state as addon_state

    original = addon_state.get("downloader", "choice")
    try:
        addon_state.set_("downloader", "choice", "bogus")
        # Helper guards against garbage values from a hand-edited state.json
        assert wizard._downloader_choice() is None
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_pending_download_roundtrip() -> None:
    from bot.addons import state as addon_state

    addon_state.delete("downloader", "pending")
    try:
        wizard._store_pending_download(42, "https://x.test/v")
        # second chat — must not stomp on the first
        wizard._store_pending_download(43, "https://y.test/v")
        assert wizard._pop_pending_download(42) == "https://x.test/v"
        # pop is destructive — second call returns None
        assert wizard._pop_pending_download(42) is None
        assert wizard._pop_pending_download(43) == "https://y.test/v"
    finally:
        addon_state.delete("downloader", "pending")


# ---- markitdown helpers -------------------------------------------------


def test_markitdown_auto_default_off() -> None:
    from bot.addons import state as addon_state

    original = addon_state.get("markitdown", "auto")
    try:
        addon_state.delete("markitdown", "auto")
        assert wizard._markitdown_auto_enabled() is False
        wizard._set_markitdown_auto(True)
        assert wizard._markitdown_auto_enabled() is True
        wizard._set_markitdown_auto(False)
        assert wizard._markitdown_auto_enabled() is False
    finally:
        if original is None:
            addon_state.delete("markitdown", "auto")
        else:
            addon_state.set_("markitdown", "auto", original)


def test_markitdown_screen_text_advertises_commands() -> None:
    text = wizard._markitdown_screen_text()
    assert "дай данные" in text
    assert "посмотри и расскажи" in text
    assert "github.com/microsoft/markitdown" in text


def test_markitdown_keyboard_has_required_buttons(monkeypatch) -> None:
    # Stub installed = True so we get the full 5-button layout
    monkeypatch.setattr(wizard, "_markitdown_installed", lambda: True)
    monkeypatch.setattr(wizard, "_markitdown_auto_enabled", lambda: False)
    kb = wizard._kb_markitdown_menu()
    labels = [btn.text for row in kb.inline_keyboard for btn in row]
    # All five user-requested buttons present.
    assert any("Авто-включение" in l for l in labels)
    assert any("Авто-выключение" in l for l in labels)
    assert any("Установлен" in l or "Скачать" in l for l in labels)
    assert any("Удалить" in l for l in labels)
    assert any("Назад" in l for l in labels)


def test_markitdown_keyboard_skips_uninstall_when_not_installed(
    monkeypatch,
) -> None:
    # No uninstall button when there's nothing to uninstall.
    monkeypatch.setattr(wizard, "_markitdown_installed", lambda: False)
    monkeypatch.setattr(wizard, "_markitdown_auto_enabled", lambda: False)
    kb = wizard._kb_markitdown_menu()
    labels = [btn.text for row in kb.inline_keyboard for btn in row]
    assert not any("Удалить" in l for l in labels)
    # …but the install + on/off + back buttons are still there.
    assert any("Скачать с github" in l for l in labels)
    assert any("Назад" in l for l in labels)


# ---- downloader worker — simple_only flag ------------------------------


@pytest.mark.asyncio
async def test_enqueue_next_skips_chain_for_simple_only(monkeypatch) -> None:
    """A simple-only download must NOT auto-enqueue analyze."""
    worker = DownloaderWorker.__new__(DownloaderWorker)
    worker.queue = MagicMock()
    worker.queue.enqueue = AsyncMock()

    job = MagicMock()
    job.payload = {"simple_only": True}
    job.chat_id = 12345
    job.id = 7

    # Stub out the chat-side notifier so the test doesn't try to talk
    # to Telegram.
    notify_calls: list[tuple[int, int, dict]] = []

    async def fake_notify(chat_id, job_id, result):
        notify_calls.append((chat_id, job_id, result))

    monkeypatch.setattr(downloader_mod, "_notify_simple_download", fake_notify)

    await worker.enqueue_next(job, {"source_path": "/tmp/foo.mp4"})

    # No analyze job enqueued.
    assert worker.queue.enqueue.call_count == 0
    # But the user did get a notification.
    assert notify_calls == [(12345, 7, {"source_path": "/tmp/foo.mp4"})]


# ---- Mutually-exclusive downloader screens ------------------------------


def test_dl_screen_keyboard_when_inactive(monkeypatch) -> None:
    from bot.addons import state as addon_state

    original = addon_state.get("downloader", "choice")
    try:
        addon_state.delete("downloader", "choice")
        kb = wizard._kb_dl_screen()
        labels = [btn.text for row in kb.inline_keyboard for btn in row]
        # When /dl is NOT the active mode, the «Использовать» button
        # is shown so the user can flip it on.
        assert any("Использовать /dl" in l for l in labels)
        assert not any("Активный" in l for l in labels)
        assert any("Назад" in l for l in labels)
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_dl_screen_keyboard_when_active() -> None:
    from bot.addons import state as addon_state

    original = addon_state.get("downloader", "choice")
    try:
        addon_state.set_("downloader", "choice", "dl")
        kb = wizard._kb_dl_screen()
        labels = [btn.text for row in kb.inline_keyboard for btn in row]
        # When /dl IS active, the user sees «Активный режим» (read-only
        # indicator) instead of the «Использовать» CTA.
        assert any("Активный" in l for l in labels)
        assert not any("Использовать" in l for l in labels)
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_bash2mp4_screen_keyboard_when_installed_inactive(monkeypatch) -> None:
    from bot.addons import state as addon_state

    monkeypatch.setattr(wizard, "_bash2mp4_installed", lambda: True)
    original = addon_state.get("downloader", "choice")
    try:
        addon_state.set_("downloader", "choice", "dl")  # other mode active
        kb = wizard._kb_bash2mp4_screen()
        labels = [btn.text for row in kb.inline_keyboard for btn in row]
        assert any("Использовать bash2mp4" in l for l in labels)
        # Uninstall is always available when installed.
        assert any("Удалить bash2mp4" in l for l in labels)
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_bash2mp4_screen_keyboard_when_not_installed(monkeypatch) -> None:
    """When bash2mp4 isn't installed, the screen should offer install
    and NOT pretend «Использовать» is meaningful.
    """
    monkeypatch.setattr(wizard, "_bash2mp4_installed", lambda: False)
    from bot.addons import state as addon_state

    original = addon_state.get("downloader", "choice")
    try:
        addon_state.delete("downloader", "choice")
        kb = wizard._kb_bash2mp4_screen()
        labels = [btn.text for row in kb.inline_keyboard for btn in row]
        # Install CTA shown.
        assert any("Установить" in l for l in labels)
        # No use / remove buttons until installed.
        assert not any("Использовать bash2mp4" in l for l in labels)
        assert not any("Удалить bash2mp4" in l for l in labels)
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


def test_dl_screen_text_advertises_active_status() -> None:
    from bot.addons import state as addon_state

    original = addon_state.get("downloader", "choice")
    try:
        addon_state.set_("downloader", "choice", "dl")
        text = wizard._dl_screen_text()
        assert "Активен" in text
        # bash2mp4 is mentioned in the «отключает» disclaimer.
        assert "bash2mp4" in text
    finally:
        if original is None:
            addon_state.delete("downloader", "choice")
        else:
            addon_state.set_("downloader", "choice", original)


@pytest.mark.asyncio
async def test_enqueue_next_runs_chain_when_not_simple_only(monkeypatch) -> None:
    """The default download must still chain to analyze."""
    # We patch Worker.enqueue_next on the base class so we can detect
    # the super() call without exercising the real queue.
    from bot.workers import base as base_mod

    called: list[tuple] = []

    async def fake_super(self, job, result):
        called.append((job, result))

    monkeypatch.setattr(base_mod.Worker, "enqueue_next", fake_super)

    worker = DownloaderWorker.__new__(DownloaderWorker)
    worker.queue = MagicMock()

    job = MagicMock()
    job.payload = {}  # no simple_only flag
    job.chat_id = 12345
    job.id = 7

    await worker.enqueue_next(job, {"source_path": "/tmp/foo.mp4"})

    # super().enqueue_next was called → chain proceeds normally.
    assert len(called) == 1


# ---- YouTube cookies upload -------------------------------------------------


def test_yt_cookies_present_false_when_no_file(tmp_path, monkeypatch) -> None:
    """``_yt_cookies_present`` returns False when no cookies.txt exists."""
    monkeypatch.setattr(wizard, "_yt_cookies_path", lambda: tmp_path / "missing.txt")
    assert wizard._yt_cookies_present() is False


def test_yt_cookies_present_true_when_file_has_content(tmp_path, monkeypatch) -> None:
    """``_yt_cookies_present`` returns True for a non-empty file."""
    p = tmp_path / "yt_cookies.txt"
    p.write_text("# netscape cookies\n.youtube.com\tTRUE\t/\tTRUE\t9999\tFOO\tbar\n")
    monkeypatch.setattr(wizard, "_yt_cookies_path", lambda: p)
    assert wizard._yt_cookies_present() is True


def test_yt_cookies_present_false_for_empty_file(tmp_path, monkeypatch) -> None:
    """Zero-byte cookies file is treated as missing (defensive)."""
    p = tmp_path / "yt_cookies.txt"
    p.write_text("")
    monkeypatch.setattr(wizard, "_yt_cookies_path", lambda: p)
    assert wizard._yt_cookies_present() is False


def test_dl_screen_keyboard_includes_cookies_upload_when_missing(tmp_path, monkeypatch) -> None:
    """Cookies upload button appears on the /dl screen when no file present."""
    monkeypatch.setattr(wizard, "_yt_cookies_path", lambda: tmp_path / "missing.txt")
    kb = wizard._kb_dl_screen()
    flat = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Загрузить cookies" in t for t in flat)
    # Delete button should NOT appear when there's nothing to delete.
    assert not any("Удалить cookies" in t for t in flat)


def test_dl_screen_keyboard_includes_delete_when_present(tmp_path, monkeypatch) -> None:
    """When cookies are uploaded, screen shows «обновить» + «удалить» buttons."""
    p = tmp_path / "yt_cookies.txt"
    p.write_text("# cookies\n.youtube.com\tTRUE\t/\tTRUE\t9999\tFOO\tbar\n")
    monkeypatch.setattr(wizard, "_yt_cookies_path", lambda: p)
    kb = wizard._kb_dl_screen()
    flat = [b.text for row in kb.inline_keyboard for b in row]
    assert any("обновить" in t.lower() or "загружены" in t.lower() for t in flat)
    assert any("Удалить cookies" in t for t in flat)


# ---- yt-dlp bot-check fallback ---------------------------------------------


def test_is_yt_bot_check_detects_sign_in_message() -> None:
    """The «Sign in to confirm…» wording must trip the heuristic."""
    exc = RuntimeError(
        "yt-dlp pre-flight failed: ERROR: [youtube] PNJD6lEBrVA: "
        "Sign in to confirm you're not a bot."
    )
    assert downloader_mod._is_yt_bot_check(exc) is True


def test_is_yt_bot_check_ignores_unrelated_errors() -> None:
    """Random RuntimeErrors must NOT be misclassified as bot-checks."""
    assert downloader_mod._is_yt_bot_check(RuntimeError("video too large")) is False
    assert downloader_mod._is_yt_bot_check(RuntimeError("private video")) is False


def test_yt_player_client_fallbacks_are_not_empty() -> None:
    """Sanity: fallback list must include the canonical bypass clients."""
    fallbacks = downloader_mod._YT_PLAYER_CLIENT_FALLBACKS
    assert len(fallbacks) >= 2
    # ``tv_embedded`` and ``android`` are the two most-reliable bypasses
    # per yt-dlp's own issue tracker — both must be present.
    assert "tv_embedded" in fallbacks
    assert "android" in fallbacks


def test_ydl_options_accepts_player_client_override() -> None:
    """Passing ``player_client`` plumbs the right ``extractor_args``."""
    from pathlib import Path

    opts = downloader_mod._ydl_options(
        job_dir=Path("/tmp/fake"),
        log_path=Path("/tmp/fake/log.txt"),
        max_height=720,
        quiet=True,
        player_client="android",
    )
    assert opts["extractor_args"] == {"youtube": {"player_client": ["android"]}}


def test_ydl_options_omits_extractor_args_without_override() -> None:
    """Without explicit player_client we MUST NOT inject extractor_args."""
    from pathlib import Path

    opts = downloader_mod._ydl_options(
        job_dir=Path("/tmp/fake"),
        log_path=Path("/tmp/fake/log.txt"),
        max_height=720,
        quiet=True,
    )
    assert "extractor_args" not in opts
