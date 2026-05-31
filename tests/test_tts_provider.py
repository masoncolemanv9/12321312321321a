"""Tests for the TTS provider selector + ElevenLabs integration.

These cover storage's ``get_tts_provider`` / ``set_tts_provider`` and
the dispatch logic in ``bot.tts.synthesize_voice_ogg`` that routes
between local Supertonic and the ElevenLabs HTTP path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.storage import Storage


@pytest.fixture
def fresh_storage(tmp_path):
    """A Storage instance backed by ``tmp_path`` only.

    Critical: ``Storage.__init__`` takes a ``data_dir`` argument and
    derives ``state_file`` from it. If we don't pass the temp dir
    explicitly, ``Storage()`` falls back to the real ``DATA_DIR``
    (``/tmp/bot-data`` or ``bot-data/`` in dev) and our test mutations
    leak into the running bot's persisted state. Always pass tmp_path.
    """
    s = Storage(data_dir=tmp_path)
    s._state = {}
    return s


def test_default_provider_is_local(fresh_storage: Storage) -> None:
    assert fresh_storage.get_tts_provider() == "local"


def test_set_provider_persists(fresh_storage: Storage) -> None:
    fresh_storage.set_tts_provider("elevenlabs")
    assert fresh_storage.get_tts_provider() == "elevenlabs"


def test_set_unknown_provider_raises(fresh_storage: Storage) -> None:
    with pytest.raises(ValueError):
        fresh_storage.set_tts_provider("nonsense")


def test_clone_without_recording_downgrades_to_local(fresh_storage: Storage) -> None:
    """Picking ``clone`` is safe even if no clone is on disk."""
    fresh_storage.set_tts_provider("clone")
    # No custom voice path saved → reader downgrades to local.
    assert fresh_storage.get_tts_provider() == "local"
    fresh_storage.set_tts_custom_voice_path("/tmp/voice.json")
    # Now that a clone exists, reader returns the actual selection.
    assert fresh_storage.get_tts_provider() == "clone"


def test_elevenlabs_api_key_setter_and_getter(fresh_storage: Storage) -> None:
    fresh_storage.set_elevenlabs_api_key("el-secret-xxx")
    assert fresh_storage.get_elevenlabs_api_key() == "el-secret-xxx"


def test_elevenlabs_voice_id_setter_and_getter(fresh_storage: Storage) -> None:
    fresh_storage.set_elevenlabs_voice_id("21m00Tcm4TlvDq8ikWAM")
    assert fresh_storage.get_elevenlabs_voice_id() == "21m00Tcm4TlvDq8ikWAM"


def test_elevenlabs_env_var_fallback(monkeypatch, fresh_storage: Storage) -> None:
    """If no key is saved, ``ELEVENLABS_API_KEY`` from env is used."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    assert fresh_storage.get_elevenlabs_api_key() == "env-key"
    fresh_storage.set_elevenlabs_api_key("saved-key")
    # Saved key wins over env.
    assert fresh_storage.get_elevenlabs_api_key() == "saved-key"


def test_elevenlabs_setters_strip_whitespace(fresh_storage: Storage) -> None:
    fresh_storage.set_elevenlabs_api_key("  el-key  ")
    fresh_storage.set_elevenlabs_voice_id("  voice-id  ")
    assert fresh_storage.get_elevenlabs_api_key() == "el-key"
    assert fresh_storage.get_elevenlabs_voice_id() == "voice-id"


# ---- bot.tts dispatch -----------------------------------------------------


def test_synthesize_with_elevenlabs_missing_key_returns_none() -> None:
    """No API key / voice_id → ElevenLabs path returns None silently.

    ``_synthesize_elevenlabs`` does a local ``from .storage import
    storage`` so we patch the singleton at its real module path.
    """
    from bot import tts

    mock_storage = MagicMock()
    mock_storage.get_elevenlabs_api_key.return_value = ""
    mock_storage.get_elevenlabs_voice_id.return_value = ""
    mock_storage.ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
    with patch("bot.storage.storage", mock_storage):
        result = asyncio.run(tts._synthesize_elevenlabs("hello"))
    assert result is None


def test_synthesize_elevenlabs_passes_voice_id_into_url() -> None:
    """Verify the URL we hit includes the configured voice_id."""
    from bot import tts

    captured = {}

    class _MockResponse:
        status_code = 200
        content = b"fake-mp3-bytes"
        text = ""

    class _MockClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _MockResponse()

    # Mock storage with valid creds.
    storage_mock = MagicMock()
    storage_mock.get_elevenlabs_api_key.return_value = "test-key"
    storage_mock.get_elevenlabs_voice_id.return_value = "MyVoiceID"
    storage_mock.ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"

    with patch("bot.storage.storage", storage_mock):
        with patch("httpx.AsyncClient", _MockClient):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch("pathlib.Path.read_bytes", return_value=b"ogg-bytes"):
                    result = asyncio.run(tts._synthesize_elevenlabs("hi there"))

    assert "MyVoiceID" in captured["url"]
    assert captured["url"].endswith("/v1/text-to-speech/MyVoiceID")
    assert captured["headers"]["xi-api-key"] == "test-key"
    assert captured["json"]["text"] == "hi there"
    assert result == b"ogg-bytes"


def test_synthesize_elevenlabs_returns_none_on_4xx() -> None:
    """A 401 / 402 / 429 from ElevenLabs is logged and yields None."""
    from bot import tts

    class _MockResponse:
        status_code = 401
        content = b""
        text = '{"error":"unauthorized"}'

    class _MockClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **k):
            return _MockResponse()

    storage_mock = MagicMock()
    storage_mock.get_elevenlabs_api_key.return_value = "test-key"
    storage_mock.get_elevenlabs_voice_id.return_value = "v"
    storage_mock.ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"

    with patch("bot.storage.storage", storage_mock):
        with patch("httpx.AsyncClient", _MockClient):
            result = asyncio.run(tts._synthesize_elevenlabs("hi"))

    assert result is None
