"""Tests for the new voice picker / cloning storage methods.

These tests stay pure-Python — they exercise the storage getters /
setters and the in-memory voice-style resolution. The actual Supertonic
synth path is not exercised here (it would download a multi-hundred-MB
model on first run).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot.storage import Storage
from bot.tts import (
    BUILTIN_FEMALE_VOICES,
    BUILTIN_MALE_VOICES,
    BUILTIN_VOICES,
    _resolve_voice_style_sync,
)


@pytest.fixture
def fresh_storage(tmp_path: Path) -> Storage:
    return Storage(data_dir=tmp_path)


def test_tts_voice_defaults_to_M1(fresh_storage: Storage) -> None:
    assert fresh_storage.get_tts_voice() == "M1"
    assert fresh_storage.get_tts_custom_voice_path() == ""


def test_tts_voice_setter_persists_value(fresh_storage: Storage) -> None:
    fresh_storage.set_tts_voice("F3")
    assert fresh_storage.get_tts_voice() == "F3"


def test_tts_custom_voice_path_round_trip(fresh_storage: Storage) -> None:
    fresh_storage.set_tts_custom_voice_path("/tmp/my_voice.json")
    assert fresh_storage.get_tts_custom_voice_path() == "/tmp/my_voice.json"
    fresh_storage.clear_tts_custom_voice_path()
    assert fresh_storage.get_tts_custom_voice_path() == ""


def test_builtin_voices_constants() -> None:
    # The user asked for the 5+5 layout — verify the constants are
    # exactly what the wizard advertises so the picker UI stays honest.
    assert BUILTIN_MALE_VOICES == ("M1", "M2", "M3", "M4", "M5")
    assert BUILTIN_FEMALE_VOICES == ("F1", "F2", "F3", "F4", "F5")
    assert BUILTIN_VOICES == BUILTIN_MALE_VOICES + BUILTIN_FEMALE_VOICES


def test_resolve_voice_style_picks_builtin_when_no_custom_path() -> None:
    # Fake Supertonic TTS object that records which voice was asked for.
    fake = MagicMock()
    fake.get_voice_style.return_value = "STYLE_OBJ_F3"

    out = _resolve_voice_style_sync(fake, "F3", None)

    assert out == "STYLE_OBJ_F3"
    fake.get_voice_style.assert_called_once_with(voice_name="F3")
    fake.get_voice_style_from_path.assert_not_called()


def test_resolve_voice_style_prefers_custom_path_over_voice() -> None:
    fake = MagicMock()
    fake.get_voice_style_from_path.return_value = "STYLE_OBJ_CLONE"

    out = _resolve_voice_style_sync(fake, "M2", "/tmp/voice.json")

    assert out == "STYLE_OBJ_CLONE"
    fake.get_voice_style_from_path.assert_called_once_with("/tmp/voice.json")
    fake.get_voice_style.assert_not_called()


def test_resolve_voice_style_falls_back_to_default_on_unknown_voice() -> None:
    # An unknown voice name (e.g. "Z9") must not propagate to Supertonic;
    # we should silently fall back to the M1 default.
    fake = MagicMock()
    fake.get_voice_style.return_value = "STYLE_OBJ_M1"

    out = _resolve_voice_style_sync(fake, "Z9", None)

    assert out == "STYLE_OBJ_M1"
    fake.get_voice_style.assert_called_once_with(voice_name="M1")
