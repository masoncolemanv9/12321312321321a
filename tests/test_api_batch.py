"""Tests for the «🔌 API» batch-import parser.

Covers the round-trip from the user's pasted blob through
:func:`bot.api_batch.parse_and_apply` and into the storage singleton.
We mock storage as a plain Python object recording every setter call
so the tests don't touch disk.
"""

from __future__ import annotations

import pytest

from bot.api_batch import parse_and_apply


class _RecordingStorage:
    """Test double for bot.storage.storage — records what the parser set."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        # Stand-ins for whatever storage.* setters the parser may hit.
        self.brain1_api: str | None = None
        self.brain1_model: str | None = None
        self.brain1_url: str | None = None
        self.brain1_provider: str | None = None
        self.brain2: dict[str, str] = {}
        self.el_api: str | None = None
        self.el_voice: str | None = None
        self.ext: dict[str, str] = {}

    # Brain 1
    def set_provider_key(self, provider: str, key: str) -> None:
        self.calls.append(("set_provider_key", (provider, key), {}))
        if provider == "openrouter":
            self.brain1_api = key

    def set_model(self, model: str) -> None:
        self.calls.append(("set_model", (model,), {}))
        self.brain1_model = model

    def set_base_url(self, url: str) -> None:
        self.calls.append(("set_base_url", (url,), {}))
        self.brain1_url = url

    def set_provider(self, provider: str) -> None:
        self.calls.append(("set_provider", (provider,), {}))
        self.brain1_provider = provider

    # Brain 2
    def set_brain_slot2_field(self, field: str, value: str) -> None:
        self.calls.append(("set_brain_slot2_field", (field, value), {}))
        self.brain2[field] = value

    # ElevenLabs
    def set_elevenlabs_api_key(self, key: str) -> None:
        self.calls.append(("set_elevenlabs_api_key", (key,), {}))
        self.el_api = key

    def set_elevenlabs_voice_id(self, vid: str) -> None:
        self.calls.append(("set_elevenlabs_voice_id", (vid,), {}))
        self.el_voice = vid

    # External tools
    def set_external_tool_key(self, tool: str, key: str) -> None:
        self.calls.append(("set_external_tool_key", (tool, key), {}))
        self.ext[tool] = key


def test_empty_blob_returns_one_error() -> None:
    storage = _RecordingStorage()
    report = parse_and_apply("", storage)
    assert report.applied == []
    assert len(report.errors) == 1
    assert "Пустая" in report.errors[0]


def test_whitespace_only_blob_returns_one_error() -> None:
    storage = _RecordingStorage()
    report = parse_and_apply("   \n\t  ", storage)
    assert report.applied == []
    assert len(report.errors) == 1


def test_brain1_full_block_applies_all_four_fields() -> None:
    storage = _RecordingStorage()
    blob = (
        "brain1:provider:openrouter;brain1:api:sk-or-xxx;"
        "brain1:model:anthropic/claude-3.5-sonnet;"
        "brain1:url:https://openrouter.ai/api/v1"
    )
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert set(report.applied) == {"brain1.provider", "brain1.api", "brain1.model", "brain1.url"}
    assert storage.brain1_provider == "openrouter"
    assert storage.brain1_api == "sk-or-xxx"
    assert storage.brain1_model == "anthropic/claude-3.5-sonnet"
    assert storage.brain1_url == "https://openrouter.ai/api/v1"


def test_brain2_block_translates_api_to_api_key_and_url_to_base_url() -> None:
    storage = _RecordingStorage()
    blob = (
        "brain2:provider:custom;brain2:api:gsk_yyy;"
        "brain2:model:whisper-large-v3;brain2:url:https://api.groq.com/openai/v1"
    )
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    # api → api_key, url → base_url
    assert storage.brain2["api_key"] == "gsk_yyy"
    assert storage.brain2["base_url"] == "https://api.groq.com/openai/v1"
    assert storage.brain2["model"] == "whisper-large-v3"
    assert storage.brain2["provider"] == "custom"


def test_three_blocks_separated_by_triple_dollar() -> None:
    storage = _RecordingStorage()
    blob = (
        "brain1:api:sk-1$$$brain2:api:gsk_2$$$tavily:api:tvly-3"
    )
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert set(report.applied) == {"brain1.api", "brain2.api", "tavily.api"}
    assert storage.brain1_api == "sk-1"
    assert storage.brain2["api_key"] == "gsk_2"
    assert storage.ext["tavily"] == "tvly-3"


def test_external_tool_shorthand_two_part_form() -> None:
    storage = _RecordingStorage()
    blob = "tavily:tvly-xxx$$$brave:bravo-yyy$$$apify:apify_zzz"
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.ext == {"tavily": "tvly-xxx", "brave": "bravo-yyy", "apify": "apify_zzz"}


def test_github_alias_maps_to_github_pat() -> None:
    storage = _RecordingStorage()
    blob = "github:ghp_abcdef"
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.ext == {"github_pat": "ghp_abcdef"}


def test_elevenlabs_shorthand_treats_value_as_api_key() -> None:
    storage = _RecordingStorage()
    blob = "elevenlabs:el-key-xxx;elevenlabs:voice:21m00Tcm4TlvDq8ikWAM"
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.el_api == "el-key-xxx"
    assert storage.el_voice == "21m00Tcm4TlvDq8ikWAM"


def test_unknown_target_collects_error_and_doesnt_break_others() -> None:
    storage = _RecordingStorage()
    blob = "brain1:api:sk-good;unknown:foo:bar$$$tavily:api:tvly-2"
    report = parse_and_apply(blob, storage)
    assert storage.brain1_api == "sk-good"
    assert storage.ext == {"tavily": "tvly-2"}
    assert any("unknown" in e for e in report.errors)


def test_brain1_unknown_field_collects_error() -> None:
    storage = _RecordingStorage()
    blob = "brain1:nonsense:value"
    report = parse_and_apply(blob, storage)
    assert storage.brain1_api is None
    assert any("nonsense" in e for e in report.errors)


def test_brain_target_without_key_returns_error_not_crash() -> None:
    storage = _RecordingStorage()
    blob = "brain1:lonelyvalue"  # only 2 parts → key inferred None
    report = parse_and_apply(blob, storage)
    # Brain* doesn't have a default field — must report an error.
    assert any("brain1" in e for e in report.errors)
    assert report.applied == []


def test_value_with_colons_is_preserved() -> None:
    """URLs contain colons; the parser must split only on the first two."""
    storage = _RecordingStorage()
    blob = "brain2:url:https://api.example.com:8443/v1"
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.brain2["base_url"] == "https://api.example.com:8443/v1"


def test_empty_blocks_around_separators_are_skipped() -> None:
    storage = _RecordingStorage()
    blob = "$$$brain1:api:sk-xxx$$$$$$"  # leading/trailing/double separators
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.brain1_api == "sk-xxx"


def test_empty_fields_inside_block_skipped_silently() -> None:
    storage = _RecordingStorage()
    blob = "brain1:api:sk-xxx;;brain1:model:llama"
    report = parse_and_apply(blob, storage)
    assert report.errors == []
    assert storage.brain1_api == "sk-xxx"
    assert storage.brain1_model == "llama"


def test_external_tool_with_unknown_field_returns_error() -> None:
    storage = _RecordingStorage()
    blob = "tavily:secret:tvly-xxx"
    report = parse_and_apply(blob, storage)
    assert storage.ext == {}
    assert any("tavily" in e for e in report.errors)


def test_report_ok_property() -> None:
    storage = _RecordingStorage()
    good = parse_and_apply("brain1:api:sk", storage)
    assert good.ok is True
    bad = parse_and_apply("garbage", storage)
    assert bad.ok is False
    partial = parse_and_apply("brain1:api:sk;junkfield", storage)
    assert partial.ok is False  # has applied entries AND errors


def test_storage_exception_surfaces_as_error_not_crash() -> None:
    class _Boom:
        def set_provider_key(self, provider: str, key: str) -> None:
            raise RuntimeError("disk is on fire")

    report = parse_and_apply("brain1:api:sk-xxx", _Boom())
    assert report.applied == []
    assert any("disk is on fire" in e for e in report.errors)


@pytest.mark.parametrize(
    "form",
    [
        "tavily:api:tvly-1",
        "tavily:key:tvly-1",
        "tavily:token:tvly-1",
        "tavily:tvly-1",
    ],
)
def test_external_tool_accepts_api_key_token_aliases(form: str) -> None:
    storage = _RecordingStorage()
    report = parse_and_apply(form, storage)
    assert report.errors == []
    assert storage.ext == {"tavily": "tvly-1"}
