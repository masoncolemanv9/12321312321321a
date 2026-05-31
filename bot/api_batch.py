"""Batch import parser for the «🔌 API» button.

Lets the user paste a single string that configures multiple secrets /
endpoints / models at once. The string format::

    <field>;<field>;...$$$<field>;<field>;...$$$...

Each ``<field>`` is one of:

* ``target:key:value`` — explicit (preferred for ``brain1`` / ``brain2``
  which have multiple writable fields).
* ``target:value``     — shorthand for targets that have a single
  meaningful field (external tools default to ``api``; ElevenLabs
  defaults to ``api``).

Separators:

* ``$$$`` (three dollar signs) — between independent blocks.
* ``;``                        — between fields inside one block.
* ``:``                        — between target / key / value (note:
  ``value`` may itself contain colons, e.g. ``https://...:8443/v1`` —
  the value is everything after the second colon, not split further).

Targets supported:

* ``brain1``, ``brain2`` — keys: ``api``, ``model``, ``url``,
  ``provider``.
* ``elevenlabs``       — keys: ``api``, ``voice``.
* ``tavily``, ``brave``, ``exa``, ``firecrawl``, ``apify``,
  ``github_pat`` (alias: ``github``) — key inferred as ``api`` when
  using the 2-part form.

The parser returns a :class:`BatchReport` describing applied + failed
fields so the wizard can show a single banner instead of crashing on
the first malformed token. Errors are friendly Russian strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Aliases the user might type instead of the canonical storage key.
_EXTERNAL_TOOL_ALIASES: dict[str, str] = {
    "github": "github_pat",
}
# Known external tools (keys in ``KNOWN_EXTERNAL_TOOLS``) duplicated
# here so the parser doesn't have to import storage at module load.
_KNOWN_EXTERNAL_TOOLS: tuple[str, ...] = (
    "apify", "firecrawl", "tavily", "brave", "exa", "github_pat",
)
_BRAIN_FIELDS: tuple[str, ...] = ("api", "model", "url", "provider")
_BRAIN_TARGETS: tuple[str, ...] = ("brain1", "brain2")
_ELEVENLABS_FIELDS: tuple[str, ...] = ("api", "voice")


@dataclass
class BatchReport:
    """Result of parsing a batch-API blob.

    ``applied`` is the list of human-readable labels ("brain1.api",
    "elevenlabs.voice", ...) we successfully persisted. ``errors`` is
    the list of friendly error strings, one per malformed field.
    Empty input is reported as a single error so the caller doesn't
    have to guard for it separately.
    """

    applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.applied) and not self.errors


def parse_and_apply(blob: str, storage_obj: object) -> BatchReport:
    """Parse ``blob`` and apply every recognised field to ``storage_obj``.

    ``storage_obj`` is the bot's storage singleton (``bot.storage.storage``).
    We use the public setters — ``set_openrouter_api_key`` etc. — and
    fall back to graceful errors when a target is unknown.

    Returns a :class:`BatchReport` describing what got persisted and
    what didn't. Never raises.
    """
    report = BatchReport()
    blob = (blob or "").strip()
    if not blob:
        report.errors.append("Пустая строка — нечего применять.")
        return report
    blocks = [b for b in blob.split("$$$") if b.strip()]
    if not blocks:
        report.errors.append("Не нашёл ни одного блока (разделитель — `$$$`).")
        return report
    for block in blocks:
        for raw_field in block.split(";"):
            raw_field = raw_field.strip()
            if not raw_field:
                continue
            _apply_one_field(raw_field, storage_obj, report)
    return report


def _apply_one_field(raw: str, storage_obj: object, report: BatchReport) -> None:
    """Parse a single ``target:key:value`` (or ``target:value``) field."""
    parts = raw.split(":", 2)
    if len(parts) < 2:
        report.errors.append(
            f"Не понял поле «{raw}» — формат: <code>target:key:value</code>."
        )
        return
    target = parts[0].strip().lower()
    if len(parts) == 2:
        key = None
        value = parts[1].strip()
    else:
        key = parts[1].strip().lower() or None
        value = parts[2].strip()
    if not value:
        report.errors.append(f"Пустое значение в «{raw}» — пропускаю.")
        return
    target = _EXTERNAL_TOOL_ALIASES.get(target, target)
    if target in _BRAIN_TARGETS:
        _apply_brain(target, key, value, storage_obj, report, raw)
    elif target == "elevenlabs":
        _apply_elevenlabs(key, value, storage_obj, report, raw)
    elif target in _KNOWN_EXTERNAL_TOOLS:
        _apply_external(target, key, value, storage_obj, report, raw)
    else:
        report.errors.append(
            f"Неизвестный target «{target}» в «{raw}». Поддерживаются: "
            "brain1, brain2, elevenlabs, "
            + ", ".join(_KNOWN_EXTERNAL_TOOLS)
            + "."
        )


def _apply_brain(
    target: str,
    key: str | None,
    value: str,
    storage_obj: object,
    report: BatchReport,
    raw: str,
) -> None:
    if key is None:
        report.errors.append(
            f"Для «{target}» нужен ключ (api / model / url / provider). "
            f"Пример: <code>{target}:api:KEY</code>."
        )
        return
    if key not in _BRAIN_FIELDS:
        report.errors.append(
            f"Неизвестное поле «{key}» для {target} в «{raw}». "
            f"Доступно: {', '.join(_BRAIN_FIELDS)}."
        )
        return
    try:
        if target == "brain1":
            _apply_brain1_field(key, value, storage_obj)
        else:
            # ``api`` shortcut → ``api_key`` field on the slot.
            slot_field = "api_key" if key == "api" else key
            slot_field = "base_url" if slot_field == "url" else slot_field
            storage_obj.set_brain_slot2_field(slot_field, value)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — surface failure as text
        report.errors.append(f"Не получилось сохранить {target}.{key}: {exc}")
        return
    report.applied.append(f"{target}.{key}")


def _apply_brain1_field(key: str, value: str, storage_obj: object) -> None:
    """Brain 1 lives at the top of storage settings (legacy fields).

    The bot always reads Brain 1's API key from the ``openrouter`` slot
    regardless of provider — see ``capture_api_key`` in wizard.py for
    the same convention.
    """
    if key == "api":
        storage_obj.set_provider_key("openrouter", value)  # type: ignore[attr-defined]
    elif key == "model":
        storage_obj.set_model(value)  # type: ignore[attr-defined]
    elif key == "url":
        storage_obj.set_base_url(value)  # type: ignore[attr-defined]
    elif key == "provider":
        storage_obj.set_provider(value)  # type: ignore[attr-defined]
    else:
        # _apply_brain already validated; this branch is unreachable.
        raise ValueError(f"unknown brain1 field: {key}")


def _apply_elevenlabs(
    key: str | None,
    value: str,
    storage_obj: object,
    report: BatchReport,
    raw: str,
) -> None:
    # Shorthand: ``elevenlabs:<key>`` (no explicit key) treats the value
    # as the API key. This mirrors the external-tools shortcut so users
    # don't have to type ``elevenlabs:api:...`` every time.
    effective = key or "api"
    if effective not in _ELEVENLABS_FIELDS:
        report.errors.append(
            f"Неизвестное поле «{effective}» для elevenlabs в «{raw}». "
            f"Доступно: {', '.join(_ELEVENLABS_FIELDS)}."
        )
        return
    try:
        if effective == "api":
            storage_obj.set_elevenlabs_api_key(value)  # type: ignore[attr-defined]
        else:  # voice
            storage_obj.set_elevenlabs_voice_id(value)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"Не получилось сохранить elevenlabs.{effective}: {exc}")
        return
    report.applied.append(f"elevenlabs.{effective}")


def _apply_external(
    target: str,
    key: str | None,
    value: str,
    storage_obj: object,
    report: BatchReport,
    raw: str,
) -> None:
    # External tools only have ``api`` so the shorthand ``tool:value``
    # always means ``tool:api:value``. Explicit keys other than ``api``
    # are reported as errors instead of silently coerced.
    effective = (key or "api").lower()
    if effective not in ("api", "key", "token"):
        report.errors.append(
            f"Для {target} есть только поле <code>api</code> — нашёл «{effective}» в «{raw}»."
        )
        return
    try:
        storage_obj.set_external_tool_key(target, value)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"Не получилось сохранить {target}.api: {exc}")
        return
    report.applied.append(f"{target}.api")
