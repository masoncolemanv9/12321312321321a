import contextlib
import json
import os
from pathlib import Path

from .config import DATA_DIR, HISTORY_LIMIT, PROJECTS_DIR

# Settings live under this key inside state.json so they don't collide with
# user-id-keyed entries (Telegram user ids are stringified ints).
_SETTINGS_KEY = "_settings"

# LLM providers we know about. Each entry maps to a UI label and the env-var
# that acts as a fallback when nothing has been set via Telegram.
KNOWN_PROVIDERS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# External research/scraping tools that the bot may need to authenticate
# against. Researcher-type bots (github_scout, reddit_scout, apify_runner,
# etc.) read these via ``storage.get_external_tool_key("apify")`` when they
# perform their work. Each value is the env-var fallback name.
#
# Adding a new tool here automatically adds a button to the /setup wizard
# under "🔑 Внешние API" — no UI changes needed.
KNOWN_EXTERNAL_TOOLS: dict[str, dict[str, str]] = {
    "apify": {
        "env_var": "APIFY_API_TOKEN",
        "label": "Apify",
        "url": "https://console.apify.com/account/integrations",
        "hint": "Готовые scrap-actors для Reddit / TikTok / YT / Twitter и др.",
    },
    "firecrawl": {
        "env_var": "FIRECRAWL_API_KEY",
        "label": "Firecrawl",
        "url": "https://firecrawl.dev/app/api-keys",
        "hint": "Превращает любой сайт в чистый markdown для LLM.",
    },
    "tavily": {
        "env_var": "TAVILY_API_KEY",
        "label": "Tavily",
        "url": "https://app.tavily.com/home",
        "hint": "Search-API для AI-агентов (1000 запросов / мес бесплатно).",
    },
    "brave": {
        "env_var": "BRAVE_SEARCH_API_KEY",
        "label": "Brave Search",
        "url": "https://api-dashboard.search.brave.com/app/keys",
        "hint": "Альтернатива Google без трекинга (2000 / мес бесплатно).",
    },
    "exa": {
        "env_var": "EXA_API_KEY",
        "label": "Exa",
        "url": "https://dashboard.exa.ai/api-keys",
        "hint": "Семантический поиск (1000 / мес бесплатно).",
    },
    "github_pat": {
        "env_var": "GITHUB_RESEARCH_PAT",
        "label": "GitHub PAT (read-only)",
        "url": "https://github.com/settings/tokens",
        "hint": "Personal Access Token (Fine-grained, public-repo read) — поднимает rate-limit с 60 до 5000 запросов/час.",
    },
}
# NB: deAPI / fal / Replicate / Runway are now configured via the
# dedicated "🎬 Мозг для видео и фото" menu (see bot/media.py and the
# media wizard) — not via the generic external-tools list. This keeps
# the UI for "research API keys" separate from "media generation
# providers", which each have URL + key + multiple model slots.


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}***{key[-4:]}"


class Storage:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.state_file = data_dir / "state.json"
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))
        # Best-effort chmod — platforms without POSIX permissions silently fall through.
        with contextlib.suppress(OSError):
            os.chmod(self.state_file, 0o600)

    # ---- per-user state -------------------------------------------------

    def _user(self, user_id: int) -> dict:
        key = str(user_id)
        if key not in self._state:
            self._state[key] = {"cwd": None, "history": []}
        return self._state[key]

    def get_cwd(self, user_id: int) -> Path | None:
        cwd = self._user(user_id).get("cwd")
        if cwd is None:
            return None
        path = Path(cwd)
        return path if path.exists() else None

    def set_cwd(self, user_id: int, path: Path) -> None:
        self._user(user_id)["cwd"] = str(path)
        self._save()

    def get_history(self, user_id: int) -> list[dict]:
        return list(self._user(user_id).get("history", []))

    def append_history(self, user_id: int, message: dict) -> None:
        history = self._user(user_id).setdefault("history", [])
        history.append(message)
        cap = self.get_history_limit() * 2
        if len(history) > cap:
            del history[: len(history) - cap]
        self._save()

    def clear_history(self, user_id: int) -> None:
        self._user(user_id)["history"] = []
        self._save()

    def list_projects(self) -> list[str]:
        if not PROJECTS_DIR.exists():
            return []
        return sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())

    # ---- bot-wide settings (keys, model, enabled flag) ------------------

    def _settings(self) -> dict:
        return self._state.setdefault(_SETTINGS_KEY, {})

    def set_provider_key(self, provider: str, key: str) -> None:
        provider = provider.lower()
        if provider not in KNOWN_PROVIDERS:
            raise ValueError(
                f"unknown provider '{provider}'. known: {', '.join(KNOWN_PROVIDERS)}"
            )
        keys = self._settings().setdefault("keys", {})
        keys[provider] = key
        self._save()

    def delete_provider_key(self, provider: str) -> bool:
        provider = provider.lower()
        keys = self._settings().get("keys", {})
        if provider in keys:
            del keys[provider]
            self._save()
            return True
        return False

    def get_provider_key(self, provider: str) -> str:
        """Return the key for ``provider`` from state, falling back to env."""
        provider = provider.lower()
        stored = self._settings().get("keys", {}).get(provider)
        if stored:
            return stored
        env_var = KNOWN_PROVIDERS.get(provider)
        if env_var:
            return os.environ.get(env_var, "")
        return ""

    def list_provider_keys(self) -> dict[str, dict]:
        """List known providers with the source and a masked preview."""
        out: dict[str, dict] = {}
        keys = self._settings().get("keys", {})
        for provider, env_var in KNOWN_PROVIDERS.items():
            stored = keys.get(provider, "")
            env_value = os.environ.get(env_var, "")
            active = stored or env_value
            out[provider] = {
                "source": "telegram" if stored else ("env" if env_value else "none"),
                "masked": _mask_key(active),
                "env_var": env_var,
            }
        return out

    # ---- external research/scraping tools -------------------------------

    def set_external_tool_key(self, tool: str, key: str) -> None:
        tool = tool.lower()
        if tool not in KNOWN_EXTERNAL_TOOLS:
            raise ValueError(
                f"unknown external tool '{tool}'. known: "
                f"{', '.join(KNOWN_EXTERNAL_TOOLS)}"
            )
        tools = self._settings().setdefault("external_tools", {})
        tools[tool] = key
        self._save()

    def delete_external_tool_key(self, tool: str) -> bool:
        tool = tool.lower()
        tools = self._settings().get("external_tools", {})
        if tool in tools:
            del tools[tool]
            self._save()
            return True
        return False

    def get_external_tool_key(self, tool: str) -> str:
        """Return the key for ``tool`` from state, falling back to env.

        Used by researcher-type bots (github_scout, apify_runner, etc.) to
        authenticate against external APIs. Returns empty string if the
        tool was never configured — callers must handle that explicitly.
        """
        tool = tool.lower()
        stored = self._settings().get("external_tools", {}).get(tool)
        if stored:
            return stored
        meta = KNOWN_EXTERNAL_TOOLS.get(tool)
        if meta is not None:
            return os.environ.get(meta["env_var"], "")
        return ""

    def list_external_tool_keys(self) -> dict[str, dict]:
        """List known external tools with source + masked preview."""
        out: dict[str, dict] = {}
        tools = self._settings().get("external_tools", {})
        for tool, meta in KNOWN_EXTERNAL_TOOLS.items():
            stored = tools.get(tool, "")
            env_value = os.environ.get(meta["env_var"], "")
            active = stored or env_value
            out[tool] = {
                "source": "telegram" if stored else ("env" if env_value else "none"),
                "masked": _mask_key(active),
                "env_var": meta["env_var"],
                "label": meta["label"],
                "url": meta["url"],
                "hint": meta["hint"],
            }
        return out

    # ---- media brain (img2video / img2img / rmbg) -----------------------
    #
    # The "Мозг для видео и фото" UI keeps up to 3 slots. Each slot is a
    # provider config (URL + API key + per-task model). One slot is
    # active for each task ("video", "photo", "rmbg"). The user can mix
    # providers — e.g. deAPI for video, fal.ai for photo — by pointing
    # different active_media_slots at different slots.

    MEDIA_TASKS = ("video", "photo", "rmbg")
    MEDIA_SLOT_IDS = ("1", "2", "3")

    def _media_root(self) -> dict:
        return self._settings().setdefault(
            "media",
            {"slots": {}, "active": {}},
        )

    def get_media_slot_raw(self, slot_id: str) -> dict:
        """Return the raw slot dict (creates an empty one if missing)."""
        slot_id = str(slot_id)
        slots = self._media_root().setdefault("slots", {})
        return slots.setdefault(slot_id, {})

    def list_media_slots_raw(self) -> dict[str, dict]:
        slots = self._media_root().get("slots", {})
        return {sid: dict(slots.get(sid, {})) for sid in self.MEDIA_SLOT_IDS}

    def get_media_slot(self, slot_id: str):
        """Return a typed ``MediaSlot`` view (or ``None`` if completely empty)."""
        from .media import MediaSlot, detect_provider_name

        slot_id = str(slot_id)
        raw = self._media_root().get("slots", {}).get(slot_id)
        if not raw:
            return None
        url = raw.get("url", "")
        return MediaSlot(
            slot_id=slot_id,
            url=url,
            api_key=raw.get("api_key", ""),
            name=raw.get("name") or detect_provider_name(url) or slot_id,
            video_model=raw.get("video_model", ""),
            photo_model=raw.get("photo_model", ""),
            rmbg_model=raw.get("rmbg_model", ""),
        )

    def set_media_slot_field(self, slot_id: str, field: str, value: str) -> None:
        """Update one field of a slot. URL changes auto-rename the slot."""
        from .media import detect_provider_name

        if field not in {
            "url", "api_key", "name", "video_model", "photo_model", "rmbg_model",
        }:
            raise ValueError(f"unknown media slot field: {field}")
        raw = self.get_media_slot_raw(slot_id)
        raw[field] = value
        if field == "url":
            # Auto-name based on URL match. User can override via "name".
            auto = detect_provider_name(value)
            if auto:
                raw["name"] = auto
        self._save()

    def delete_media_slot(self, slot_id: str) -> None:
        slot_id = str(slot_id)
        slots = self._media_root().get("slots", {})
        slots.pop(slot_id, None)
        active = self._media_root().setdefault("active", {})
        for task, sid in list(active.items()):
            if sid == slot_id:
                active.pop(task, None)
        self._save()

    def set_active_media_slot(self, task: str, slot_id: str | None) -> None:
        if task not in self.MEDIA_TASKS:
            raise ValueError(f"unknown media task: {task}")
        active = self._media_root().setdefault("active", {})
        if slot_id is None:
            active.pop(task, None)
        else:
            active[task] = str(slot_id)
        self._save()

    def get_active_media_slot(self, task: str):
        slot_id = self._media_root().get("active", {}).get(task)
        if slot_id is None:
            # No explicit selection — pick the first configured slot with
            # the relevant model set. Lets users get away with a single
            # configured slot used for everything.
            from .media import detect_provider_name  # noqa: F401  — keep import warm

            for sid in self.MEDIA_SLOT_IDS:
                slot = self.get_media_slot(sid)
                if slot is None:
                    continue
                model_attr = f"{task}_model" if task != "rmbg" else "rmbg_model"
                if slot.configured and getattr(slot, model_attr):
                    return slot
            return None
        return self.get_media_slot(slot_id)

    def set_model(self, model: str) -> None:
        self._settings()["model"] = model
        self._save()

    def get_model(self) -> str:
        from .config import DEFAULT_MODEL  # local import to avoid cycles

        return self._settings().get("model") or DEFAULT_MODEL

    def is_enabled(self) -> bool:
        return bool(self._settings().get("enabled", True))

    # ---- brain mode (auto vs devin) -------------------------------------

    # ---- role gate (gate that requires picking a /role) -----------------
    #
    # The original bot required the user to select a persona via /role
    # before any project-touching command (/exec /clone /work /...)
    # would respond. That gate is now opt-in: by default it's OFF so
    # the bot replies to any allowed user. Toggle it on via Settings →
    # «🎭 Роли» when you want the role-narrowing behaviour back.

    def get_role_gate_enabled(self) -> bool:
        """True if the role gate is enabled (commands require /role first)."""
        return bool(self._settings().get("role_gate_enabled", False))

    def set_role_gate_enabled(self, enabled: bool) -> None:
        self._settings()["role_gate_enabled"] = bool(enabled)
        self._save()

    # ---- TTS (голосовой ответчик) ----------------------------------------
    #
    # When enabled, every agent reply (the one produced by ``run_agent``
    # in ``handle_text``) is additionally synthesised via Supertonic and
    # sent as a Telegram voice message. Code blocks and bare URLs are
    # stripped from the spoken version so the audio stays listenable.
    # Default is OFF — opt in via Settings → «🔊 Голосовой ответчик».

    def get_tts_enabled(self) -> bool:
        """True when the bot should narrate text replies via Supertonic."""
        return bool(self._settings().get("tts_enabled", False))

    def set_tts_enabled(self, enabled: bool) -> None:
        self._settings()["tts_enabled"] = bool(enabled)
        self._save()

    # Voice selection for the on-device Supertonic TTS. ``tts_voice``
    # is one of the 10 built-in Supertonic names (M1..M5, F1..F5).
    # ``tts_custom_voice_path`` overrides it when set — it points at
    # a Voice-Builder JSON (zero-shot clone) on disk. Both default
    # to empty so the on-disk state stays small.

    def get_tts_voice(self) -> str:
        return str(self._settings().get("tts_voice", "M1") or "M1")

    def set_tts_voice(self, voice: str) -> None:
        self._settings()["tts_voice"] = str(voice or "M1")
        self._save()

    def get_tts_custom_voice_path(self) -> str:
        """Absolute path to the active user-cloned voice JSON, or empty."""
        return str(self._settings().get("tts_custom_voice_path", "") or "")

    def set_tts_custom_voice_path(self, path: str) -> None:
        self._settings()["tts_custom_voice_path"] = str(path or "")
        self._save()

    def clear_tts_custom_voice_path(self) -> None:
        self._settings()["tts_custom_voice_path"] = ""
        self._save()

    # ---- TTS provider selector (which engine speaks for the bot) --------
    #
    # ``"local"`` — Supertonic with one of the built-in voices (M1..F5).
    # ``"clone"`` — Supertonic seeded by the user's recorded clone JSON.
    # ``"elevenlabs"`` — ElevenLabs HTTP API (api.elevenlabs.io). Requires
    # a key + voice_id; if either is missing the synth call falls back to
    # ``local`` with a friendly explanatory message so we never silently
    # break the user's chat.

    _VALID_TTS_PROVIDERS = ("local", "clone", "elevenlabs")

    def get_tts_provider(self) -> str:
        """Active TTS provider. Defaults to ``"local"`` (Supertonic)."""
        raw = str(self._settings().get("tts_provider", "local") or "local")
        if raw not in self._VALID_TTS_PROVIDERS:
            return "local"
        # If user picked "clone" but no clone is on disk, fall through to
        # local so the chat doesn't silently break after a restart that
        # wiped /tmp on free Render.
        if raw == "clone" and not self.get_tts_custom_voice_path():
            return "local"
        return raw

    def set_tts_provider(self, provider: str) -> None:
        provider = (provider or "").strip().lower()
        if provider not in self._VALID_TTS_PROVIDERS:
            raise ValueError(
                f"unknown tts provider '{provider}', expected one of "
                f"{', '.join(self._VALID_TTS_PROVIDERS)}"
            )
        self._settings()["tts_provider"] = provider
        self._save()

    # ---- ElevenLabs (cloud TTS provider) ---------------------------------
    #
    # URL is hardcoded to ``https://api.elevenlabs.io`` so the user only
    # needs to paste an API key and a voice_id (copied from their voice
    # library page). Both default to empty so the bot stays fully local
    # until the user explicitly opts in.

    ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"

    def get_elevenlabs_api_key(self) -> str:
        stored = str(self._settings().get("elevenlabs_api_key", "") or "")
        if stored:
            return stored
        return os.environ.get("ELEVENLABS_API_KEY", "")

    def set_elevenlabs_api_key(self, key: str) -> None:
        self._settings()["elevenlabs_api_key"] = str(key or "").strip()
        self._save()

    def get_elevenlabs_voice_id(self) -> str:
        return str(self._settings().get("elevenlabs_voice_id", "") or "")

    def set_elevenlabs_voice_id(self, voice_id: str) -> None:
        self._settings()["elevenlabs_voice_id"] = str(voice_id or "").strip()
        self._save()

    # ---- secondary brain slot (Brain 2) ----------------------------------
    #
    # Brain 1 is the legacy top-level config (provider / base_url / model
    # / openrouter_api_key under ``_settings``) — unchanged for backward
    # compatibility. Brain 2 is a parallel slot stored in a nested dict
    # so the user can keep both an OpenRouter key and a Groq key
    # configured and switch / fail-over between them.

    def get_brain_slot2(self) -> dict:
        """Return Brain 2 raw dict (creates an empty one if missing)."""
        return self._settings().setdefault(
            "brain_slot2",
            {"api_key": "", "base_url": "", "model": "", "provider": "custom"},
        )

    def set_brain_slot2_field(self, field: str, value: str) -> None:
        if field not in {"api_key", "base_url", "model", "provider"}:
            raise ValueError(f"unknown brain slot 2 field: {field}")
        slot = self.get_brain_slot2()
        if field == "base_url":
            value = (value or "").rstrip("/")
        slot[field] = value
        self._save()

    def get_active_brain_slot(self) -> str:
        """Which brain slot is active: ``"1"`` (default) or ``"2"``."""
        raw = str(self._settings().get("active_brain_slot", "1"))
        return raw if raw in ("1", "2") else "1"

    def set_active_brain_slot(self, slot: str) -> None:
        slot = str(slot)
        if slot not in ("1", "2"):
            raise ValueError(f"unknown brain slot '{slot}', expected '1' or '2'")
        self._settings()["active_brain_slot"] = slot
        self._save()

    # ---- history limit (user-configurable message memory) ----------------

    def get_history_limit(self) -> int:
        """Max conversation pairs stored. Defaults to ``HISTORY_LIMIT``."""
        raw = self._settings().get("history_limit")
        if raw is not None:
            try:
                n = int(raw)
                if n > 0:
                    return n
            except (ValueError, TypeError):
                pass
        return HISTORY_LIMIT

    def set_history_limit(self, n: int) -> None:
        n = max(1, min(int(n), 500))
        self._settings()["history_limit"] = n
        self._save()

    # ---- RAM guard settings ----------------------------------------------

    def get_ram_limit_mb(self) -> int:
        """Soft RAM limit in MB. 0 means disabled."""
        return int(self._settings().get("ram_limit_mb", 0) or 0)

    def set_ram_limit_mb(self, mb: int) -> None:
        self._settings()["ram_limit_mb"] = max(0, int(mb))
        self._save()

    def get_ram_behavior(self) -> str:
        """``"compress"`` (shrink history at 80%, refuse at 95%) or
        ``"refuse"`` (just refuse at limit)."""
        v = self._settings().get("ram_behavior", "compress")
        return v if v in ("compress", "refuse") else "compress"

    def set_ram_behavior(self, mode: str) -> None:
        if mode not in ("compress", "refuse"):
            raise ValueError(f"unknown ram_behavior {mode!r}")
        self._settings()["ram_behavior"] = mode
        self._save()

    def get_ram_show(self) -> bool:
        """Show RSS in the status bubble on each agent step."""
        return bool(self._settings().get("ram_show", False))

    def set_ram_show(self, on: bool) -> None:
        self._settings()["ram_show"] = bool(on)
        self._save()

    # ---- brain mode (auto vs devin) -------------------------------------

    def get_brain(self) -> str:
        """Return current brain mode: ``auto`` (LLM via OpenRouter) or ``devin``.

        ``devin`` mode means the bot does NOT auto-reply; incoming messages
        are logged to ``data/inbox.log`` and a real Devin session (with shell
        access to the same VM) responds via ``python -m bot.send``.
        """
        return str(self._settings().get("brain", "auto"))

    def set_brain(self, mode: str) -> None:
        if mode not in ("auto", "devin"):
            raise ValueError(f"unknown brain mode '{mode}', expected 'auto' or 'devin'")
        self._settings()["brain"] = mode
        self._save()

    def set_enabled(self, enabled: bool) -> None:
        self._settings()["enabled"] = bool(enabled)
        self._save()

    # ---- owner (single-owner claim) -------------------------------------

    def get_owner_id(self) -> int | None:
        """Telegram user-id that owns this container, or ``None`` if unclaimed."""
        raw = self._settings().get("owner_id")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def set_owner_id(self, user_id: int) -> None:
        """Lock the container to a single Telegram user. Idempotent."""
        self._settings()["owner_id"] = int(user_id)
        self._save()

    # ---- access mode + co-owners ---------------------------------------
    #
    # Three modes control who can do what:
    #
    # * ``private`` (default) — only the owner and explicit co-owners can
    #   use the bot at all and change settings. Other users are silently
    #   ignored. Behaviour matches the original ``_is_owner`` check.
    # * ``public`` — anyone can USE the bot (chat, photo, helpzavr,
    #   pretty-text, media-generation). API keys / models / statistics
    #   stay owner-only. Co-owners count as admins.
    # * ``full_public`` — anyone can use AND change settings. No gating
    #   beyond Telegram chat membership.
    #
    # ``co_owners`` is a list of Telegram user-ids that get the same
    # rights as the owner regardless of access mode — they're the "give
    # access to" pool from the «Доступ» settings screen.

    _ACCESS_MODES = ("private", "public", "full_public")

    def get_access_mode(self) -> str:
        raw = str(self._settings().get("access_mode", "private"))
        if raw not in self._ACCESS_MODES:
            return "private"
        return raw

    def set_access_mode(self, mode: str) -> None:
        if mode not in self._ACCESS_MODES:
            raise ValueError(
                f"unknown access mode '{mode}', "
                f"expected one of {self._ACCESS_MODES}"
            )
        self._settings()["access_mode"] = mode
        self._save()

    def get_co_owners(self) -> list[int]:
        raw = self._settings().get("co_owners", [])
        out: list[int] = []
        if isinstance(raw, list):
            for item in raw:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
        return out

    def add_co_owner(self, user_id: int) -> bool:
        """Append ``user_id`` to the co-owner list. Returns False if it
        was already there (idempotent), True if added."""
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return False
        existing = set(self.get_co_owners())
        if uid in existing:
            return False
        owner = self.get_owner_id()
        if owner is not None and uid == owner:
            return False
        owners = self._settings().setdefault("co_owners", [])
        if not isinstance(owners, list):
            owners = []
        owners.append(uid)
        self._settings()["co_owners"] = owners
        self._save()
        return True

    def remove_co_owner(self, user_id: int) -> bool:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return False
        owners = self.get_co_owners()
        if uid not in owners:
            return False
        owners.remove(uid)
        self._settings()["co_owners"] = owners
        self._save()
        return True

    def is_owner_or_co_owner(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return False
        if self.get_owner_id() == uid:
            return True
        return uid in self.get_co_owners()

    def can_use_bot(self, user_id: int | None) -> bool:
        """True when ``user_id`` is allowed to USE bot features (chat,
        photo flows, helpzavr / pretty-text / mailbox / media)."""
        if user_id is None:
            return False
        if self.is_owner_or_co_owner(user_id):
            return True
        return self.get_access_mode() in ("public", "full_public")

    def can_admin_bot(self, user_id: int | None) -> bool:
        """True when ``user_id`` is allowed to CHANGE settings — API
        keys, models, mailbox creds, access mode, co-owner list, etc."""
        if user_id is None:
            return False
        if self.is_owner_or_co_owner(user_id):
            return True
        return self.get_access_mode() == "full_public"

    # ---- GitHub project metadata + auto-clone toggle -------------------
    #
    # The agent has tools to clone arbitrary git URLs (``clone_repo``)
    # and switch between cloned projects (``switch_project``).  Two
    # things live here:
    #
    # 1. ``github_auto_clone`` (bool, default True): when True, the
    #    agent's system prompt tells it to call ``clone_repo``
    #    automatically when the user shares a GitHub URL. When False,
    #    the prompt tells it to wait for an explicit "склонируй"
    #    instruction first.
    #
    # 2. ``github_repos`` (dict[name, dict]): per-repo metadata we
    #    extract on clone — URL, short README-derived description,
    #    timestamp, optional usage example. The wizard's «🐙 GitHub»
    #    screen reads from this so the user can browse / delete repos
    #    without poking at the filesystem.

    def get_github_auto_clone(self) -> bool:
        """True iff the agent should auto-clone GitHub URLs from chat.

        Defaults to True so the feature is on out of the box (the user
        asked for "по умолчанию работает без напоминания"). Stored
        under ``_settings.github_auto_clone``.
        """
        return bool(self._settings().get("github_auto_clone", True))

    def set_github_auto_clone(self, enabled: bool) -> None:
        self._settings()["github_auto_clone"] = bool(enabled)
        self._save()

    def _github_repos(self) -> dict:
        return self._settings().setdefault("github_repos", {})

    def list_github_repos(self) -> list[dict]:
        """Return a list of repo metadata dicts, sorted by name.

        Each entry has at least: ``name``, ``url``, ``description``,
        ``cloned_at``. ``example`` may be empty. Filtered to entries
        whose on-disk folder still exists — stale metadata for
        manually-deleted folders is omitted.
        """
        repos = self._github_repos()
        out: list[dict] = []
        for name, meta in repos.items():
            if not isinstance(meta, dict):
                continue
            if not (PROJECTS_DIR / name).exists():
                continue
            out.append(
                {
                    "name": name,
                    "url": str(meta.get("url", "")),
                    "description": str(meta.get("description", "")),
                    "example": str(meta.get("example", "")),
                    "cloned_at": int(meta.get("cloned_at", 0) or 0),
                }
            )
        out.sort(key=lambda x: x["name"])
        return out

    def get_github_repo(self, name: str) -> dict | None:
        """Return metadata for a single repo or None if unknown."""
        meta = self._github_repos().get(name)
        if not isinstance(meta, dict):
            return None
        if not (PROJECTS_DIR / name).exists():
            return None
        return {
            "name": name,
            "url": str(meta.get("url", "")),
            "description": str(meta.get("description", "")),
            "example": str(meta.get("example", "")),
            "cloned_at": int(meta.get("cloned_at", 0) or 0),
        }

    def set_github_repo_meta(
        self,
        name: str,
        url: str,
        description: str = "",
        example: str = "",
    ) -> None:
        """Record a freshly-cloned repo's metadata.

        Called by ``tools.clone_repo`` right after a successful
        ``git clone`` so the wizard's GitHub screen can list the repo
        with a human-readable description.
        """
        import time

        self._github_repos()[name] = {
            "url": url,
            "description": description,
            "example": example,
            "cloned_at": int(time.time()),
        }
        self._save()

    def forget_github_repo(self, name: str) -> bool:
        """Forget a repo's metadata. Returns True if it was known."""
        repos = self._github_repos()
        if name in repos:
            del repos[name]
            self._save()
            return True
        return False

    # ---- LLM provider selection (auto-mode backend) ---------------------

    def get_provider(self) -> str:
        """Which LLM provider auto-mode talks to.

        Defaults to ``openrouter`` for backwards compatibility. Other valid
        values are ``custom`` (a self-hosted OpenAI-compatible endpoint) or
        ``devin`` (handled separately by brain=devin and ignored here).
        """
        return str(self._settings().get("provider", "openrouter"))

    def set_provider(self, provider: str) -> None:
        if provider not in ("openrouter", "custom"):
            raise ValueError(
                f"unknown provider '{provider}', expected 'openrouter' or 'custom'"
            )
        self._settings()["provider"] = provider
        self._save()

    def get_base_url(self) -> str:
        """Base URL for the active provider. Empty = use provider's default."""
        return str(self._settings().get("base_url", ""))

    def set_base_url(self, url: str) -> None:
        self._settings()["base_url"] = url.rstrip("/")
        self._save()

    # ---- persona override (per-bot role change from TG) ------------------

    def get_persona_override(self) -> str | None:
        """Read TG-set persona override. None = fall back to BOT_PERSONA env var.

        The override is set via the `/role` command and persists across
        restarts in state.json. Acts as a soft re-assignment without
        touching Render env vars.
        """
        raw = self._settings().get("persona_override")
        if raw and isinstance(raw, str):
            return raw.strip().lower() or None
        return None

    def set_persona_override(self, persona_key: str) -> None:
        """Lock this bot into a new persona until cleared."""
        self._settings()["persona_override"] = persona_key.strip().lower()
        self._save()

    def clear_persona_override(self) -> None:
        """Remove the override → fall back to BOT_PERSONA env var."""
        if "persona_override" in self._settings():
            del self._settings()["persona_override"]
            self._save()

    # ---- heartbeat OpenRouter (idle ping toggle) -------------------------

    def get_heartbeat_enabled(self) -> bool:
        """True if OpenRouter heartbeat pings are enabled.

        When enabled, a background task pings the OpenRouter API
        periodically with a tiny prompt to keep the connection warm
        (and burn a fraction of free-tier quota). Independent of which
        provider is currently the active brain.
        """
        return bool(self._settings().get("heartbeat_enabled", False))

    def set_heartbeat_enabled(self, enabled: bool) -> None:
        self._settings()["heartbeat_enabled"] = bool(enabled)
        self._save()

    # ---- token usage tracking --------------------------------------------

    def log_llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str = "chat",
    ) -> None:
        """Record one LLM call's token usage.

        Stored as a flat list of dicts under ``_settings.token_log``. We
        cap the log at 5000 entries so state.json stays under a megabyte
        on long-running bots; older entries roll off FIFO.
        """
        import time

        log = self._settings().setdefault("token_log", [])
        log.append(
            {
                "ts": int(time.time()),
                "provider": provider,
                "model": model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "purpose": purpose,
            }
        )
        if len(log) > 5000:
            del log[: len(log) - 5000]
        self._save()

    def get_token_log(self, since_ts: int = 0) -> list[dict]:
        """Return token log entries with ``ts >= since_ts``."""
        log = self._settings().get("token_log", [])
        if since_ts <= 0:
            return list(log)
        return [e for e in log if e.get("ts", 0) >= since_ts]


storage = Storage()
