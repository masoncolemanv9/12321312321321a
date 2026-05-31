# Lilush bot — Phase-1/voice-picker test plan (PR #1)

Tests the changes shipped in PR #1
(https://github.com/hendersonjos/bot2/pull/1):

1. removal of `[failover Мозг N]` / `[model X]` prefixes,
2. brain slot chain no longer silently failovers (Brain 2 reserved for
   transcription),
3. voice picker UI (5 male M1–M5 + 5 female F1–F5),
4. record-voice submenu (запись / прослушать / сохранить / импорт /
   экспорт / Voice Builder).

## Goals & non-goals

Goal: prove every wizard button and every voice/brain code-path the user
asked about actually does what its label says and that the audio
synthesised on this VM changes when the user changes the voice.

Non-goals:

- No code review / unit-test summarising — only **runtime** behaviour.
- No fix work — if anything fails, exit testing mode and patch.

## Execution tracks

The user asked me to spin up a server, paste the TG token, and walk
every button.  Two issues block "real Telegram UI" testing without
their input:

- they haven't yet confirmed I can take over polling on this VM (their
  phone / Render copy would stop receiving messages while I poll), and
- they haven't sent their **Telegram numeric id** (without it I can't
  claim ownership of the `/start` session).

So the plan runs two tracks in parallel:

| Track | Needs from user | What it proves |
|-------|-----------------|----------------|
| **A — Python driver** (always-on, headless) | nothing | every wizard callback and storage write, plus real OGG synthesis with each voice. |
| **B — Telegram UI walkthrough** (recorded) | TG id + go-ahead | the user-visible UX matches the spec, the picker actually changes the audio in chat, ✨/📬/🎬/📥 buttons behave. |

If Track B is still blocked at the end of the run, the affected
assertions are reported as **untested** with a clear note — they are
not silently skipped.

## Track A — Python-driver tests

Run from `/home/ubuntu/repos/bot2` with `pytest` + ad-hoc Python.  All
assertions below either lock in existing test files or add a new
ad-hoc check (run from a `python -m bot ...` shell).

### A1 — `[failover Мозг N]` / `[model X]` prefixes are gone

- `grep -nE '\[failover|\[model'  bot/agent.py`
  Expect: **no** match (only docstring / changelog mentions allowed).
- Read `bot/agent.py` around the answer-emit path, look for any
  `f"[{...}]\n{answer}"` style format string left over.  Expect:
  none.
- Run `pytest tests/test_brain_slot_chain.py -q`. Expect 2/2 pass.

### A2 — `_slot_chain()` returns only the active slot

```py
from bot import storage
from bot.agent import _slot_chain

# Configure both slots so we know a fallback is _possible_.
storage.set_brain_cfg("1", {"provider": "openrouter", "api_key": "k1", "model": "m1"})
storage.set_brain_cfg("2", {"provider": "openrouter", "api_key": "k2", "model": "m2"})

storage.set_active_brain_slot("1")
assert _slot_chain() == ["1"], _slot_chain()
storage.set_active_brain_slot("2")
assert _slot_chain() == ["2"], _slot_chain()
```

Expect both asserts to pass (already covered by
`tests/test_brain_slot_chain.py`, this just re-runs the assertion
shape end-to-end).

### A3 — Storage round-trip for voice settings

```py
from bot import storage

storage.set_tts_voice("F3")
assert storage.get_tts_voice() == "F3"

storage.set_tts_custom_voice_path("/tmp/x.json")
assert storage.get_tts_custom_voice_path() == "/tmp/x.json"

storage.clear_tts_custom_voice_path()
assert storage.get_tts_custom_voice_path() == ""
```

(Already locked in by `tests/test_voice_settings.py`; this run just
re-runs `pytest tests/test_voice_settings.py -q` and expects 7/7.)

### A4 — Voice picker keyboard builds the right buttons

Build the keyboard in-process and assert every callback string + label:

```py
from bot import storage
from bot.wizard import _kb_voice_picker_root, _kb_voice_picker_list
from bot.tts import BUILTIN_MALE_VOICES, BUILTIN_FEMALE_VOICES

assert [b.callback_data for row in _kb_voice_picker_root().inline_keyboard for b in row] == [
    "tts:voice:male", "tts:voice:female", "tts:voice:back",
]

storage.set_tts_voice("M3")
storage.clear_tts_custom_voice_path()
male_rows = _kb_voice_picker_list("male").inline_keyboard
male_labels = [b.text for row in male_rows for b in row]
male_cbs    = [b.callback_data for row in male_rows for b in row]

# 5 male voices + back button = 6 rows.
assert len(male_rows) == 6, male_rows
# Active M3 has a ▶ marker; the rest do not.
assert "▶ M3" in male_labels
assert sum(1 for l in male_labels if l.startswith("▶")) == 1
# Every M-voice has a tts:voice:set:M<N> callback.
for v in BUILTIN_MALE_VOICES:
    assert f"tts:voice:set:{v}" in male_cbs, v

female_rows = _kb_voice_picker_list("female").inline_keyboard
female_cbs  = [b.callback_data for row in female_rows for b in row]
for v in BUILTIN_FEMALE_VOICES:
    assert f"tts:voice:set:{v}" in female_cbs, v
```

Expect: all asserts pass.

### A5 — Record menu keyboard builds the right buttons

```py
from bot import storage
from bot.wizard import _kb_record_menu

storage.clear_tts_custom_voice_path()
rows = _kb_record_menu().inline_keyboard
cbs  = [b.callback_data for row in rows for b in row]
# Запись is always shown; Прослушать/Сохранить only after a clone exists.
assert "tts:rec:record"  in cbs
assert "tts:rec:listen"  not in cbs
assert "tts:rec:save"    not in cbs
assert "tts:rec:import"  in cbs
assert "tts:rec:export"  in cbs
assert "tts:rec:vb"      in cbs
assert "tts:menu"        in cbs  # ← Назад

# Once a clone is configured, Прослушать / Сохранить are surfaced.
storage.set_tts_custom_voice_path("/tmp/fake.json")
rows = _kb_record_menu().inline_keyboard
cbs  = [b.callback_data for row in rows for b in row]
assert "tts:rec:listen" in cbs
assert "tts:rec:save"   in cbs
storage.clear_tts_custom_voice_path()
```

Expect: all asserts pass.

### A6 — Voice Builder info screen is correct

The user said the screen must contain a clickable link and step-by-step
instructions.  Inspect the callback handler text:

- `bot/wizard.py:1444-1470` (cb_tts_rec_vb) must contain:
  - the literal URL `https://supertonic.supertone.ai/voice_builder`,
  - an `InlineKeyboardButton(url=...)` row so the link is clickable in
    Telegram (not just an HTML `<a>`),
  - step-by-step instructions (numbered list).

`grep -n 'supertonic.supertone.ai/voice_builder' bot/wizard.py` —
expect ≥ 2 matches (HTML `<a>` + `url=` button).

### A7 — `synthesize_voice_ogg()` actually changes audio per voice

This is the load-bearing test for the picker.  Run as a one-shot
script (Supertonic model is already cached on this VM from the earlier
session):

```py
import asyncio, hashlib, os
from bot.tts import synthesize_voice_ogg, BUILTIN_VOICES, _selected_voice
from bot import storage

async def go():
    sigs = {}
    for v in ["M1", "F1"]:                # spot-check 1 male + 1 female
        storage.set_tts_voice(v)
        storage.clear_tts_custom_voice_path()
        ogg = await synthesize_voice_ogg("привет, тест голоса " + v)
        assert ogg and ogg[:4] == b"OggS", (v, ogg[:8] if ogg else None)
        sigs[v] = hashlib.sha256(ogg).hexdigest()
    assert sigs["M1"] != sigs["F1"], sigs
    print("M1 hash", sigs["M1"][:12], "F1 hash", sigs["F1"][:12])

asyncio.run(go())
```

Pass: both `OggS` magic bytes are present **and** the two hashes
differ (proves the voice argument is actually wired through).  If
Supertonic isn't installed in this venv, fall back to asserting
`bot.tts.get_unavailable_reason()` returns a clear message — the user
needs to know if their deployment is missing the wheel.

### A8 — Import → Save → Export round-trip (JSON path)

```py
import json, os
from bot import storage
from bot.config import config

# Pretend we ran the Импорт capture handler on this JSON.
sample = {"voice": "demo", "style": {"foo": 1}}
voice_dir = os.path.join(config.data_dir, "tts_voices")
os.makedirs(voice_dir, exist_ok=True)
path = os.path.join(voice_dir, "12345_voice.json")
with open(path, "w") as f:
    json.dump(sample, f)

storage.set_tts_custom_voice_path(path)
assert storage.get_tts_custom_voice_path() == path
assert json.load(open(path)) == sample          # exporter would re-send this
```

Pass: path is persisted and the file round-trips byte-for-byte (this
is what the Экспорт handler does — read the file and send the bytes
back).

### A9 — `maybe_send_voice_reply` honours the picker

Patch a fake `message.bot.send_voice` capture and assert that:

- when `tts_enabled` is False → nothing sent,
- when `tts_enabled` is True + `tts_voice="F2"` → `synthesize_voice_ogg`
  invoked with that voice name passed through.

```py
import asyncio
from unittest.mock import AsyncMock, patch
from bot.tts import maybe_send_voice_reply
from bot import storage

class FakeMsg:
    def __init__(self):
        self.bot = AsyncMock()
        self.chat = type("C", (), {"id": 1})()
        self.message_thread_id = None
    async def answer_voice(self, *a, **kw): pass

# OFF: no synthesis.
storage.set_tts_enabled(False)
with patch("bot.tts.synthesize_voice_ogg", AsyncMock(return_value=b"OggS....")) as p:
    asyncio.run(maybe_send_voice_reply(FakeMsg(), "hi"))
    assert p.call_count == 0

# ON + F2: called with voice=F2 (or custom_path if set, but cleared here).
storage.set_tts_enabled(True)
storage.set_tts_voice("F2")
storage.clear_tts_custom_voice_path()
with patch("bot.tts.synthesize_voice_ogg", AsyncMock(return_value=b"OggS....")) as p:
    asyncio.run(maybe_send_voice_reply(FakeMsg(), "hi"))
    assert p.call_count == 1, p.call_args
```

Pass: zero calls when off, exactly one when on.

### A10 — `BOT_TOKEN_LILUSH` is usable

`getMe` against the Telegram API with the token in `BOT_TOKEN_LILUSH`
to prove the secret resolves and the bot user exists:

```sh
curl -s "https://api.telegram.org/bot${BOT_TOKEN_LILUSH}/getMe" | jq .
```

Expect: `"ok": true` + a username.  This is the **only** Track A check
that needs the network.

## Track B — Telegram UI walkthrough (recorded)

**Blocked** until the user sends their Telegram numeric id and
confirms I can take over polling.  The plan is the same regardless:

### B0 — Setup

1. `export BOT_TOKEN=$BOT_TOKEN_LILUSH` (no fancy proxy, just the
   single-bot quick path).
2. `python -m bot.main &` in a foreground shell so I can see logs.
3. Maximise Chrome window + `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`.
4. `recording_start(hide_cursor=False)`.
5. Open `https://web.telegram.org` (already authed from `chrome-data/`).
6. Open the bot chat from the sidebar, send `/start`.
7. Bot prompts to claim → "Я владелец" — works only because we passed
   the user-provided numeric id into `OWNER_ID` env.

### B1 — Failover prefix gone (chat, 1 step)

- Annotate `test_start="It should not prefix replies with [failover] or [model] tags"`.
- Send a plain chat message ("привет, как дела?").
- Assertion: reply text does not start with `[`.
  (If Brain 1 isn't configured, the bot says "не настроен" — that's
  fine as long as the message itself is unprefixed.)

### B2 — Voice picker UI exists

- Tap **⚙️ Настройки → 🔊 Голосовой ответчик**.
- Confirm 4 buttons visible: Авто/Выкл / 🎙 Выбрать голос (сейчас: M1) /
  🎤 Запись голоса / ← Назад.
- Tap **🎙 Выбрать голос → 👨 Мужской**. See 5 buttons M1–M5 with ▶ on
  M1.  Tap M3.  Toast / refresh → ▶ moves to M3, label of the picker
  button on the parent screen now reads "сейчас: M3".
- Tap **👩 Женский**.  See 5 buttons F1–F5.  Tap F2.  Verify same.

### B3 — Picker actually changes audio in chat

- Make sure Авто on, voice = M3.
- Send chat message → bot replies with a voice note in a deeper
  voice.
- Switch to F2, send the same message.  Voice note clearly higher /
  different timbre.
- Annotation: assertion "Voice note timbre changes when picker is
  switched between M3 and F2".

### B4 — Запись submenu — 🔴 Запись flow

- Tap **🎤 Запись голоса → 🔴 Запись**.  Bot replies "🎤 Жду voice / audio…".
- Drag-drop or record a 5-sec voice note in Telegram Web.
- Bot replies: "✅ Запись сохранена. Файл: data/tts_voices/<id>_recording.ogg".
- `ls -la /tmp/bot-data/tts_voices/` (shell) → file exists.

### B5 — Voice Builder info screen

- Back to record menu.  Tap **🎶 Voice Builder**.
- Bot replies with:
  - clickable `supertonic.supertone.ai/voice_builder` link,
  - numbered step-by-step instructions,
  - inline button "🌐 Открыть Voice Builder" (url=...) which when
    clicked actually opens Voice Builder in a new browser tab.
- Annotation: "Voice Builder screen contains URL + clickable button
  and steps 1-6 are present."

### B6 — Импорт (JSON)

- Inside record menu, **📥 Импорт (JSON)**.  Bot says "📥 Жду JSON-файл…".
- Send a tiny `{"voice": "test", "style": []}` document via Telegram
  Web.
- Bot replies "✅ JSON принят, сохранил клон".
- `cat /tmp/bot-data/tts_voices/<id>_voice.json` (shell) → exact JSON
  back.

### B7 — Прослушать / Сохранить

- After import (so the buttons exist), tap **🔊 Прослушать**.  Bot
  asks for text.  Send "проверка моего голоса".  Bot replies with a
  voice note synthesised with the JSON.
- Tap **💾 Сохранить как активный** — bot confirms.  Picker button on
  the parent screen now reads "сейчас: мой клон".
- Send a normal chat message → reply is voiced with the cloned style
  (or, if Supertonic rejects the synthetic JSON, an explicit error
  message — not a silent fall-back).

### B8 — Экспорт

- Tap **📤 Экспорт (JSON)**.  Bot sends back the JSON document.
- Download in Telegram Web, `sha256sum` against the original → equal.

### B9 — Brain 1 / Brain 2 — no silent failover

- ⚙️ → 🧠 Перенастроить мозг.  Configure Brain 1 with the real key
  (or fake one if user didn't share).  Configure Brain 2 with a known
  bad key.
- Active = Brain 1.  Send a chat message.  Reply works (or fails with
  a clear Brain-1 error).
- Make Brain 1 fail (revoke or set a bad key).  Send a message.
  Expect a Brain-1 failure message — **not** a silent jump to Brain
  2.  Switch active slot to Brain 2 manually; now Brain 2 is used.

### B10 — Other top-level buttons (smoke)

For each of these I just tap the button and screenshot the resulting
screen.  No state changes, no real downloads.

- 🤖 Helpzavr → opens helpzavr submenu (or "загружаю модель…" toast).
- ✨ Красивый текст → opens pretty-text style picker.
- 📬 Проверка почты → opens mailbox prompt for IMAP creds (don't
  actually fill in).
- 🎬 Генерация фото и видео → opens media submenu.
- 📥 Скачать видео → bot asks for a URL.  Send a short, freely
  shareable YouTube URL (e.g. Big Buck Bunny clip).  Expect either a
  successful mp4 or a clearly worded "не получилось скачать" — never
  a stack trace.
- 🖥 Комп (localdesktop) → bot prints the install plan as `/work`
  commands.  Don't actually run them on this VM.
- ⬇️ Скачать (bash2mp4) → same.
- 🧪 Анализатор (agent-skills) → same.
- 🎭 Сменить роль → cycles through saved role definitions; back out.
- 📊 Статистика токенов → shows usage table (empty if no chat yet).

### B11 — Cleanup

- `recording_stop(title="Lilush PR #1 walk", summary=...)`.
- Kill the local bot.
- Restore `tts_voice` / `tts_custom_voice_path` to whatever they were
  before (or leave; the user can reset).

## Reporting

- One GitHub comment on PR #1.  `<details>` per track.  Track A
  collapsed by default (it's the noisy stuff); Track B expanded by
  default (the visual proof).
- Inline screenshots from the recording for B2 (picker), B5 (Voice
  Builder screen), B6 (Импорт), B8 (Экспорт).
- Recording attached to the final user message.

## Risks

- **Supertonic 99 MB model is not pinned** in `pyproject.toml`.  On a
  cold VM the first synth call downloads it (~3 s here).  If that
  download fails in CI, A7 falls back to checking
  `get_unavailable_reason()` instead.
- **Track B requires user action** (TG id + go-ahead).  If they don't
  reply, Track B becomes "untested" — Track A still proves the
  important contracts and the actual audio generation.
- **Brain testing without keys**: if the user doesn't share an
  OpenRouter key for Brain 1, B9 can only test the failure path, not
  the success path.  That's still useful — the user's bug report was
  about the *failure* path failing-over silently.
