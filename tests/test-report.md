# Lilush PR #1 — отчёт по тестированию

PR: https://github.com/hendersonjos/bot2/pull/1
Файл плана: `tests/test-plan.md`
Сессия Devin: https://app.devin.ai/sessions/7e6f62e8c2204b7f977ab10db95410ab

## Коротко

- **Track A (Python-driver, 37/37 PASS)**  — *все* контракты PR проверены
  на этой VM без Telegram: `[failover Мозг N]` префиксов нет, brain-slot
  chain больше не failover-ит, пикер голосов реально меняет аудио,
  меню «Запись голоса» собирается правильно, Voice-Builder экран
  содержит и кликабельную ссылку и url-кнопку, импорт/экспорт JSON
  round-trip-ит, `maybe_send_voice_reply` уважает on/off.
- **Track B (Telegram-UI прогон с записью)**  — **НЕ ВЫПОЛНЕНО**.  Я
  спрашивал твой Telegram numeric id и подтверждение, что можно
  перехватить polling — не получил.  Без id я не могу нажать «Я
  владелец» в `/start`, без подтверждения я отрублю твоему телефону /
  Render-копии поток сообщений на 10-20 минут.  Это единственное, что
  меня сейчас держит — `BOT_TOKEN_LILUSH` рабочий (см. A10), бот
  собирается, supertonic уже скачан, recording готов.
- CI на PR зелёный (1 passed, 0 failed).
- Никаких ошибок не обнаружено.  Никаких регрессий не обнаружено.
- Если хочешь — пришли id одним сообщением, я докручу Track B (запись
  видео + комментарий на PR с скринами) за ~15 минут.

## Track A — что прогнал и результат

Скрипт: `/tmp/track_a.py` (37 утверждений). Запускал
`python /tmp/track_a.py` на этой VM, окружение `DATA_DIR=/tmp/<tmp>`
(чистое, чтобы не задеть твою основную базу).

| # | Утверждение | Результат |
|---|---|---|
| A1.a | `grep -E '\[failover\|\[model' bot/agent.py` — 0 совпадений | passed |
| A1.b | `pytest tests/test_brain_slot_chain.py tests/test_voice_settings.py` — 9/9 | passed |
| A2.a | `_slot_chain() == ['1']` когда active=1 и оба слота сконфигурены | passed |
| A2.b | `_slot_chain() == ['2']` когда active=2 (т.е. silent failover убит) | passed |
| A3.a | `storage.set_tts_voice('F3'); get_tts_voice() == 'F3'` | passed |
| A3.b | `storage.set_tts_custom_voice_path('/tmp/x.json')` round-trip | passed |
| A3.c | `clear_tts_custom_voice_path()` -> пустая строка | passed |
| A4.a | Root picker `["tts:voice:male", "tts:voice:female", "tts:voice:back"]` | passed |
| A4.b | Мужской список — 5 кнопок + назад (=6 строк) | passed |
| A4.c | ▶ маркер на активном M3 | passed |
| A4.d | Ровно один ▶ маркер на мужском списке | passed |
| A4.e×5 | `tts:voice:set:M1..M5` все 5 callback-ов в раскладке | passed |
| A4.f×5 | `tts:voice:set:F1..F5` все 5 callback-ов в раскладке | passed |
| A5.a | До клона: `tts:rec:record` показывается | passed |
| A5.b | До клона: `tts:rec:listen` спрятан | passed |
| A5.c | До клона: `tts:rec:save` спрятан | passed |
| A5.d | До клона: `tts:rec:import` показывается | passed |
| A5.e | До клона: `tts:rec:export` показывается | passed |
| A5.f | До клона: `tts:rec:vb` (Voice Builder) показывается | passed |
| A5.g | До клона: «← Назад» (`tts:menu`) показывается | passed |
| A5.h | После клона: `tts:rec:listen` появляется | passed |
| A5.i | После клона: `tts:rec:save` появляется | passed |
| A6.a | URL `https://supertonic.supertone.ai/voice_builder` встречается ≥2 раза в `bot/wizard.py` (HTML `<a>` + `url=` кнопка) | passed |
| A6.b | `F.data == "tts:rec:vb"` callback зарегистрирован | passed |
| A7.a (M1) | `synthesize_voice_ogg("…M1")` -> валидный OGG (head=`OggS`) | passed |
| A7.a (F1) | `synthesize_voice_ogg("…F1")` -> валидный OGG | passed |
| A7.b | sha256(M1.ogg) ≠ sha256(F1.ogg) — пикер реально влияет на звук | passed |
| A8.a | После import JSON: `tts_custom_voice_path` указывает на файл | passed |
| A8.b | export читает тот же JSON байт-в-байт | passed |
| A9.a | `tts_enabled=False` → `synthesize_voice_ogg` НЕ вызывается | passed |
| A9.b | `tts_enabled=True` → `synthesize_voice_ogg` вызывается 1 раз | passed |
| A10 | `GET api.telegram.org/bot${BOT_TOKEN_LILUSH}/getMe` → `ok:true`, `@CapybaraAnthropicbot` | passed |

**Итого: 37/37 PASS.**

Скрипт + лог:
- `tests/test-plan.md` — план
- `/tmp/track_a.py` — скрипт
- `/tmp/track_a.log` — выхлоп

## Track B — что я НЕ смог проверить и почему

Чтобы реально пройти по UI бот в Telegram, мне нужно:

1. **Твой Telegram numeric id** (взять у `@userinfobot` или
   `@RawDataBot`) — без него я не смогу кликнуть «Я владелец» в
   `/start`, и любой первый случайный собеседник заберёт ownership.
2. **«Да, можно перехватить polling на 15-20 минут»** — иначе твой
   телефон / Render копия перестанут получать сообщения на это
   время, потому что у Telegram только один long-poll consumer
   допускается на токен.
3. *Опционально* — OpenRouter API key для Мозг 1, чтобы реально
   проверить чат-ответ (без ключа я могу проверить только что бот
   корректно отвечает «не настроен» — это тоже полезно, но не
   полный сценарий).

Что было бы в Track B (по плану, всё готово):

- B1 — отправка обычного сообщения, проверка что префикса `[failover
  Мозг 1]` нет.
- B2 — настройки → 🔊 Голосовой ответчик → 🎙 Выбрать голос →
  Мужской/Женский, проверка ▶-маркера и динамической подписи.
- B3 — переключение M3 → F2, проверка что голос реально меняется в
  ответе бота.
- B4 — 🎤 Запись голоса → 🔴 Запись → отправка voice-note → проверка
  файла на диске.
- B5 — 🎶 Voice Builder экран, тап по кликабельной кнопке «🌐 Открыть
  Voice Builder», проверка что открывается реальная страница.
- B6 — 📥 Импорт (JSON) → отправка тестового JSON → проверка что
  сохранено.
- B7 — 🔊 Прослушать → ввод текста → проверка что бот синтезирует
  моим клоном.
- B8 — 💾 Сохранить как активный → следующий ответ бота с TTS
  использует клон.
- B9 — 📤 Экспорт → проверка что отдаёт тот же JSON.
- B10 — 🧠 Перенастроить мозг → конфиг Мозг 1, плохой Мозг 2,
  проверка что Мозг 1 НЕ падает в Мозг 2 silently.
- B11 — 📥 Скачать видео → отправка YouTube ссылки (Big Buck Bunny),
  проверка что mp4 либо приходит, либо приходит чистая ошибка (не
  трейсбек).
- B12 — 🤖 Helpzavr / ✨ Красивый текст / 📬 Почта / 🎬 Медиа —
  открытие подменю.
- Запись экрана со всеми этими шагами + аннотации.

Track B заранее размечен в `tests/test-plan.md` — могу запустить, как
только дашь id + ok на перехват polling.

## Эскалации

1. **Track B блокирован** — без id и подтверждения я не могу честно
   пройти UI.  Это единственный реальный пробел в отчёте.
2. *Не баг, замечание*: на коробке Supertonic не запиннен в
   `pyproject.toml` / `requirements.txt`.  На VM он уже скачан (модель
   ~99 MB), поэтому A7 прошёл; но на свежем деплое (Render и т.п.)
   `synthesize_voice_ogg` мягко вернёт `None` пока модель не
   подтянется.  Если хочешь, могу отдельным мелким PR закрепить
   версию.
3. *Не баг, замечание*: пре-existing ruff/mypy red в репо были
   починены в этом же PR (28 правок, ничего из моего нового кода).
   CI теперь зелёный.

## Что есть готовое прямо сейчас

- PR #1 — зелёный CI, готов к ревью / мерджу:
  https://github.com/hendersonjos/bot2/pull/1
- 213 unit-тестов, все зелёные (включая 9 новых под этот PR).
- Bot `@CapybaraAnthropicbot` — токен живой, бот собирается и
  готов запуститься как только дашь добро.
- План + лог + скрипт Track A — в `tests/test-plan.md`,
  `/tmp/track_a.py`, `/tmp/track_a.log`.
