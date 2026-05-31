# Merged bot — что и откуда

Один Telegram-бот (один токен, один event loop, один Dockerfile, один Render-сервис), объединяющий **три** ранее независимых кодовых базы:

| Источник | Что взято | Куда положено |
|---|---|---|
| **lilush** | Полностью весь код, кроме главного меню. Это основа. | `bot/` |
| **tg_bot+(13)** | Скриншот-помощник (vision + thinking + OCR + рамка/стрелка). Чистая логика без дублей роутера. | `bot/addons/helpzavr/` |
| **freelance-mailbox-mcp** | IMAP-опрашиватель + парсер YouDo + Markdown→TG-HTML форматтер | `bot/addons/mailbox/` + `bot/addons/pretty_text/` |

## Что добавлено в lilush (минимальные правки)

1. **`bot/wizard.py` → `_kb_main_after_claim()`** — добавлены **3 кнопки** в верх главного меню:
   - 🤖 Helpzavr → `callback="main:helpzavr"`
   - ✨ Красивый текст → `callback="main:pretty"`
   - 📬 Проверка почты → `callback="main:mailbox"`
   - Старые 4 кнопки остались ниже без изменений.
2. **`bot/wizard.py` → `cb_main_menu()`** — добавлены 3 обработчика, которые показывают экраны соответствующих аддонов. Если аддон не загрузился (например, нет системного пакета) — корректное сообщение в чат.
3. **`bot/main.py`** — добавлен `dp.include_router(addon_router)` для каждого аддона и фоновая задача `mailbox_poller_loop`.
4. **`bot/handlers.py`** — добавлен гард `_role_chosen()` для `/projects`, `/project`, `/clone`, `/cd`, `/pwd`, `/exec`, `/git`. И для `/dl` — текст-хвост «Привязать к проекту» теперь показывается только когда роль выбрана.
5. **`Dockerfile`** — добавлены apt-пакеты `tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng libgl1 libglib2.0-0` (нужны Helpzavr).
6. **`requirements.txt`** — добавлены `pillow opencv-python-headless numpy pytesseract imap-tools`.
7. **`render.yaml`** — заменён на одно-сервисный конфиг (старый 20-bot вариант сохранён как `render.farm.yaml`).

## Что НЕ менялось

- Алгоритм `_make_status_updater` (это уже была «думалка-stream» в lilush) — нетронут.
- Логика `agent.py`, `media_ui.py`, `workers/`, `jobs.py`, `persona.py`, `storage.py` — нетронуты.
- 20 персонажей-роботов (`bot/personas/`) — все на месте.
- Видео-конвейер (yt-dlp → whisper → scenedetect → SEO) — нетронут.

## Как «думалка» теперь работает (стиль tg_bot+(13) везде)

**Helpzavr** (`bot/addons/helpzavr/handlers.py → _run_pipeline()`):
```
Смотрю на картинку…
   → Анализирую расположение полей… (Groq)
   → Думаю, что туда написать…
   → Уточняю границы поля…
   → Рисую…
   → [готово, статус удаляется]
```

**LLM-чат в lilush** (`bot/agent.py`) — раньше было «🔄 Думаю... (model) → 🔄 Шаг 1: вызываю LLM → 🔧 Шаг 1: использую read_file». Сейчас переписано в стиль tg_bot+(13) — без эмодзи-префиксов, без «Шаг N», с человеческим описанием tool-вызовов:
```
Думаю…
   → Размышляю над ответом…
   → Смотрю что в проекте…   (вместо «использую list_dir»)
   → Читаю файл…             (вместо «использую read_file»)
   → Запускаю команду…       (вместо «использую exec_bash»)
   → Думаю дальше…
   → Готово
```
Маппинг tool → фраза лежит в `agent.py:_TOOL_VERBS` и расширяется одной строкой при добавлении новых tool'ов.

**Pretty Text → стиль «📰 Стандарт»** (`bot/addons/pretty_text/handlers.py`):
```
Думаю над оформлением…
   → Перевожу в формат TG-поста…
   → Применяю разметку…
   → [готово, статус удаляется]
```

## Pretty Text — два стиля оформления

Внутри 📬 ✨ Красивый текст теперь есть кнопка **«🎨 Поменять стиль»**. На выбор:

| Стиль | Как работает |
|---|---|
| **✂️ Без LLM** *(дефолт)* | Старое поведение — чисто механический Markdown → TG-HTML. Мгновенно, бесплатно, сохраняет ровно тот текст что ты прислал. |
| **📰 Стандарт** | LLM (через OpenRouter, тот же ключ что у lilush-чата) переписывает текст в формат TG-поста: жирный заголовок-вопрос → курсивный интро-абзац с эмодзи → буллет-список через `-` → подытоживающий абзац с эмодзи → жирный CTA в конце. Соответствует примеру который ты прислал (Telegram-канал автора-инфлюенсера). |

Выбранный стиль запоминается per-chat и переживает рестарт.

## Условие «не показывать проектные кнопки до выбора роли»

Реализовано через хелпер `_role_chosen()` в `bot/handlers.py`:
```python
def _role_chosen() -> bool:
    return storage.get_persona_override() is not None
```
Команды `/projects /project /clone /cd /pwd /exec /git` теперь отдают:
> «Сначала выбери роль через /role или кнопку «🎭 Сменить роль» в /start — пока роль не выбрана, проектная часть бота молчит.»

Команда `/dl` работает всегда (она нужна для базового скачивания видео), но хвост с «Привязать к проекту» добавляется только если роль выбрана.

## API-ключи

| Ключ | Где взять | Куда вписать |
|---|---|---|
| `BOT_TOKEN` | https://t.me/BotFather → /newbot | Render env vars |
| `GROQ_API_KEY` | https://console.groq.com → API Keys → Create | Render env vars **или** TG-визард внутри 🤖 Helpzavr |
| `OPENROUTER_API_KEY` | https://openrouter.ai → Keys → Create | Render env vars **или** TG-визард |
| `MAILBOX_EMAIL` + `MAILBOX_PASSWORD` | Yandex/Mail.ru/Gmail → пароли приложений | Render env vars **или** TG-визард в 📬 Проверка почты → ⚙️ Настройки почты |
| `ALLOWED_USER_IDS` | Получи через @userinfobot | Render env vars (опционально — иначе первый `/start` становится владельцем) |

## Что в `state.json`

Lilush уже хранил настройки в `$DATA_DIR/state.json`. Аддоны теперь пишут туда же — под ключом `_settings.addons.<name>`:
```json
{
  "_settings": {
    "addons": {
      "pretty_text": { "by_chat": { "353508050": { "enabled": true } } },
      "helpzavr": {
        "groq_api_key": "gsk_...",
        "openrouter_api_key": "sk-or-...",
        "by_chat": { "353508050": { "enabled": true } }
      },
      "mailbox": {
        "email": "alex@yandex.ru",
        "password": "abcdefghijklmnop",
        "notify_all_emails": true,
        "notify_chat_id": 353508050
      }
    }
  }
}
```
Один файл, один том на Render, ничего не теряется при рестарте.

## Запуск локально

```bash
pip install -r requirements.txt
export BOT_TOKEN=...   # от BotFather
export DATA_DIR=./data
python -m bot.main
```

## Деплой на Render

1. `git push` → GitHub-репозиторий.
2. https://dashboard.render.com → New → Blueprint → выбери репо.
3. Render прочитает `render.yaml` и спросит секреты — вставь `BOT_TOKEN` (остальные опциональны, их можно ввести через TG-визард после старта).
4. Apply. Render собирает Docker-образ (~3-5 минут) и запускает.
5. В Telegram: `/start` → кнопка «🚀 Запустить и стать владельцем».
6. `/role` → выбери роль (например, **Boss** или любую другую).
7. На главном меню теперь 7 кнопок. Тыкай и пользуйся.

Если хочешь полную ферму из 20 ботов — переименуй `render.farm.yaml` в `render.yaml`.

## Раунд v3 (Editor Agent + UI правки)

Изменения по запросу пользователя из чата:

1. **«🎭 Сменить роль» убрана из главного меню** — дублировала «⚙️ Настройки → 🎭 Роли». Команда `/role` и кнопка в настройках остались.
2. **«🧠 Перенастроить мозг» — добавлена кнопка «← Назад»** (callback `brain:back` → главное меню). Раньше пользователь застревал на экране без выхода.
3. **Алгоритмы → «🤖 Определить порядок с помощью ИИ»** — улучшен UX:
   - Pre-flight проверка наличия настроенного мозга (раньше ИИ молча проглатывал сообщение и потом отвечал ошибкой)
   - Понятнее сообщение с примерами приказов и описанием кнопок
   - Если ИИ ответил, но не распарсился — показываем сырой ответ и предлагаем переформулировать
4. **Новая кнопка «🎞 Монтажёр»** в главном меню (admin-only) — открывает экран Editor Agent (`bot/addons/editor_agent/`).
   - Версия пайплайна (v1 / v2 / v2.1 / v6), профиль качества (light / medium / heavy), v6-планировщик ON/OFF, ползунок уникальности.
   - Изменения хранятся в `_settings.addons.editor_agent.*`. Без юзерских правок — env-defaults сохраняются.
   - `EditorV2Worker` подключается через `build_editor_worker()` в `bot/main.py` — выбор пайплайна по `EDITOR_VERSION`.
   - `bot/uniqueization/` — поддержка планировщика v6 (cuts / mirror / blur-fill / hook-emphasis / unique-distance).
   - Команда `/editor` открывает тот же экран.

Скопированные тесты из editor-архива: `test_editor_v2_unit.py`, `test_editor_agent_addon.py`, `test_uniq_foundations.py`, `test_uniqueness_randomization.py`. Все 455 тестов проходят.
