# Lilush

**Self-hosted Telegram-бот с мульти-агентным видео-конвейером.**
Кидаешь ссылку → `yt-dlp` качает → анализатор находит лучшие моменты →
редактор режет в вертикальные шорты → SEO-агент пишет метаданные →
паблишер заливает на YT/TikTok/IG.

Деплоится на любое облако (Render, Railway, Fly.io, Heroku, VPS) **с
одним секретом — `BOT_TOKEN`**. Всё остальное (LLM-провайдер, API-ключ,
модель) настраивается **изнутри Telegram через кнопки**.

> ⚠️ Real-world hosting: тяжёлые стадии (ffmpeg / whisper) на free-tier
> Render/Railway/Fly **не запустятся** — нужен 4 vCPU / 8 GB RAM минимум.
> Подробности и реальные free-варианты в
> [`.devin/tentacles/lilush-pipeline/CONTEXT.md`](.devin/tentacles/lilush-pipeline/CONTEXT.md).

## 🚜 Farm: 20 ботов одним Blueprint

Помимо одиночного деплоя (см. ниже) репо разворачивается как
**ферма из 20 Render-сервисов** — Boss + 4 Department Lead'а
(Research / Debate / Coder / DevOps) + 15 worker'ов. Каждый сервис —
это та же кодовая база, активированная разной `BOT_PERSONA`.

```
Leadership (5):
  [lilush-boss]                /start /research /debate /implement /status
  [lilush-research-lead]       координирует research-отдел
  [lilush-debate-lead]         координирует debate-отдел
  [lilush-coder-lead]          координирует coder-отдел
  [lilush-devops-lead]         координирует DevOps-отдел

Research (4):  github-scout, reddit-scout, hn-scout, apify-runner
Debate (4):    d1-skeptic, d2-optimist, judge, tz-writer
Coder (3):     devin-spawner, pr-reviewer, tester
DevOps (4):    watchdog, render-admin, github-admin, archivist
```

**Деплой:**

1. Создай ботов в [@BotFather](https://t.me/BotFather) — можешь делать
   постепенно. Один уже есть — `@openaiopus_bot` для Boss. Для каждого
   нового бота: `/newbot` → имя → username → токен → Group Privacy: Off.
2. <https://dashboard.render.com> → **New +** → **Blueprint** → подключи
   этот репо (или дай Public Git URL без OAuth).
3. Render прочитает `render.yaml`, найдёт **20 сервисов** и для каждого
   спросит `BOT_TOKEN` + `ALLOWED_USER_IDS`. Для ботов которых ещё нет
   в BotFather — **оставь поле пустым**. Эти сервисы поднимутся в
   **dormant-режиме** (только `/healthz`, не падают, ничего не тратят).
4. **Apply Blueprint** — Render развернёт 20 сервисов параллельно.
5. Активные боты в Telegram: `/start` → **🚀 Запустить и стать
   владельцем** → выбери мозг → введи API-ключ. Полностью в TG.
6. Dormant-сервисы активируются позже: Render dashboard → нужный
   сервис → **Environment** → впиши `BOT_TOKEN` → Save → бот
   автоматом перезапустится в обычном режиме.

Каждый бот получает свой 1GB persistent disk на `/data` и
само-пингует свой `/healthz` каждые 4-7 минут (только активные —
dormant сервис не пингает себя, просто спит).

**Кастомизировать список ролей:** редактируй `bot/persona.py` и
регенерируй `render.yaml`:

```bash
python scripts/gen_render_yaml.py            # все 20 service'ов (default)
python scripts/gen_render_yaml.py --first 5  # только boss + 4 lead'а
```

### 🔑 Внешние API для research-ботов

Researcher-боты (`github_scout`, `reddit_scout`, `apify_runner`, ...)
ходят за данными в платные/полу-платные сервисы. Чтобы не сетапить
их через env vars в Render — добавь ключи прямо из Telegram:

`/setup` → **🛠 Внешние API** → выбери инструмент → пришли ключ
одним сообщением (бот удалит твоё сообщение после сохранения).

Что поддерживается из коробки:

| Инструмент | Зачем | Где взять |
|---|---|---|
| **Apify** | Готовые scrap-actors (Reddit, TikTok, YT, Twitter) | <https://console.apify.com/account/integrations> |
| **Firecrawl** | Сайт → чистый markdown для LLM | <https://firecrawl.dev/app/api-keys> |
| **Tavily** | Search-API для AI-агентов | <https://app.tavily.com/home> |
| **Brave Search** | Альтернатива Google без трекинга | <https://api-dashboard.search.brave.com/app/keys> |
| **Exa** | Семантический поиск | <https://dashboard.exa.ai/api-keys> |
| **GitHub PAT** | Поднимает GitHub API rate-limit 60 → 5000/час | <https://github.com/settings/tokens> |

Из кода бота: `storage.get_external_tool_key("apify")` (fallback на
env var `APIFY_API_TOKEN`).

### 🎬 Мозг для видео и фото (Media Brain)

Отдельный раздел `/setup → 🎬 Мозг для видео и фото` управляет
провайдерами генеративного медиа — поверх existing «Мозг» (LLM) и
«Внешние API» (research). Поддерживает до **3 слотов** (URL + API key
+ per-task model). Имя слота автоматически становится `deapi` / `fal`
/ `replicate` по URL.

Каждый слот можно сделать активным для трёх задач (можно
независимо — `deAPI` для видео, `fal.ai` для фото):

- 🎬 **video** — img2video (анимация фото в короткий ролик)
- 📷 **photo** — img2img (трансформация фото по промпту)
- 🪄 **rmbg** — background-removal (PNG с альфой для Photoshop)

Когда пользователь шлёт боту фото, появляются inline-кнопки
`📷 Фото` / `🎬 Видео` / `❌ Назад`:

- **Фото** → промпт сообщением или `🪄 Убрать фон`. После промпта
  выбор формата `JPEG` / `PNG (с альфой)`.
- **Видео** → выбор качества `480 / 720 / 1080 / 2К / Авто`.

Поддерживается **deAPI.ai** (полный набор: img2video LTX-Video,
img2img Qwen-Image-Edit / Flux-2-Klein, img-rmbg Ben2). $5 бесплатно
при регистрации на <https://deapi.ai/dashboard>. fal.ai / Replicate /
Runway оставлены как stub-адаптеры — добавь свой `submit_and_poll`
в `bot/media.py`.

Реализация: <code>bot/media.py</code> (клиент) + <code>bot/media_ui.py</code>
(wizard + photo flow).

Back-compat: env-var `DEAPI_API_KEY` всё ещё работает как fallback
если ни один слот не настроен — удобно для первого деплоя.

### 🦊 vercel-labs/agent-browser

`Dockerfile` уже ставит [vercel-labs/agent-browser](https://github.com/vercel-labs/agent-browser)
глобально (Rust CLI, ~3 МБ). Researcher-боты могут вызывать его через
shell-tool который у Lilush уже есть:

```python
# в коде research-бота
await shell("agent-browser open https://reddit.com/r/ffmpeg")
await shell("agent-browser snapshot")   # accessibility tree
await shell("agent-browser get text @e1")
```

Chrome для headless-режима **не вкомпилирован** в образ — `agent-browser`
скачивает его лениво в `/data/.cache` на первом запуске (~200 МБ),
дальше переиспользует благодаря persistent disk.

## 🛠 Архитектура (кратко)

```
Telegram /dl <url>  →  [download] → [analyze] → [edit] → [seo] → [publish]
                                ⬑  SQLite jobs.db  ⬐
```

Каждая стадия — отдельный async-воркер, работают параллельно:
пока `download` качает фильм #2, `editor` уже режет фильм #1, а `seo`
пишет описания для уже нарезанных клипов. Подробности —
[`.devin/tentacles/pipeline-skeleton/CONTEXT.md`](.devin/tentacles/pipeline-skeleton/CONTEXT.md).

## 🎯 3 шага до работающего бота

### 1. Создай свой Telegram-бот

1. Открой [@BotFather](https://t.me/BotFather) в Telegram
2. `/newbot` → придумай имя и username
3. Сохрани **BOT_TOKEN** (формат `1234567890:AAH...`)

### 2. Залей этот стартер в свой GitHub-репо

**Вариант А — через web-интерфейс GitHub** (самый простой, без терминала):

1. Открой <https://github.com/new>, придумай имя репо (любое)
   - **ВАЖНО**: НЕ ставь галочки на "Add a README file", ".gitignore" и "license" —
     иначе репо не будет пустым и трюк ниже не сработает
2. Нажми **Create repository**
3. На пустой странице нового репо нажми ссылку **uploading an existing file**
   (она в первом параграфе "Quick setup")
4. Распакуй `codesp-bot-starter.zip` у себя на компе → перетащи в окно загрузки
   **содержимое** папки `codesp-bot-starter/` (Dockerfile, render.yaml, bot/, и т.д.) —
   именно файлы и папку `bot/`, а не саму папку `codesp-bot-starter/`
5. Внизу нажми **Commit changes**

Готово. Все файлы (Dockerfile, render.yaml, bot/...) теперь у тебя в репо.

> Если перетаскивание папки `bot/` не срабатывает — drag&drop папок зависит от браузера.
> Используй Chrome (стабильнее всех) или Вариант Б через терминал.

**Вариант Б — через терминал** (если есть git):
```bash
unzip codesp-bot-starter.zip
cd codesp-bot-starter
git init
git add .
git commit -m "init"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 3. Развернуть в облаке

Выбери один:

#### Render (рекомендую — проще всего, бесплатно)

1. <https://dashboard.render.com> → **New +** → **Blueprint**
2. **Connect a repository** → найди свой свежий репо → **Connect**
3. Render прочитает `render.yaml` и попросит ввести `BOT_TOKEN` → вставь свой
4. **Apply** — жди 2-4 минуты

#### Railway

1. <https://railway.app/new> → **Deploy from GitHub repo** → пик свой репо
2. После создания → **Variables** → добавь `BOT_TOKEN=<токен>`
3. Авто-деплой стартанёт сам

#### Fly.io (всегда-on, не засыпает)

```bash
curl -L https://fly.io/install.sh | sh
fly auth signup
fly launch --copy-config --name <уникальное-имя>
fly secrets set BOT_TOKEN=<токен>
fly deploy
```

#### Свой VPS

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
docker build -t codesp-bot .
docker run -d --name codesp-bot --restart unless-stopped \
  -p 8080:8080 \
  -e BOT_TOKEN="<токен>" \
  -v $(pwd)/data:/data \
  codesp-bot
```

---

## 🤖 Активация в Telegram

После того как контейнер поднялся:

1. Открой свой бот в Telegram (по username)
2. Отправь `/start`
3. Нажми кнопку **🚀 Запустить и стать владельцем**
   → ты теперь единственный пользователь, кто может с ним общаться
4. Появится меню выбора «мозга» — 3 кнопки:
   - **🧠 OpenRouter** (рекомендую) — нажми её
   - **🛠 Devin.ai** — для случая когда ты хочешь подключить Devin как агента
   - **🔌 Другое** — для self-hosted endpoint-ов (vLLM, Ollama, Groq и т.д.)
5. В подменю нажми **🔑 API ключ** → вставь свой ключ (получить: <https://openrouter.ai/keys>)
6. **✅ Готово** — пиши боту любой текст, он отвечает через LLM

---

## 📂 Структура

```
.
├── README.md              ← этот файл
├── Dockerfile             ← образ контейнера
├── requirements.txt       ← Python deps
├── .env.example           ← шаблон env-переменных (для локального запуска)
├── .gitignore
├── render.yaml            ← Render Blueprint
├── railway.json           ← Railway service config
├── fly.toml               ← Fly.io app config
└── bot/                   ← Python-пакет
    ├── __init__.py
    ├── main.py            ← entrypoint (polling + health-сервер + воркеры)
    ├── config.py          ← env-vars
    ├── handlers.py        ← /help, /clone, /exec, /git, /dl, /jobs
    ├── wizard.py          ← /start, /setup, FSM, кнопки выбора мозга
    ├── storage.py         ← persistent state в data/state.json
    ├── agent.py           ← LLM-агент (OpenRouter / OpenAI-compatible)
    ├── tools.py           ← list_dir / read_file / write_file / exec_bash
    ├── inbox.py           ← лог входящих для devin-brain режима
    ├── jobs.py            ← SQLite-очередь для pipeline-джобов
    ├── workers/           ← воркеры pipeline-стадий (download/analyze/edit/seo/publish)
    └── send.py            ← CLI: python -m bot.send <chat_id> "msg"
```

## 🧪 Разработка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# линт + типы + тесты — то же что в CI
ruff check bot/ tests/
mypy bot/ tests/
BOT_TOKEN=dummy:dummy pytest tests/ -v
```

---

## 🔐 Безопасность

- **Owner-claim**: первый пользователь, который нажмёт кнопку запуска, становится **единственным**, кто может говорить с ботом. Все остальные получают «принадлежит другому владельцу».
- API-ключи **не хранятся в env-vars** на облаке — только в `data/state.json` внутри контейнера. Сообщение пользователя с ключом удаляется ботом сразу после получения.
- Файл `data/state.json` имеет права `0600` (чтение только владельцу процесса).
- `exec_bash` как tool — это полный shell без sandbox. Whitelist через owner-claim **обязателен**.

## 🆘 Troubleshooting

| Проблема | Решение |
|---|---|
| `KeyError: 'BOT_TOKEN'` в логах | Не задал env-var → добавь и передеплой |
| `TelegramUnauthorizedError` | Токен неверный → перевыпусти через `/revoke` в @BotFather |
| Кнопки не работают в группах | @BotFather → `/setprivacy` → твой бот → **Disable** |
| «уже привязан к другому владельцу» | Кто-то нажал первым (или старый state). Зайди в shell контейнера → `rm /data/state.json` → передеплой |
| Render Free засыпает | По умолчанию **уже включён self-ping** на `/healthz` каждые 4-7 минут (рандомно, со смещением к 7) — Render таймер засыпания 15 минут, поэтому бот не должен спать. Если всё равно засыпает: проверь `RENDER_EXTERNAL_URL` в env или задай `KEEP_ALIVE_URL=https://<твой-сервис>.onrender.com` явно. Если нужно отключить (например, на Fly или своём VPS) — `KEEP_ALIVE_INTERVAL=0` |

---

## Что дальше

- В TG: `/help` — список команд
- `/setup` — заново выбрать мозг и поменять ключи
- `/setbrain devin` — переключить в режим «Devin отвечает за меня» (нужна параллельная сессия Devin с шелл-доступом)
- `/keys`, `/setkey`, `/models`, `/setmodel` — низкоуровневое управление ключами и моделями
