# Bot capabilities

_This file is auto-maintained by `bot/capabilities.py`._
_Each `<!-- key: ... -->` block is appended once and never_
_rewritten, so manual edits between blocks survive restarts._

<!-- key: github-auto-clone -->
### Авто-клонирование GitHub репозиториев

I can clone a git repository (GitHub / GitLab / Bitbucket) directly from a URL the user pastes in chat, then read files, run commands, and answer questions about its code. Powered by the clone_repo / list_projects / switch_project tools.

*Пример:* Пользователь: «посмотри https://github.com/aiogram/aiogram и расскажи как у них роутеры устроены» → я вызываю clone_repo(url=...), потом list_dir, потом read_file, потом отвечаю.

<!-- key: github-settings-screen -->
### Экран ⚙️ Настройки → 🐙 GitHub

The user can browse all cloned repositories, see each repo's description, and delete repos (which removes them from disk and from my memory). There is also a toggle to disable auto-clone if they prefer explicit instructions.

<!-- key: menu-button-commands -->
### Меню-кнопка ≡ рядом с «отправить»

Telegram shows a popup with 8 quick-action commands when the user taps the round menu button next to the input (/start, /chat, /helpzavr, /pretty, /mailbox, /media, /github, /work, /settings). ⚙️ Настройки sits at the bottom.

<!-- key: main-menu-layout -->
### Главное меню после /start

Inline keyboard shown after /start has feature buttons on top — 🤖 Helpzavr, ✨ Красивый текст, 📬 Проверка почты, 🎬 Генерация фото/видео, 🐙 GitHub проекты, 🖥 Терминал — followed by admin-only install batches and at the very bottom ⚙️ Настройки so it never gets buried.

<!-- key: terminal-shortcut -->
### 🖥 Терминал — пакетный /work из меню

Main-menu and settings now have a 🖥 Терминал shortcut that drops the user straight into /work's batch mode: send a newline-separated list of shell / git / clone / exec / cd / project commands and I run them sequentially with a 1-hour per-command timeout. Exit with /cancel.

<!-- key: settings-learn-helper -->
### 📚 Узнать о функционале — LLM-помощник в настройках

⚙️ Настройки → 📚 Узнать о функционале opens a tutorial chat. Any text message in this mode is routed through the user's main LLM (Мозг 1 or custom endpoint) with the full capabilities log injected as system prompt, so the LLM can explain every feature of the bot. Exit with /cancel.

<!-- key: tts-auto-enable-on-pick -->
### 🔊 Голосовой ответчик включается автоматически при выборе

Picking a built-in voice (M1-M5 / F1-F5) or importing a Voice-Builder JSON now auto-enables the narrator AND auto-activates the imported clone as the current voice. Previously users had to flip a separate toggle and click «Сохранить как активный» — most assumed the picker was broken when in fact it just needed extra clicks.

<!-- key: tts-ram-guard -->
### 🔊 RAM-страховка: TTS не валит бот на 512 MB хостинге

Supertonic synthesis peaks ~470 MB RSS, which OOM-kills the bot on Render Free/Starter (both 0.5 GB). The TTS pipeline now reads cgroup memory accounting before each synthesis and silently skips with a logged warning when <450 MB is free (configurable via TTS_MIN_AVAILABLE_MB). On capped hosts the user still sees text replies; the ⚙️ Настройки → 🔊 Голосовой ответчик screen surfaces the exact reason so people know voice replies need Standard (2 GB) or bigger plan to work.

<!-- key: tts-baked-model -->
### 🔊 Голосовая модель запечена в Docker-образ

The 386 MB Supertonic ONNX model is downloaded during the Docker build (where Render gives the builder lots of RAM/bandwidth) instead of at runtime on the cramped container. Previously the first voice reply would trigger a HuggingFace download that OOM-killed the bot mid-transfer and restart-looped. Now the runtime container starts with the model already on disk.

<!-- key: unified-thinking-style -->
### Единая Соображалка во всех потоках

Whatever the user picks in ⚙️ Настройки → 🧠 Соображалка applies everywhere: main chat with LLM, Helpzavr photo pipeline, Pretty Text → Standard, Mailbox, and media generation. Three styles: ⚪ White, ⚪ White + model (suffix the model name), ⬛ Black/native typing-indicator.

<!-- key: access-modes -->
### 3 режима доступа + со-владельцы

I support private / public / full_public access modes plus a co-owner list managed via ⚙️ Настройки → ⛔ Доступ. In public mode strangers can USE features but cannot see API keys / models / stats; in full_public they can do everything; in private only owner + co-owners get in.

<!-- key: algorithm-slots -->
### 🧩 Алгоритм — 10 слотов с пошаговыми планами

⚙️ Настройки → 🧩 Алгоритм — 10 слотов на чат. В каждый слот можно сохранить пошаговый план: либо вручную, либо по тексту/новости — бот сам разложит на шаги через LLM. Приоритет планировщика: «Другое» (custom OpenAI-совместимый) → Brain 1 → Brain 2. Кнопка «▶ Запустить» исполняет шаги строго последовательно через агентский режим — каждый шаг видит ВСЕ тулзы бота: локальные (read_file, exec_bash, write_file) и внешние (tavily_search, brave_search, exa_search, firecrawl_scrape, apify_run_actor, github_search_code, github_get_file). Падение одного шага останавливает алгоритм с понятной причиной.

<!-- key: algorithm-periodic -->
### 🧩 Алгоритм — периодика (мин / час / день / неделя / месяц)

У каждого слота есть «⏱ Периодичность: N мин» — дробные ОК (0.5 = 30 сек). Подсказки: 1440 = день, 10080 = неделя, 43200 = месяц. Фоновый scheduler-task просыпается каждые 30 сек и запускает все слоты у которых пришло время. Идеально для новостного мониторинга: «каждые 60 мин сходи tavily_search по теме X и пришли резюме».

<!-- key: external-research-tools -->
### 🛠 Внешние API-тулзы агента (Tavily / Brave / Exa / Firecrawl / Apify / GitHub)

Агент умеет ходить в интернет: tavily_search (AI-tuned поиск с готовым answer'ом), brave_search (Google-like), exa_search (семантический поиск похожих страниц), firecrawl_scrape (URL → markdown), apify_run_actor (скрейперы TikTok / Reddit / Twitter и др.), github_search_code + github_get_file (поиск и чтение кода с GitHub). Ключи задаются в /setup → 🛠 Внешние API; без ключа тулза возвращает дружелюбное «открой /setup и вставь ключ» вместо краша.

<!-- key: elevenlabs-tts -->
### ☁️ ElevenLabs — облачный TTS-провайдер

⚙️ Настройки → 🔊 Голосовой ответчик → ☁️ ElevenLabs. Альтернатива локальному Supertonic — синтез делается на стороне ElevenLabs, не ест RAM. Поля: 🔑 API ключ + 🎤 Voice ID (копируется из <a href='https://elevenlabs.io/app/voice-library'>Voice Library</a>). URL хардкод. Кнопка «▶ Использовать ElevenLabs» делает его активным TTS-провайдером — встроенные голоса (M1..F5) и клон помечаются «▶ Использовать встроенный голос» / «▶ Использовать клон» в своих подменю. Free-тариф ElevenLabs: 10k символов/мес. Если ключ/Voice ID не заданы или API упал — мягкий фоллбэк на локальный Supertonic с warning'ом в логе.

<!-- key: brain2-quick-pick-models -->
### 🧠 Мозг 2 — быстрый выбор моделей (Whisper / Llama-Scout / Llama-70b)

⚙️ Настройки → 🧠 Перенастроить мозг → выбери слот 2 → 🤖 Модель. Открывается клавиатура с быстрыми пресетами: 🎙 whisper-large-v3 (⭐ дефолт), 🎙 whisper-large-v3-turbo, 🦙 llama-4-scout-17b-16e-instruct, 🦙 llama-3.3-70b-versatile, и ✍️ Свой вариант (ручной ввод для редких моделей). Все пресеты — Groq endpoint; для своего ввода помни задать base_url через 🌐 URL.

<!-- key: batch-api-import -->
### 🔌 API — массовый импорт ключей одной строкой

⚙️ Настройки → 🔌 API (массовый импорт). Открывает поле ввода — пришли одной строкой все ключи которые хочешь загрузить. Формат: <code>target:key:value</code> внутри поля, <code>;</code> между полями, <code>$$$</code> между независимыми блоками. Поддерживаются: brain1/brain2 (api/model/url/provider), elevenlabs (api/voice), tavily, brave, exa, firecrawl, apify, github_pat. Сообщение с ключами автоматически удаляется из истории чата сразу после применения. Бот возвращает отчёт: «применено» + «ошибки» построчно.
