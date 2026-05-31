# Деплой бота с фичей «🎬 Мозг для видео и фото»

Полная инструкция для развёртывания на Render (free-tier подходит).
Все примеры — для **Render Blueprint** (репо уже содержит `render.yaml`).

---

## 0. Что нового по сравнению с твоей текущей версией

| Файл | Что |
|---|---|
| `bot/media.py` | **новый** — клиент deAPI (img2video / img2img / img-rmbg) с поддержкой LTX-Video / Qwen-Image-Edit / Ben2 |
| `bot/media_llm.py` | **новый** — LLM-помощник (3 RU-варианта промпта + RU→EN перевод) через твой OpenRouter |
| `bot/media_ui.py` | **новый** — wizard `🎬 Мозг для видео и фото` + photo-handler с ветками Фото / Видео / Убрать-фон |
| `bot/storage.py` | + media-слоты (`media.slots.1..3`, `media.active.{video,photo,rmbg}`) |
| `bot/wizard.py` | + кнопка `🎬 Мозг для видео и фото` в `/setup` |
| `bot/main.py` | регистрация `media_router` между wizard и main |
| `bot/handlers.py` | удалён старый `F.photo` handler (заменён в media_ui) |
| `.env.example` | блок про `DEAPI_API_KEY` (back-compat fallback) |
| `README.md` | секция «Media Brain» |

Старая фича `bot/i2v.py` удалена.

---

## 1. Что нужно перед деплоем

### 1.1. Telegram bot token
Если ты деплоишь **новый** бот: создай его через [@BotFather](https://t.me/BotFather) → `/newbot` → запиши токен.

Если **обновляешь существующий** на Render: токен уже стоит в env-vars — пропусти этот шаг.

### 1.2. deAPI ключ
Зарегистрируйся на <https://deapi.ai/dashboard> → создай API key. $5 на старте.

### 1.3. OpenRouter ключ (для «✨ Указ»)
Без него работает всё, кроме LLM-генерации промптов. Если уже стоял — ничего делать не надо.
<https://openrouter.ai/keys>

---

## 2. Деплой на Render — обновление существующего сервиса

1. **Залей изменения в свой GitHub-репо.** Я положил готовый коммит в этот ZIP — распакуй, скопируй файлы поверх своего репо, проверь diff в git, закоммить и запушь:
   ```bash
   unzip lilush.zip -d /tmp/lilush-update
   rsync -av --exclude '.git' /tmp/lilush-update/ /path/to/your/lilush/
   cd /path/to/your/lilush
   git add -A
   git commit -m "feat: media brain (deAPI img2video/img2img/rmbg + LLM prompt helper + GIF)"
   git push
   ```

2. **Добавь env-var на Render:**
   - Render Dashboard → твой сервис `lilush-boss` → **Environment**
   - Add Environment Variable:
     - Key: `DEAPI_API_KEY`
     - Value: `твой-deAPI-ключ`
   - Save Changes

3. **Render автоматически передеплоит** из нового коммита (~2-3 мин).

4. **В Telegram:**
   - `/setup` → 🧠 Мозг → убедись что OpenRouter ключ настроен (нужен для «✨ Указ»)
   - `/setup` → 🎬 Мозг для видео и фото → опционально настрой слот через UI (если не настраивать — будет использован `DEAPI_API_KEY` из env)
   - Кидай фото в чат

---

## 3. Деплой на Render с нуля (Blueprint)

1. **Создай GitHub-репо** (приватный/публичный — не важно), запушь содержимое ZIP туда.

2. **Render** → **New +** → **Blueprint** → **Connect a repository** → выбери свой репо.

3. Render прочтёт `render.yaml` и предложит 20 сервисов. Для теста медиа-фичи нужен только `lilush-boss` — остальные оставь с пустыми токенами (поднимутся в dormant-режиме, ресурсов не жрут).

4. На сервисе `lilush-boss` → Environment → Add:
   - `BOT_TOKEN` = токен от @BotFather
   - `DEAPI_API_KEY` = ключ от deAPI
   - (опционально) `OPENROUTER_API_KEY` = ключ от OpenRouter, если хочешь сразу запихнуть в env вместо `/setup`

5. **Apply** → Render развернёт за ~3-5 минут.

6. В Telegram: `/start` → клейм владельца → `/setup` → настрой мозги → шли фото.

---

## 4. UX-флоу для пользователя

### Фото-ветка (после `📷 Фото`)
1. Бот показывает: **«✨ Указ»** / **«🪄 Убрать фон»** / **«↩️ Вернуться»**
2. Опции:
   - **Написать промпт текстом** — переведу на EN через LLM → img2img → отдам **JPG-превью + PNG-файл без сжатия**
   - **«✨ Указ»** → пришли ключевые слова → LLM сгенерирует **3 варианта на русском** → выбираешь номер → перевожу на EN → img2img → дубль-выдача
   - **«🪄 Убрать фон»** → запускаю Ben2 → отдаю **JPG-превью + PNG-документ с альфой** (готово для Photoshop)

### Видео-ветка (после `🎬 Видео`)
1. Бот показывает экран промпта: **«✨ Указ»** / **«→ Пропустить (caption/дефолт)»** / **«↩️ Вернуться»**
2. Опции:
   - Свой промпт текстом — переведу на EN → quality picker
   - «✨ Указ» — то же что и для фото, но варианты заточены под движение/ракурс/эмоцию
   - «Пропустить» — caption из подписи фото либо дефолт `Cinematic gentle motion...`
3. **Quality picker:** 480 / 720 / 1080 / 2К / Авто / 🎞 GIF / ↩️ Вернуться
4. После выбора → img2video → отдаю **inline-видео + MP4-файл без сжатия**.
   Для GIF: если ffmpeg доступен → конвертирую mp4→gif, отдаю inline-анимацию + .gif файл. Если нет — mp4-as-animation (TG всё равно покажет как GIF).

---

## 5. Мульти-провайдер и слоты

В `/setup → 🎬 Мозг для видео и фото`:
- До **3 слотов**. URL + API key + per-task model (video / photo / rmbg)
- Имя слота **автоматически** становится `deapi` / `fal` / `replicate` / `runway` по URL
- Для каждой задачи можно выбрать **разный** активный слот:
  - `video` ← slot 1 (deAPI)
  - `photo` ← slot 2 (например, fal.ai)
  - `rmbg` ← slot 1 (deAPI)
- Слот можно очистить кнопкой 🗑️ — стираются все его поля

**Поддерживаемые провайдеры:**
| Провайдер | URL | Статус |
|---|---|---|
| **deAPI.ai** | `https://api.deapi.ai` | ✅ полная (img2video LTX-Video, img2img Qwen/Flux, img-rmbg Ben2) |
| **fal.ai** | `https://fal.run` | 🚧 stub (URL и пресет моделей подсказан, но shape `_submit_and_poll` отличается — допилить в `bot/media.py`) |
| **Replicate** | `https://api.replicate.com` | 🚧 stub |
| **Runway** | `https://api.runwayml.com` | 🚧 stub |
| **Custom** | любой URL | имя слота останется как номер `1/2/3` |

---

## 6. Стоимость (deAPI)

Примерные расходы (на твоих $5 хватит надолго):
- img2video LTX-Video 0.9.8 (480p/720p) ~$0.003 за 2-сек ролик
- img2video LTX-2 19B (1080p/2K) ~$0.02 за ролик
- img2img Qwen-Image-Edit ~$0.002 за картинку
- img-rmbg Ben2 ~$0.001 за фото

LLM-вызовы (OpenRouter) — используется free-модель из твоего fallback-списка, расход ≈ $0.

---

## 7. Проверка после деплоя

```bash
# health-check (Render)
curl https://<твой-сервис>.onrender.com/healthz
# → "ok"
```

В Telegram:
1. `/start` → должен ответить как обычно (проверка что бот жив)
2. Кинь любое фото → должны появиться `📷 Фото / 🎬 Видео / ❌ Назад`

Если бот не отвечает на фото — посмотри логи Render. Самые частые проблемы:
- `MediaError: no slot configured for task=...` — нужно либо настроить слот через `/setup`, либо проверить что `DEAPI_API_KEY` в env-vars
- `NoLLMError` при «Указ» — настроить OpenRouter ключ через `/setup → 🧠 Мозг`
- HTTP 401/403 от deAPI — ключ невалидный или баланс на нуле

---

## 8. Free-tier Render: важно

- Бот спит после 15 мин неактивности. Self-ping каждые 4-7 мин уже встроен (`KEEP_ALIVE_*` в `bot/main.py`) — держит сервис в рамках 750 free-часов/мес
- Когда фото уходит на генерацию, deAPI/fal крутит на своих GPU. Твоему Render-сервису это бесплатно — он только держит open HTTP-соединение
- ffmpeg на free-tier `python:3.12-slim` Dockerfile-образе **отсутствует**. Если нужен настоящий .gif — добавь в `Dockerfile`:
  ```dockerfile
  RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/*
  ```
  Без этого «🎞 GIF» отработает как mp4-as-animation — Telegram всё равно покажет это как зацикленную анимацию, просто скачиваться будет .mp4 файл.

---

## 9. Лицензия / автор
Без изменений по сравнению с твоим оригинальным репо.

---

## 10. Если что-то сломалось
- Логи Render: Dashboard → сервис → Logs
- Локальный smoke-test: `pytest tests/ -x` (106 тестов проходят)
- Lint: `ruff check bot/`
