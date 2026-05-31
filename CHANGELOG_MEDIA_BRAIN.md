# Changelog — Media Brain

## Что добавлено

### 🎬 Новый раздел `/setup → Мозг для видео и фото`
- До 3 провайдер-слотов (URL + API key + per-task model)
- Динамическое имя слота по URL (`deapi`, `fal`, `replicate`, `runway`)
- Per-task активный слот: видео / фото / убрать-фон могут жить на разных провайдерах
- Подсказки моделей (preset slug + описание) для deAPI

### 📷 Фото-хэндлер с тремя ветками
После присылки любого фото — inline-клавиатура `📷 Фото / 🎬 Видео / ❌ Назад`.

#### Фото-ветка (img2img + rmbg)
- Свой промпт текстом → LLM-перевод RU→EN → img2img → **JPG-превью + PNG-файл без сжатия**
- **«✨ Указ»** → ключевые слова → LLM генерирует 3 варианта на русском → выбор → перевод на EN → img2img
- **«🪄 Убрать фон»** → Ben2 → **JPG-превью + PNG-документ с альфой** (готово для Photoshop)

#### Видео-ветка (img2video)
- Pre-quality screen: `✨ Указ / → Пропустить / ↩️ Вернуться`
- Свой промпт текстом → LLM-перевод RU→EN
- «Указ» → 3 RU-варианта → выбор → перевод
- Quality picker: **480 / 720 / 1080 / 2К / Авто / 🎞 GIF**
- Возврат: **inline-видео + MP4-файл без сжатия**
- Для GIF: ffmpeg-конвертация → настоящий .gif (с fallback на mp4-as-animation если ffmpeg недоступен)

### 🧠 LLM-помощник промптов
- Использует тот же OpenRouter-клиент, что и основной чат-агент
- `generate_prompt_variants()` — 3 развёрнутых RU-промпта из ключевых слов
- `translate_to_english()` — RU→EN перевод перед отправкой в diffusion-модель
- Graceful fallback: если LLM не настроен, текст уходит как есть с warning'ом

## Файлы

| Файл | Изменение |
|---|---|
| `bot/media.py` | **новый** — deAPI клиент (~410 строк) |
| `bot/media_llm.py` | **новый** — LLM-помощник (~165 строк) |
| `bot/media_ui.py` | **новый** — UI + handlers (~1150 строк) |
| `bot/storage.py` | + media-слоты (~120 строк) |
| `bot/wizard.py` | + кнопка `🎬 Мозг для видео и фото` |
| `bot/main.py` | + регистрация `media_router` |
| `bot/handlers.py` | − удалён старый `F.photo` хэндлер |
| `bot/i2v.py` | **удалён** — заменён на `bot/media.py` |
| `.env.example` | + DEAPI_API_KEY как back-compat fallback |
| `README.md` | + секция «Media Brain» |
| `DEPLOY.md` | **новый** — инструкция деплоя |

## Совместимость
- Все 106 существующих тестов проходят
- `ruff check bot/` — чисто
- Не сломано: LLM-мозг, внешние API, /role, /clone, воркеры
- env-var `DEAPI_API_KEY` работает как fallback — старая интеграция (`DEAPI_API_KEY` → photo→video) продолжит работать «как раньше» если новый wizard не настраивать
