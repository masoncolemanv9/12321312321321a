FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    PORT=8080 \
    # Persist HuggingFace model cache on the mounted disk so other
    # HF downloads (faster-whisper etc.) survive container redeploys.
    HF_HOME=/data/hf_cache \
    # Supertonic uses its OWN cache dir (~/.cache/supertonic3) and
    # ignores HF_HOME. Point it at a baked-in location inside the
    # image: the model itself is pre-fetched in a later RUN step so
    # the runtime container never tries to download ~200MB on a
    # 512MB Render Free box (the previous deploys OOM-killed mid-
    # download on the FIRST voice reply, restarted forever).
    SUPERTONIC_CACHE_DIR=/opt/supertonic_cache \
    # ONNX runtime is happiest with single-threaded matmul on a 512MB
    # box; without these limits supertonic threads explode RSS during
    # synthesis. Safe to remove on bigger plans.
    SUPERTONIC_INTRA_OP_THREADS=1 \
    SUPERTONIC_INTER_OP_THREADS=1 \
    # faster-whisper / huggingface_hub may try to use the new XET
    # protocol which parallel-downloads with a noticeable RAM spike.
    # Disable it on memory-constrained free hosting.
    HF_HUB_DISABLE_XET=1

WORKDIR /app

# Base system deps. Includes Node.js (for vercel-labs/agent-browser CLI
# that research bots invoke via shell) and Chrome runtime libs so the
# bundled Chrome from agent-browser can launch headless.
#
# Tesseract OCR (+ Russian language pack) is the Helpzavr addon's
# requirement — same packages used by tg_bot+(13) so behavior matches.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates curl gnupg \
        # Chrome runtime dependencies (libs needed by headless Chrome) —
        # let agent-browser / browser-harness install Chrome itself.
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libdrm2 fonts-liberation \
        # Helpzavr addon — OCR + image processing libs.
        tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
        libgl1 libglib2.0-0 \
        # 🔊 Голосовой ответчик — ffmpeg converts the Supertonic WAV
        # output into the OGG/Opus voice format Telegram accepts.
        # libsndfile1 is required by soundfile (used by supertonic).
        ffmpeg libsndfile1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install agent-browser CLI globally. We DO NOT run `agent-browser
# install` here (it downloads ~200MB of Chrome). Researcher bots that
# actually need a browser will run it lazily on first use, storing
# Chrome under /data so it persists across redeploys without bloating
# the image.
RUN npm install -g agent-browser@latest

# Install browser-harness (self-healing CDP harness from browser-use).
# Cloned to /opt so it's outside the app code; installed editable so
# the agent's runtime edits to agent-workspace/agent_helpers.py
# persist across the running bot's lifetime. Researcher bots invoke
# the `browser-harness` binary through the shell-tool.
RUN git clone --depth=1 https://github.com/browser-use/browser-harness /opt/browser-harness \
    && pip install --no-cache-dir -e /opt/browser-harness

# python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-fetch the Supertonic voice model (~200MB) into the image during
# BUILD on Render's build infrastructure (plenty of RAM and bandwidth)
# rather than at RUNTIME on a 512MB Free box (where the download was
# OOM-killing the container halfway through 26 files). Storing it in
# /opt/supertonic_cache (SUPERTONIC_CACHE_DIR set above) means the
# runtime container starts with the model already on disk — first
# voice reply just mmap's the ONNX files instead of fetching them.
RUN mkdir -p /opt/supertonic_cache \
    && python -c "from supertonic.loader import download_model, get_cache_dir; download_model(get_cache_dir())" \
    && du -sh /opt/supertonic_cache

# app code
COPY bot/ /app/bot/

# data dir for persisted state.json + Chrome cache from agent-browser
RUN mkdir -p /data

EXPOSE 8080
CMD ["python", "-m", "bot.main"]
