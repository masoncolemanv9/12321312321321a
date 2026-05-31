"""Analyzer stage: turn a downloaded video into clip windows.

Pipeline:

1. Extract a low-bitrate mono 16 kHz WAV from the source via ``ffmpeg``.
2. Transcribe it with ``faster-whisper`` (model size from env).
3. Run ``pyscenedetect`` against the source for scene boundaries.
4. Build candidate windows (``ANALYZER_CLIP_MIN_S..ANALYZER_CLIP_MAX_S``)
   aligned to scene cuts.
5. Rank candidates: try an LLM ranker first (OpenRouter), fall back to a
   transcript-density + scene-cut heuristic if the API call fails or no
   key is configured.
6. Return the top ``ANALYZER_TARGET_CLIPS`` clips and fan out into
   ``edit`` jobs (one per clip).

Each artifact is persisted under ``data/downloads/<job_id>/`` so the
downstream stages can re-read them without redoing work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config as _config
from ..jobs import Job
from .base import Worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Segment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class _Scene:
    start_s: float
    end_s: float


@dataclass
class _Clip:
    start_s: float
    end_s: float
    hook: str
    score: float


class AnalyzerWorker(Worker):
    kind = "analyze"
    next_kind = "edit"

    async def process(self, job: Job) -> dict[str, Any]:
        # Downloader emits ``source_path``; older callers used ``file_path``.
        source_path_str = job.payload.get("source_path") or job.payload.get(
            "file_path"
        )
        if not source_path_str:
            raise ValueError("analyze payload missing 'source_path'")
        source_path = Path(source_path_str)
        if not source_path.exists():
            raise FileNotFoundError(f"analyze source not found: {source_path}")

        duration_s = float(job.payload.get("duration_s") or 0.0)
        target_count = int(
            job.payload.get("target_clip_count") or _config.ANALYZER_TARGET_CLIPS
        )
        clip_min = float(job.payload.get("clip_min_s") or _config.ANALYZER_CLIP_MIN_S)
        clip_max = float(job.payload.get("clip_max_s") or _config.ANALYZER_CLIP_MAX_S)

        work_dir = source_path.parent
        artifacts_dir = work_dir / "analyzer"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        audio_path = artifacts_dir / "audio.wav"
        transcript_path = artifacts_dir / "transcript.json"
        scenes_path = artifacts_dir / "scenes.json"

        # Heavy CPU-bound work runs in a thread so the worker loop stays free.
        return await asyncio.to_thread(
            self._analyze_blocking,
            source_path=source_path,
            audio_path=audio_path,
            transcript_path=transcript_path,
            scenes_path=scenes_path,
            duration_s=duration_s,
            target_count=target_count,
            clip_min=clip_min,
            clip_max=clip_max,
            language=job.payload.get("language"),
        )

    @staticmethod
    def _analyze_blocking(
        *,
        source_path: Path,
        audio_path: Path,
        transcript_path: Path,
        scenes_path: Path,
        duration_s: float,
        target_count: int,
        clip_min: float,
        clip_max: float,
        language: str | None,
    ) -> dict[str, Any]:
        extract_audio(source_path, audio_path)
        segments, detected_lang = transcribe(audio_path, language=language)
        transcript_path.write_text(
            json.dumps(
                {
                    "language": detected_lang,
                    "segments": [s.__dict__ for s in segments],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        scenes = detect_scenes(source_path)
        scenes_path.write_text(
            json.dumps([s.__dict__ for s in scenes], ensure_ascii=False, indent=2)
        )

        if duration_s <= 0:
            duration_s = max(
                [s.end_s for s in segments] + [s.end_s for s in scenes],
                default=0.0,
            )

        candidates = build_candidate_windows(
            scenes=scenes,
            duration_s=duration_s,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        clips, method = pick_top_clips(
            candidates=candidates,
            segments=segments,
            target_count=target_count,
        )

        return {
            "source_path": str(source_path),
            "duration_s": duration_s,
            "language": detected_lang,
            "transcript_path": str(transcript_path),
            "scenes_path": str(scenes_path),
            "analysis_method": method,
            "clips": [
                {
                    "start_s": c.start_s,
                    "end_s": c.end_s,
                    "hook": c.hook,
                    "score": c.score,
                }
                for c in clips
            ],
        }

    async def enqueue_next(self, job: Job, result: dict[str, Any]) -> None:
        """Fan out: one ``edit`` job per detected clip."""
        source_path = result["source_path"]
        for idx, clip in enumerate(result.get("clips", [])):
            await self.queue.enqueue(
                self.next_kind or "edit",
                {
                    "source_path": source_path,
                    "duration_s": result.get("duration_s"),
                    "language": result.get("language"),
                    "transcript_path": result.get("transcript_path"),
                    "clip_index": idx,
                    **clip,
                },
                chat_id=job.chat_id,
                parent_id=job.id,
            )


# ---- helpers -------------------------------------------------------------


def extract_audio(source_path: Path, audio_path: Path) -> None:
    """Extract a 16 kHz mono WAV via ffmpeg.

    16 kHz mono is what whisper expects internally — anything else just
    burns CPU on resampling.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr.strip()}") from exc


def transcribe(
    audio_path: Path, *, language: str | None
) -> tuple[list[_Segment], str]:
    """Run faster-whisper on ``audio_path``. Returns (segments, language)."""
    model = _load_whisper_model()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments: list[_Segment] = []
    for seg in segments_iter:
        segments.append(
            _Segment(
                start_s=float(seg.start),
                end_s=float(seg.end),
                text=str(seg.text).strip(),
            )
        )
    return segments, str(getattr(info, "language", language or "und"))


_WHISPER_MODEL_CACHE: dict[str, Any] = {}


def _load_whisper_model() -> Any:
    """Load (and cache) a whisper model so subsequent jobs reuse it."""
    from faster_whisper import WhisperModel

    cache_key = (
        f"{_config.WHISPER_MODEL_SIZE}/{_config.WHISPER_DEVICE}"
        f"/{_config.WHISPER_COMPUTE_TYPE}"
    )
    if cache_key not in _WHISPER_MODEL_CACHE:
        logger.info("loading whisper model: %s", cache_key)
        _WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
            _config.WHISPER_MODEL_SIZE,
            device=_config.WHISPER_DEVICE,
            compute_type=_config.WHISPER_COMPUTE_TYPE,
        )
    return _WHISPER_MODEL_CACHE[cache_key]


def detect_scenes(source_path: Path) -> list[_Scene]:
    """Run pyscenedetect's content-aware detector against ``source_path``."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(source_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=_config.SCENEDETECT_THRESHOLD))
    sm.detect_scenes(video=video)
    raw = sm.get_scene_list()
    out: list[_Scene] = []
    for start, end in raw:
        out.append(
            _Scene(start_s=float(start.get_seconds()), end_s=float(end.get_seconds()))
        )
    return out


def build_candidate_windows(
    *,
    scenes: list[_Scene],
    duration_s: float,
    clip_min: float,
    clip_max: float,
) -> list[_Scene]:
    """Build clip-length windows from scene boundaries.

    Strategy: for each scene cut, emit a window starting there and ending
    at min(scene_end, scene_start + clip_max). Skip windows shorter than
    ``clip_min``. If pyscenedetect returned nothing (e.g. the file was a
    static talking-head shot), fall back to fixed-stride windows.
    """
    candidates: list[_Scene] = []
    if scenes:
        for scene in scenes:
            length = scene.end_s - scene.start_s
            if length < clip_min:
                continue
            end = min(scene.end_s, scene.start_s + clip_max)
            candidates.append(_Scene(start_s=scene.start_s, end_s=end))
    if not candidates and duration_s > clip_min:
        # Fallback: walk the whole video in clip_max-sized strides.
        stride = clip_max
        cursor = 0.0
        while cursor + clip_min <= duration_s:
            candidates.append(
                _Scene(start_s=cursor, end_s=min(cursor + stride, duration_s))
            )
            cursor += stride
    return candidates


def pick_top_clips(
    *,
    candidates: list[_Scene],
    segments: list[_Segment],
    target_count: int,
) -> tuple[list[_Clip], str]:
    """Select ``target_count`` clips. Returns (clips, method)."""
    if not candidates:
        return [], "heuristic"

    # Try the LLM ranker first; fall back to the heuristic if it fails for
    # any reason (no key, network down, malformed JSON).
    if _config.OPENROUTER_API_KEY:
        try:
            picks = _rank_with_llm(
                candidates=candidates,
                segments=segments,
                target_count=target_count,
            )
            if picks:
                return picks, "llm"
        except Exception as exc:  # noqa: BLE001 — fallback path
            logger.warning("LLM ranker failed, using heuristic: %s", exc)

    return _rank_with_heuristic(
        candidates=candidates,
        segments=segments,
        target_count=target_count,
    ), "heuristic"


def _segments_inside(window: _Scene, segments: list[_Segment]) -> list[_Segment]:
    """Segments whose midpoint falls inside ``window``."""
    out: list[_Segment] = []
    for seg in segments:
        mid = (seg.start_s + seg.end_s) / 2
        if window.start_s <= mid <= window.end_s:
            out.append(seg)
    return out


def _rank_with_heuristic(
    *,
    candidates: list[_Scene],
    segments: list[_Segment],
    target_count: int,
) -> list[_Clip]:
    """Score by transcript density (chars per second of video).

    A talky scene generally makes a stronger short than a silent one.
    """
    scored: list[tuple[float, _Scene, str]] = []
    for window in candidates:
        inside = _segments_inside(window, segments)
        chars = sum(len(s.text) for s in inside)
        length = max(window.end_s - window.start_s, 1.0)
        density = chars / length
        hook = " ".join(s.text for s in inside)[:200] or "untitled clip"
        scored.append((density, window, hook))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:target_count]
    return [
        _Clip(
            start_s=window.start_s,
            end_s=window.end_s,
            hook=hook,
            score=density,
        )
        for density, window, hook in top
    ]


def _rank_with_llm(
    *,
    candidates: list[_Scene],
    segments: list[_Segment],
    target_count: int,
) -> list[_Clip]:
    """Ask an LLM to pick the most clip-worthy windows.

    Returns an empty list if the model response can't be parsed; the
    caller will then fall back to the heuristic.
    """
    import httpx

    # Build a compact transcript view per candidate so the prompt fits.
    bullets: list[dict[str, Any]] = []
    for idx, window in enumerate(candidates[:30]):  # cap for token budget
        inside = _segments_inside(window, segments)
        bullets.append(
            {
                "id": idx,
                "start_s": round(window.start_s, 2),
                "end_s": round(window.end_s, 2),
                "transcript": " ".join(s.text for s in inside)[:600],
            }
        )

    system = (
        "You are a short-form video editor. Pick the most engaging clips for "
        "TikTok / Shorts / Reels. Reply with strict JSON: a list of objects "
        "{id, hook}. Pick exactly N items where N is given. The 'hook' is a "
        "short caption (max 80 chars) you'd put on the clip."
    )
    user = json.dumps(
        {
            "n_to_pick": target_count,
            "candidates": bullets,
        },
        ensure_ascii=False,
    )

    payload = {
        "model": os.environ.get("ANALYZER_LLM_MODEL", "openai/gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {_config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM ranker returned unparseable response: {exc}") from exc

    picks_raw = parsed.get("picks") or parsed.get("clips") or parsed
    if not isinstance(picks_raw, list):
        raise RuntimeError("LLM ranker JSON is not a list of picks")

    out: list[_Clip] = []
    for entry in picks_raw[:target_count]:
        try:
            cid = int(entry["id"])
            hook = str(entry.get("hook") or "")
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= cid < len(candidates):
            window = candidates[cid]
            out.append(
                _Clip(
                    start_s=window.start_s,
                    end_s=window.end_s,
                    hook=hook[:200] or "untitled clip",
                    score=1.0,  # LLM picks aren't ordered numerically
                )
            )
    return out
