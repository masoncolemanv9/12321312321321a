"""Sampled YuNet face detection (v2.0 baseline).

Spec §6.5: v2.0 face-aware reframe uses a **constant** crop center
derived from the median of ~10 evenly-spaced YuNet detections per
clip — NOT per-frame tracking. v6 (Phase C) replaces this with a
time-varying zoom_curve.

Failure policy (§9.13, summarised):

* missing ONNX model file → ``warn`` + center fallback
* ``import cv2`` fails (extras not installed) → ``warn`` + center fallback
* detection returns no faces → ``warn`` + center fallback
* invalid bbox dimensions → ``warn`` + center fallback

Tests monkeypatch ``cv2`` so the production import path stays lazy.
"""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 — argv-list only
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import AssetError, ensure_yunet_model
from .probe import ProbeResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceSample:
    """One detected face at one sampled timestamp.

    Bounding-box fields are in **percent of frame size** (0..1 floats),
    not absolute pixels, so they stay valid after later scale/crop
    stages.
    """

    timestamp_s: float
    bbox_pct: tuple[float, float, float, float]  # x, y, w, h in 0..1
    confidence: float

    def center_pct(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_pct
        return x + w / 2.0, y + h / 2.0


@dataclass(frozen=True, slots=True)
class FaceSampling:
    """Aggregated result of :func:`sample_faces` over a clip."""

    samples: tuple[FaceSample, ...]
    median_center_pct: tuple[float, float]
    fallback: bool  # True when median_center_pct is the geometric center
    reason: str = ""

    @property
    def detected_face_count(self) -> int:
        return len(self.samples)


_CENTER = (0.5, 0.5)


def _evenly_spaced_timestamps(
    duration_s: float, *, count: int, margin_s: float = 0.1
) -> list[float]:
    """``count`` timestamps spread across the clip with a small margin on each end."""
    if count <= 0 or duration_s <= 0:
        return []
    if count == 1:
        return [duration_s / 2.0]
    start = max(0.0, margin_s)
    end = max(start, duration_s - margin_s)
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty sequence")
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _try_import_cv2() -> Any | None:
    """Lazy ``cv2`` import; returns module or None when missing.

    Importing OpenCV when extras aren't installed must not blow up
    package import — that would break ``EDITOR_VERSION=v1`` deploys.
    """
    try:
        import cv2  # noqa: PLC0415 — lazy import is the whole point
    except ImportError as exc:
        logger.warning("cv2 not importable: %s (face reframe disabled)", exc)
        return None
    return cv2


def _extract_frame(
    source_path: Path, timestamp_s: float, out_path: Path, ffmpeg_bin: str = "ffmpeg"
) -> bool:
    """Render a single PNG at ``timestamp_s``; return True on success.

    Failures are logged but never raised — the caller will simply drop
    the missing sample and fall back if all samples fail.
    """
    argv = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(source_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    try:
        subprocess.run(  # noqa: S603 — argv list, no shell
            argv, check=True, capture_output=True, timeout=30.0
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as exc:
        logger.warning("frame extract @%.2fs failed: %s", timestamp_s, exc)
        return False
    return out_path.exists()


def _detect_one(
    cv2_mod: Any,
    detector: Any,
    image_path: Path,
    timestamp_s: float,
) -> FaceSample | None:
    """Run YuNet on a single PNG and convert to :class:`FaceSample`.

    Returns the highest-confidence face. None if the image fails to
    load or no faces are detected. ``detector`` is the prebuilt
    ``cv2.FaceDetectorYN`` instance — we don't re-create it per sample
    to avoid the ONNX init cost.
    """
    img = cv2_mod.imread(str(image_path))
    if img is None:
        logger.warning("cv2.imread failed for %s", image_path)
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # YuNet returns [x, y, w, h, ...landmarks..., confidence]; pick max-conf.
    best = max(faces, key=lambda row: float(row[-1]))
    fx, fy, fw, fh = (float(best[0]), float(best[1]), float(best[2]), float(best[3]))
    conf = float(best[-1])
    if fw <= 0 or fh <= 0:
        return None
    return FaceSample(
        timestamp_s=timestamp_s,
        bbox_pct=(fx / w, fy / h, fw / w, fh / h),
        confidence=conf,
    )


def _fallback(reason: str) -> FaceSampling:
    return FaceSampling(
        samples=(),
        median_center_pct=_CENTER,
        fallback=True,
        reason=reason,
    )


def sample_faces(
    probe: ProbeResult,
    *,
    model_path: Path,
    expected_sha256: str = "",
    sample_count: int = 10,
    ffmpeg_bin: str = "ffmpeg",
    confidence_threshold: float = 0.7,
) -> FaceSampling:
    """Sample ``sample_count`` evenly-spaced frames and detect faces.

    Returns a :class:`FaceSampling`:

    * ``median_center_pct`` is the median of all successful samples
      (median per-axis), which becomes the constant crop center for
      v2.0 zoom-reframe (§6.5).
    * ``fallback=True`` means the caller should ignore any samples
      and use the geometric center (0.5, 0.5).

    This function is **synchronous** even though :class:`ProbeResult`
    came from an async probe — the OpenCV calls don't release the GIL
    in a useful way, so callers run this in an executor.
    """
    cv2_mod = _try_import_cv2()
    if cv2_mod is None:
        return _fallback("cv2_not_installed")

    try:
        ensure_yunet_model(model_path, expected_sha256=expected_sha256)
    except AssetError as exc:
        logger.warning("YuNet model unavailable: %s", exc)
        return _fallback("yunet_model_missing")

    try:
        detector = cv2_mod.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),  # placeholder; setInputSize() runs per-image
            score_threshold=confidence_threshold,
            nms_threshold=0.3,
            top_k=20,
        )
    except Exception as exc:  # noqa: BLE001 — cv2 raises a variety of types
        logger.warning("FaceDetectorYN.create failed: %s", exc)
        return _fallback("yunet_create_failed")

    timestamps = _evenly_spaced_timestamps(probe.duration_s, count=sample_count)
    if not timestamps:
        return _fallback("zero_duration")

    detected: list[FaceSample] = []
    with tempfile.TemporaryDirectory(prefix="uniq-yunet-") as tmpdir:
        for idx, ts in enumerate(timestamps):
            png = Path(tmpdir) / f"sample_{idx:02d}.png"
            if not _extract_frame(probe.path, ts, png, ffmpeg_bin=ffmpeg_bin):
                continue
            face = _detect_one(cv2_mod, detector, png, ts)
            if face is not None:
                detected.append(face)

    if not detected:
        return _fallback("no_faces_detected")

    centers_x = [s.center_pct()[0] for s in detected]
    centers_y = [s.center_pct()[1] for s in detected]
    return FaceSampling(
        samples=tuple(detected),
        median_center_pct=(_median(centers_x), _median(centers_y)),
        fallback=False,
        reason="ok",
    )
