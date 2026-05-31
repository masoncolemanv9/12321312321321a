"""Asset path resolver for Editor Agent v2.

This module owns the on-disk layout of the package's static assets:

* ``bot/uniqueization/assets/schemas/``: JSON Schemas shipped with the
  package (director_brief, edit_plan). Read-only.
* ``bot/uniqueization/assets/models/``: model files (e.g. YuNet ONNX)
  fetched at runtime. Not committed.

YuNet model fetch is intentionally NOT done here — ``final_spec_FULL.md``
§13.5 forbids bundling heavy assets in the repo. Instead this module
resolves a path (from env or a default location) and verifies SHA-256
when caller supplies an expected hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_ROOT = _PACKAGE_ROOT / "assets"


class AssetError(RuntimeError):
    """Raised when a required asset is missing or fails verification."""


def schemas_dir() -> Path:
    """Absolute path to the package's bundled JSON Schemas dir."""
    return _ASSETS_ROOT / "schemas"


def yunet_model_path(env_override: str = "") -> Path:
    """Resolve the YuNet ONNX model path.

    Precedence:

    1. ``env_override`` (typically ``EDITOR_YUNET_MODEL_PATH``).
    2. Package default at ``assets/models/face_detection_yunet_2023mar.onnx``.

    Existence is NOT checked here; callers fall back to center-reframe
    when the file is missing (§9.13 failure-policy: YuNet unavailable
    → ``warn + center fallback``). Use :func:`ensure_yunet_model` to
    enforce existence + checksum.
    """
    if env_override:
        return Path(env_override).expanduser()
    return _ASSETS_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def ensure_yunet_model(path: Path, expected_sha256: str = "") -> Path:
    """Verify YuNet model exists and (optionally) matches ``expected_sha256``.

    Used by the Part 2 ``face_yunet.py`` module before calling into
    OpenCV. Returns the resolved path on success.

    :raises AssetError: when the file is missing or the digest does not
        match a supplied ``expected_sha256``.
    """
    if not path.exists():
        raise AssetError(
            f"YuNet model not found at {path}. Set EDITOR_YUNET_MODEL_PATH or "
            f"drop the ONNX file into bot/uniqueization/assets/models/."
        )
    if not path.is_file():
        raise AssetError(f"YuNet model path is not a file: {path}")
    if expected_sha256:
        digest = _sha256_of(path)
        if digest.lower() != expected_sha256.lower():
            raise AssetError(
                f"YuNet model SHA-256 mismatch at {path}: "
                f"expected {expected_sha256}, got {digest}"
            )
    return path


def _sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
