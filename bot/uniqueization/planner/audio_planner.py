"""Audio-emphasis planner (final_spec §8.8).

Per-beat ``audio_policy`` enum + emphasis-word volume-bump events.
Profile gating:

* ``light``: ``normal`` | ``duck_music_for_dialogue``.
* ``medium``: + ``volume_bump_emphasis_word`` (≤ +2 dB).
* ``heavy``: + ``silence_around_punchline`` (≤ 250 ms).

Hard invariants enforced *only* by the planner output (the worker
adapter / encoder still owns I/EBU R128 limiting):

* ``loudness_target_lufs`` constant **−14.0** LUFS by default.
* ``true_peak_max_dbtp`` ≤ **−1.0** dBTP.
* No per-beat policy may emit ``delta_db`` outside its profile cap.

Pure function. Deterministic. No I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .beat_sheet import Beat
from .locks import StyleLock
from .subtitle_planner import EmphasisWord, SubtitlePlan

__all__ = (
    "AUDIO_POLICY_ENUM",
    "AudioBumpEvent",
    "AudioBeatPolicy",
    "AudioPlan",
    "DEFAULT_LOUDNESS_TARGET_LUFS",
    "DEFAULT_TRUE_PEAK_MAX_DBTP",
    "PROFILE_AUDIO_CAPS",
    "plan_audio",
)


AUDIO_POLICY_ENUM: tuple[str, ...] = (
    "normal",
    "duck_music_for_dialogue",
    "volume_bump_emphasis_word",
    "silence_around_punchline",
)


# §8.8 invariants.
DEFAULT_LOUDNESS_TARGET_LUFS: float = -14.0
DEFAULT_TRUE_PEAK_MAX_DBTP: float = -1.0


@dataclass(frozen=True, slots=True)
class ProfileAudioCaps:
    """§8.8 per-profile cap table row."""

    allowed_policies: frozenset[str]
    bump_cap_db: float
    silence_cap_s: float


PROFILE_AUDIO_CAPS: Mapping[str, ProfileAudioCaps] = {
    "light": ProfileAudioCaps(
        allowed_policies=frozenset({"normal", "duck_music_for_dialogue"}),
        bump_cap_db=0.0,
        silence_cap_s=0.0,
    ),
    "medium": ProfileAudioCaps(
        allowed_policies=frozenset(
            {
                "normal",
                "duck_music_for_dialogue",
                "volume_bump_emphasis_word",
            }
        ),
        bump_cap_db=2.0,
        silence_cap_s=0.0,
    ),
    "heavy": ProfileAudioCaps(
        allowed_policies=frozenset(
            {
                "normal",
                "duck_music_for_dialogue",
                "volume_bump_emphasis_word",
                "silence_around_punchline",
            }
        ),
        bump_cap_db=3.0,
        silence_cap_s=0.25,
    ),
}


# §8.8 base policy by purpose. We bias punchline/reveal beats toward
# emphasis-bumps when the profile allows it; emotional beats lean
# duck-music.
_BASE_POLICY_BY_PURPOSE: Mapping[str, str] = {
    "hook": "volume_bump_emphasis_word",
    "reveal": "volume_bump_emphasis_word",
    "tension": "duck_music_for_dialogue",
    "emotional_hold": "duck_music_for_dialogue",
    "action": "normal",
    "dialogue": "duck_music_for_dialogue",
    "reaction": "duck_music_for_dialogue",
    "transition": "normal",
    "resolution": "duck_music_for_dialogue",
}


@dataclass(frozen=True, slots=True)
class AudioBumpEvent:
    """One §8.8 ``volume_bump_emphasis_word`` event."""

    beat_id: str
    word: str
    start_s: float
    end_s: float
    delta_db: float


@dataclass(frozen=True, slots=True)
class AudioBeatPolicy:
    """One §8.8 per-beat audio decision."""

    beat_id: str
    start_s: float
    end_s: float
    policy: str
    duck_target_db: float = 0.0
    silence_duration_s: float = 0.0
    bump_events: tuple[AudioBumpEvent, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AudioPlan:
    """Container for :func:`plan_audio` output."""

    beat_policies: tuple[AudioBeatPolicy, ...] = field(default_factory=tuple)
    loudness_target_lufs: float = DEFAULT_LOUDNESS_TARGET_LUFS
    true_peak_max_dbtp: float = DEFAULT_TRUE_PEAK_MAX_DBTP
    profile: str = "medium"
    lock_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _resolve_policy(
    purpose: str, *, caps: ProfileAudioCaps
) -> str:
    """Map purpose → policy then downgrade until profile allows it."""
    proposed = _BASE_POLICY_BY_PURPOSE.get(purpose, "normal")
    if proposed in caps.allowed_policies:
        return proposed
    # Downgrade cascade.
    if (
        proposed == "silence_around_punchline"
        and "volume_bump_emphasis_word" in caps.allowed_policies
    ):
        return "volume_bump_emphasis_word"
    if (
        proposed == "volume_bump_emphasis_word"
        and "duck_music_for_dialogue" in caps.allowed_policies
    ):
        return "duck_music_for_dialogue"
    return "normal"


def _clamp_delta_db(proposed: float, cap_db: float) -> float:
    if cap_db <= 0:
        return 0.0
    if proposed > cap_db:
        return cap_db
    if proposed < 0.0:
        return 0.0
    return proposed


def _bump_events_for_beat(
    beat: Beat,
    emphasis_words: tuple[EmphasisWord, ...],
    *,
    cap_db: float,
    default_db: float,
) -> tuple[AudioBumpEvent, ...]:
    if cap_db <= 0 or not emphasis_words:
        return ()
    bumps: list[AudioBumpEvent] = []
    for ew in emphasis_words:
        delta = _clamp_delta_db(default_db, cap_db)
        if delta <= 0.0:
            continue
        bumps.append(
            AudioBumpEvent(
                beat_id=beat.beat_id,
                word=ew.word,
                start_s=ew.start_s,
                end_s=ew.end_s,
                delta_db=delta,
            )
        )
    return tuple(bumps)


def plan_audio(
    beat_sheet: tuple[Beat, ...],
    subtitle_plan: SubtitlePlan | None = None,
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> AudioPlan:
    """Per-beat audio policy plan.

    ``subtitle_plan`` (optional) carries the §8.4 emphasis-word
    selection — when present, the planner can attach
    ``volume_bump_emphasis_word`` events to the *same* words the
    subtitle plan highlighted. Without it, bump events are skipped
    (policy degrades to ``duck_music_for_dialogue`` if available).
    """
    caps = PROFILE_AUDIO_CAPS.get(profile, PROFILE_AUDIO_CAPS["medium"])
    lock_id = lock.lock_id if lock is not None else None
    if not beat_sheet:
        return AudioPlan(
            beat_policies=(),
            profile=profile,
            lock_id=lock_id,
        )

    emphasis_by_beat: dict[str, tuple[EmphasisWord, ...]] = {}
    if subtitle_plan is not None:
        for ev in subtitle_plan.events:
            emphasis_by_beat[ev.beat_id] = ev.emphasis_words

    # Default bump intensity per profile (well under cap so we leave
    # headroom for limiter).
    default_db = caps.bump_cap_db * 0.75

    policies: list[AudioBeatPolicy] = []
    warnings: list[str] = []
    for beat in beat_sheet:
        policy = _resolve_policy(beat.purpose, caps=caps)
        bumps: tuple[AudioBumpEvent, ...] = ()
        silence_s = 0.0
        duck_db = 0.0

        if policy == "volume_bump_emphasis_word":
            words = emphasis_by_beat.get(beat.beat_id, ())
            if words:
                bumps = _bump_events_for_beat(
                    beat,
                    words,
                    cap_db=caps.bump_cap_db,
                    default_db=default_db,
                )
            if not bumps:
                # No emphasis available — downgrade.
                policy = (
                    "duck_music_for_dialogue"
                    if "duck_music_for_dialogue" in caps.allowed_policies
                    else "normal"
                )
                warnings.append(
                    f"{beat.beat_id}:downgraded_bump_to_{policy}"
                )

        if policy == "duck_music_for_dialogue":
            # Conventional duck floor; never below -12 dB.
            duck_db = -6.0

        if policy == "silence_around_punchline":
            silence_s = min(0.20, caps.silence_cap_s)

        # Final cap re-check (paranoid; should be redundant).
        if bumps:
            for b in bumps:
                if b.delta_db > caps.bump_cap_db + 1e-6:
                    warnings.append(
                        f"{beat.beat_id}:bump_clamped_{b.delta_db:.3f}_to_{caps.bump_cap_db}"
                    )

        policies.append(
            AudioBeatPolicy(
                beat_id=beat.beat_id,
                start_s=beat.start_s,
                end_s=beat.end_s,
                policy=policy,
                duck_target_db=duck_db,
                silence_duration_s=silence_s,
                bump_events=bumps,
                reason=f"purpose={beat.purpose};profile={profile}",
            )
        )

    return AudioPlan(
        beat_policies=tuple(policies),
        loudness_target_lufs=DEFAULT_LOUDNESS_TARGET_LUFS,
        true_peak_max_dbtp=DEFAULT_TRUE_PEAK_MAX_DBTP,
        profile=profile,
        lock_id=lock_id,
        warnings=tuple(warnings),
    )
