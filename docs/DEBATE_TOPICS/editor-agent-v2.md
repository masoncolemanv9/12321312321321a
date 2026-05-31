# Debate Topic: Editor Agent v2 — Uniqueization Pipeline

> **Format:** This is a topic file for the debate-pot tool. The two LLM agents will read this and propose competing refined specs.
>
> **Goal:** Produce a refined design for the next version of the editor worker in Lilush, implementing the **uniqueization pipeline** described in `docs/UNIQUEIZATION_SPEC.md`.

---

## 1. Current state (v1 — what's already shipped)

**File:** `bot/workers/editor.py`

**Behavior:**

1. Receives `(source_path, start_s, end_s, clip_index)` from the analyzer.
2. Single ffmpeg invocation:
   - `-ss start_s -to end_s` extracts the chosen window
   - `crop=ih*9/16:ih` keeps a centered vertical 9:16 strip
   - `scale=1080:1920` normalizes to shorts resolution
   - Optional `overlay` logo (if `OVERLAY_LOGO_PATH` is set)
   - Re-encode with libx264 CRF 23 + AAC 128k
3. Output: `<source_dir>/clips/<job_id>_<clip_index>.mp4`

**Result:** A vertical 1080×1920 short. Pixel-identical to a center-cropped slice of the source. **Trivially detectable** by Content ID, hash matching, perceptual deduplication.

---

## 2. Source spec to refine

**File:** `docs/UNIQUEIZATION_SPEC.md` (must be read in full by both agents).

That spec describes 9 future tentacles to layer on top of v1's vertical crop:

1. `zoom-reframe-agent` — 30-50% zoom + face-aware crop adjustment
2. `effects-overlay-agent` — lights + snow at ~15% opacity
3. `colorgrade-agent` — LUT + particles overlay
4. `music-remover-agent` — silence-detect + mute/replace music
5. `transitions-agent` — cut-point animations (low priority)
6. `ai-upscale-agent` — Real-ESRGAN 2× then re-encode at 1080p
7. `audio-fx-agent` — telephone/vinyl preset at 25%
8. `mirror-segment-agent` — 1.5s horizontal flip in middle of clip
9. (`trim-boring-sections` — extension of analyzer, not editor)

The spec gives a **suggested order** (cheapest → most expensive) and **default parameter values** drawn from the source tutorial.

---

## 3. The debate's job

Refine `docs/UNIQUEIZATION_SPEC.md` into an **executable v2 plan**. The agents must produce concrete, opinionated answers to:

### 3.1. Architecture: monolithic vs split

`UNIQUEIZATION_SPEC.md` proposes **9 separate tentacles** (one ffmpeg pass per tentacle, chained). This is clean modularly but means **9 separate ffmpeg encodes** — each lossy, each adds wall time. Alternative: **1 monolithic ffmpeg filter graph** that does all of zoom + crop + overlays + color grade + audio fx in a single pass.

Trade-offs:
- **Monolithic:** 1 encode → faster, less quality loss, but the filter_complex becomes huge and hard to debug; harder to make individual stages optional.
- **Split:** N encodes → slower, more quality loss (CRF 23 × 4 passes ≈ visible degradation), but each stage is independently testable, debuggable, configurable.
- **Hybrid:** Group stages that can fuse into ffmpeg-monolithic (zoom, crop, overlays, color, audio fx, mirror) into one pass. Keep ML-heavy stages (face detection, music detection, AI upscale) as separate passes since they need their own runtimes.

**Debate must resolve:** monolithic, split, or hybrid? Justify with concrete impact on encode time and visual quality.

### 3.2. Stage ordering

`UNIQUEIZATION_SPEC.md` §6 suggests an order based on cost. But **functional dependencies** matter too:

- AI upscale should be FIRST (operates at higher resolution before downscale, otherwise wasted)
- Vertical crop + zoom should be early (smaller frame = faster downstream)
- Mirror must be AFTER crop (otherwise mirror affects the wrong region)
- Color grade should be LATE (after upscale's color shifts)
- Audio fx independent from video, can run in parallel
- Music removal must be BEFORE audio fx (don't apply telephone filter to silence)

**Debate must resolve:** the canonical ordered stage list, with rationale for each position.

### 3.3. Parameter values

`UNIQUEIZATION_SPEC.md` gives default values from the tutorial (zoom 35%, opacity 0.15, etc). But the tutorial is one anecdote. The agents should debate:

- Are the defaults reasonable? Too aggressive? Too mild?
- Should params be **per-platform** (TikTok tolerates more aggressive uniqueization than YouTube Shorts)?
- Should params be **per-niche** (anime vs talk-show vs gameplay needs different settings)?
- Should params be **adaptive** (e.g. if the source video is already low-res, skip downscale; if there's no music in the audio, skip music removal entirely)?

### 3.4. GPU vs CPU

`ai-upscale-agent` strongly prefers GPU. Without GPU, Real-ESRGAN on CPU is ~30× slowdown — unusable for production.

`mediapipe` face detection works on CPU, fast (~real-time on a 4-core box).

`pyannote-audio` music detection: works on CPU, but slow (~5× real-time). With GPU: real-time.

The user's current deployment is on **a CPU-only Devin VM** (and they're considering Hetzner CX22 which is also CPU-only).

**Debate must resolve:**
- Does v2 ship a CPU-only mode (skip AI upscale, use lighter alternatives) AND a GPU-enabled mode?
- If the user gets a GPU box (e.g. Hetzner with NVIDIA L4), what changes operationally?
- Should the editor publish a recommended hardware spec for "production-realistic throughput"?

### 3.5. Test plan

Uniqueization is hard to unit test — the goal is "looks identical to a human, looks different to an algorithm". The agents should propose:

- **Mocked unit tests** for the orchestration layer (which stages ran, which were skipped, what params were applied) — easy.
- **Integration test** with a known-source 10-second clip, verify the output is valid mp4 1080×1920 with expected duration. Easy.
- **Subjective quality regression** — golden samples on disk, manual diff by user. Acceptable for v1 of v2.
- **Algorithm-detection test** — compute perceptual hash of source vs output, assert distance > threshold. Useful but threshold is arbitrary.
- **End-to-end Content-ID smoke test** — upload to a throwaway test channel, see if YT flags it. Out of scope for CI but worth a manual playbook.

### 3.6. Logging / observability

`UNIQUEIZATION_SPEC.md` §4 mandates an `uniqueization.json` per clip recording every applied technique + parameters. **Debate refinement:**

- Should this log include hash distances (perceptual hash before/after) so we can correlate technique combos with detection-resistance?
- Should it include a sample frame (PNG, low-res thumbnail) so the user can visually audit without playing the full video?
- Should it be append-only across re-runs (audit trail) or overwritten?

### 3.7. User-facing controls

How does the user (or future scheduler) configure which uniqueization stages to apply for a given video?

Options:
- **Global env config** — every video uses the same combo (simplest)
- **Per-`/dl` flags** — `/dl <url> --stages mirror,audio-fx,colorgrade --skip ai-upscale` (most flexible)
- **Profiles** — predefined combos: `--profile light` (mirror + audio-fx), `--profile heavy` (all 9), `--profile no-music` (skip music remover for music-content videos) (best UX)

Debate should pick one and design the CLI surface.

### 3.8. Failure handling

If a single stage fails (e.g. mediapipe crashes on a weird frame), should the rest of the pipeline:
- (a) Skip that stage and continue (degraded output)
- (b) Abort the entire clip
- (c) Retry the stage with simpler parameters
- (d) Substitute a no-op for that stage

Per-stage policy might differ (e.g. AI upscale failure → skip; ffmpeg failure → abort).

### 3.9. MVP scope

Implementing all 9 tentacles is ~2-3 weeks of focused work. The user wants something usable **soon**. What's MVP?

Strawman MVP (debate may revise):
- mirror-segment-agent (low effort, high anti-detect impact)
- audio-fx-agent (low effort, breaks audio fingerprint)
- colorgrade-agent (medium effort, breaks color hash)

That's a 1-week MVP. Skip zoom (face detection deps), skip music removal (heavy deps), skip AI upscale (GPU dep), skip effects overlay (needs user-provided assets).

**Debate must resolve:** what's in MVP, what's v2.1, what's later.

---

## 4. Constraints (must-haves)

- **Backward compatible** with the existing job payload contract — `bot/workers/editor.py` is invoked with the same input dict from the analyzer; output keys must be a superset.
- **Async-safe** — `asyncio.to_thread` for sync deps.
- **Configurable** — env vars for every threshold / model / opacity / strength.
- **Testable** — see §3.5.
- **Cost-aware** — total per-clip compute time on a 4-core CPU box ≤ 5 minutes per minute of source. (i.e., a 60s clip should process in ≤ 5 minutes.)
- **Logging mandate** — `uniqueization.json` with full traceability per §3.6.

---

## 5. Open questions for the debate

(Same as §3 above, framed as questions.)

1. Monolithic / split / hybrid architecture?
2. Canonical stage ordering?
3. Per-platform / per-niche / adaptive parameter tuning?
4. CPU-only fallback for AI-upscale stage?
5. Subjective quality regression test methodology?
6. Logging contents — include perceptual hash + thumbnail?
7. User-facing controls — env / flags / profiles?
8. Per-stage failure policy?
9. MVP scope — which stages first?
10. **Bonus:** Is there any technique from the source tutorial (`UNIQUEIZATION_SPEC.md` §8) that the spec MISSED that should be added?

---

## 6. Lilush context the agents must read

1. `bot/workers/editor.py` — current implementation
2. `bot/workers/analyzer.py` — what the editor receives
3. `bot/workers/publisher.py` — what the editor's output feeds
4. `tests/test_editor.py` — current coverage
5. `docs/UNIQUEIZATION_SPEC.md` — the source spec being refined (§§ 1-9)
6. The `Notes for any future Devin / AI session` block in `UNIQUEIZATION_SPEC.md` §9 (rules for implementing tentacles)

---

## 7. Out of scope

- The uploader (real YT/TikTok/IG upload mechanism). Separate concern, separate spec.
- Octobrowser / AdsPower integration (covered as part of uploader, not editor).
- Niche selection / content discovery / scheduling.
- The SEO worker — covered in the sister topic file `seo-agent-v2.md`.
- The analyzer (whisper / scenedetect parameter tuning) — covered separately if the user requests.

---

## 8. Output expected from the debate

A `final_spec.md` containing:

1. **Architecture diagram** of v2 editor pipeline.
2. **Resolution of all 10 open questions** in §5 with concrete chosen answers + rationale.
3. **MVP scope statement** — exactly which stages are in v2.0, v2.1, v2.2.
4. **Per-stage spec** for the MVP stages — code interface, env vars, dependencies, failure mode, test cases.
5. **Stage execution graph** (DAG) — what runs after what, what runs in parallel.
6. **Hardware recommendation** — CPU box specs for v2.0 MVP, GPU box specs for full v2.
7. **Migration plan** from v1 to v2 — backward compatibility specifics.
8. **Test plan** — unit tests, integration tests, golden-sample regression, optional Content-ID smoke playbook.
9. **Effort estimate** — person-days per stage, total to MVP.
10. **Risk register** with mitigations.

This `final_spec.md` will be committed into Lilush as `docs/EDITOR_AGENT_V2_SPEC.md` and a future implementation Devin will build it stage-by-stage in separate PRs.
