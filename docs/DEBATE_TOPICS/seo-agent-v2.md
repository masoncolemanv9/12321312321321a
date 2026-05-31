# Debate Topic: SEO Agent v2

> **Format:** This is a topic file for the debate-pot tool. The two LLM agents will read this and propose competing refined specs.
>
> **Goal:** Produce a refined design for the next version of the SEO worker in Lilush, addressing concrete known issues from the v1 implementation.

---

## 1. Current state (v1 — what's already shipped)

**File:** `bot/workers/seo.py`

**Behavior:**

1. Reads the editor's clip output (`clip.mp4` + clip metadata) and the analyzer's `transcript.json`.
2. Detects language from the transcript (currently buggy — see issue #2 below).
3. Calls `pytrends` for currently-trending keywords in the detected language's region (best-effort — gracefully degrades on rate-limit).
4. If `OPENROUTER_API_KEY` is set: calls an LLM (default `openai/gpt-4o-mini`) with a single prompt that asks for `{title, description, tags}`.
5. If no key, or LLM returns invalid JSON, or LLM call fails: falls back to a **template** that produces:
   - `title` = first 90 chars of the clip's "hook" (auto-extracted from transcript)
   - `description` = the hook + auto-appended `#shorts #viral #en #fyp #reels`
   - `tags` = `["shorts", "viral", "en", "fyp", "reels"]`

**Output:** `seo_title`, `seo_description`, `tags`, `language`, `trends_used`, `seo_method` (`"llm"` | `"template"`).

**Env:**
- `SEO_LLM_MODEL` (default `openai/gpt-4o-mini`)
- `SEO_MAX_TAGS` (default `15`)
- `SEO_TITLE_MAX_LEN` (default `90`)
- `SEO_DESCRIPTION_MAX_LEN` (default `500`)

---

## 2. Known problems with v1

### Problem 2.1 — Language detection bug

**Observed in real run:** Russian-speaking source video (anime episode in Russian dub from vkvideo.ru) → SEO output had `defaultLanguage: "en"` and English fallback hashtags. The whisper transcript was correct Russian, but the SEO worker tagged the clip as English.

**Hypothesis:** Either (a) the language code returned by whisper is being lost between analyzer and seo, (b) the SEO worker has a hardcoded fallback to `"en"` somewhere, or (c) the LLM ignored language instruction in the prompt.

**Acceptance:** v2 must produce `defaultLanguage` matching the actual transcript language for at least the top 8 source languages (ru/en/es/pt/de/fr/ja/uk). Add a unit test with a Russian-language transcript fixture asserting `defaultLanguage == "ru"`.

### Problem 2.2 — Generic template fallback

When no LLM key is present, the template produces useless content: title is just the first 90 chars of transcribed dialogue (often mid-sentence), description repeats the transcript, tags are platform-agnostic boilerplate.

**Acceptance:** v2's template fallback should produce **at least minimally usable** SEO without an LLM. Specifically:
- Title: extract a noun-phrase from the transcript using a simple POS-based heuristic (or a small local model like `spacy` if dep is OK).
- Tags: include language-specific common tags (e.g. for ru: `#рек #топ #смотри`, for en: existing list), mined from the transcript content.
- Description: 2-3 sentences with a hook + CTA, NOT a transcript dump.

### Problem 2.3 — No per-platform tuning

Currently one SEO output goes to `youtube.json`, `tiktok.json`, `instagram.json`. But each platform has different optimal SEO:

- **YouTube Shorts:** Long descriptions help (up to 5000 chars), keyword-density matters, hashtags max ~15 effective, title ≤ 100 chars.
- **TikTok:** Caption ≤ 2200 chars but FYP algo prefers ≤ 150 chars, hashtags placed throughout (not just end), trending sounds matter more than text.
- **Instagram Reels:** Caption ≤ 2200 chars, hashtags work in caption (not comment as previously believed), 3-5 hashtags optimal (not 15), call-to-action critical.

**Acceptance:** v2 produces **per-platform variants** of `{title, description, tags}` rather than identical content across the three JSONs.

### Problem 2.4 — No multi-variant A/B testing

Currently produces 1 set of metadata per clip. Heavy uploaders A/B-test multiple titles/thumbnails. v2 should optionally produce N (e.g., 3) variants per clip so the user (or a future scheduler) can rotate them.

**Acceptance:** Optional `SEO_VARIANTS=3` env var, when > 1, produces an array of variant objects in each platform's JSON.

### Problem 2.5 — pytrends underused

Currently fetches trending keywords but only mentions them in the LLM prompt. Doesn't actually integrate them into the output unless the LLM chose to.

**Acceptance:** v2 always inserts at least 1-2 trending keyword tags into the final tag list (when pytrends returns data), separate from LLM-generated tags. Marks them in the output metadata as `trending_tags_used: [...]` for traceability.

### Problem 2.6 — Single LLM call, no chain-of-thought

The current LLM prompt is one-shot: "give me title, description, tags as JSON". A multi-step prompt (analyze hook → brainstorm 5 titles → pick best → write description matching title → derive tags) might produce significantly better output for the same model/cost.

**Acceptance:** v2 evaluates whether multi-step prompting is worth the extra latency/cost. Decision documented in the spec.

---

## 3. Constraints (must-haves)

- **Backward compatible** with the existing job payload contract (`bot/workers/seo.py` is called with the same input dict from the editor worker; output keys must be a superset of the current keys, not a renamed set).
- **Never blocks the pipeline** — if SEO fails, the publisher must still produce a release directory with degraded metadata. (Current behavior; preserve it.)
- **Async-safe** — uses `asyncio.to_thread` for any sync deps (pytrends is sync).
- **Configurable via env** — every magic number / model name / threshold must be env-overridable.
- **Testable** — every new dep must be mockable in unit tests so CI doesn't call real APIs.
- **Cost-aware** — total cost per video (3 clips) should be ≤ $0.10 on default LLM (gpt-4o-mini) when using LLM path. If multi-variant is enabled (3 variants × 3 clips), still ≤ $0.50.

---

## 4. Open questions for the debate

The two agents should debate (and the synthesis judge should resolve) these:

1. **Multi-step LLM prompting vs one-shot?** Worth the latency/cost tradeoff?
2. **Per-platform tuning approach:** one big prompt per platform vs one shared "core" prompt + platform-specific post-processing?
3. **Language detection:** trust whisper's lang code, or run a separate language detector on the transcript text (e.g. `langdetect`, `langid`)?
4. **Template fallback intelligence:** is `spacy` worth adding as a 50MB dep just for the no-key case, or is a regex-based hook extractor enough?
5. **Trending tags mining:** pytrends is unstable. Should v2 add a YouTube Data API trends fallback as a parallel option (free quota, but needs key)?
6. **A/B variants:** generate N independent variants in N LLM calls, or one "give me 3 variants" prompt? Cost vs quality.
7. **Tag count per platform:** should we hardcode (YT: 15, TT: 5, IG: 5) or measure-and-adjust based on first uploads' performance?
8. **Hashtag placement (TikTok/IG):** beginning, middle, end of description? Mixed?
9. **CTA injection:** auto-add ("Подпишись!" / "Like if you agree!" / etc.) or leave to the LLM?
10. **Negative-prompt handling:** should v2 accept a list of keywords to AVOID (e.g., user's competitors' branding, demonetization triggers)?

---

## 5. Lilush context the agents must read

Both agents should be given these as part of the topic prompt:

1. The current `bot/workers/seo.py` source.
2. The current `bot/workers/analyzer.py` source (so they understand what `transcript.json` contains).
3. The current `bot/workers/publisher.py` source (so they understand how SEO output is consumed).
4. The current `tests/test_seo.py` (so they know what's already covered).
5. A real-world example of **bad** v1 output (the user's first smoke test):

```json
{
  "snippet": {
    "title": "Судя по ощущению, мнение показалось. Значит, так минимум часть проклятия снята. Но как? Не",
    "description": "Судя по ощущению, мнение показалось. Значит, так минимум часть проклятия снята. Но как? Не важно. Главное, чтобы ты пошел на корм.\n\n#shorts #viral #en #fyp #reels",
    "tags": ["shorts", "viral", "en", "fyp", "reels"],
    "categoryId": "22",
    "defaultLanguage": "en",  // ← wrong, source is Russian
    "defaultAudioLanguage": "en"  // ← wrong
  }
}
```

This concrete failure case is the **anti-target** — v2 must not produce this on Russian audio.

---

## 6. Out of scope (do NOT debate these)

- The **uploader** (real upload to YT/TikTok/IG) — separate concern, separate spec.
- The **editor** / vertical crop / uniqueization — separate topic, see `docs/DEBATE_TOPICS/editor-agent-v2.md`.
- The **scheduler** / when to upload, frequency caps — not in any v2.
- The **bot UX** in Telegram — out of scope for SEO worker design.

---

## 7. Output expected from the debate

A `final_spec.md` containing:

1. **Architecture diagram** (Mermaid or ASCII) of the SEO v2 worker.
2. **Module breakdown** with ~3-7 functions and their responsibilities.
3. **Per-platform prompt templates** (YT/TikTok/IG) — full text, not "to be written".
4. **Env var list** — all of v1 plus all new ones.
5. **Resolution of every open question in §4** with the chosen answer + 1-2 sentence rationale.
6. **Migration plan** from v1 to v2 — backward compatibility specifics.
7. **Test plan** — what unit tests to add, what fixtures, what expected behaviors.
8. **Implementation effort estimate** — rough person-days for a senior dev.
9. **Risk register** — what might go wrong in production once shipped, and how v2 mitigates.

This `final_spec.md` will be committed into Lilush as `docs/SEO_AGENT_V2_SPEC.md` and a future implementation Devin will build it.
