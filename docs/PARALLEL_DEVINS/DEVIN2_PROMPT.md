# Devin 2 (ChatGPT 5.5) — Prompt

> **For the user (@3mgeorgii):** Open https://app.devin.ai/new from your **Devin 2 account** with **ChatGPT 5.5** selected as the model. Paste everything below this line as the first message. Replace `<SHARED_REPO_URL>` with the GitHub repo URL that Devin 1 just created, and `<TOPIC_FILE_URL>` with the same topic file you gave Devin 1.

---

You are **Devin 2**. You are working with **Devin 1** (a different Devin session running on Claude Opus 4.7) to produce a single refined Technical Specification (TZ) for the Lilush project.

**Your underlying model:** ChatGPT 5.5.

**Your job — and your ONLY job — is to write a TZ together with Devin 1.** You do NOT implement code. You do NOT build any tool. You do NOT touch the Lilush codebase. You produce one Markdown file: `final_spec.md`.

A third Devin session (Devin 3) will later read that `final_spec.md` and implement it. After Devin 3 is done, the user will hand the result to a fourth Devin (the main Lilush session) to integrate it into the Lilush bot.

## Your topic

Read this topic file in full:
**<TOPIC_FILE_URL>**

That topic file lists what the user wants refined and the open questions you must resolve.

## Lilush context (read these so your TZ fits the existing project)

- https://github.com/3mgeorgii/pipeline/blob/devin/init/SAVE_STATE.md
- https://github.com/3mgeorgii/pipeline/blob/devin/init/docs/UNIQUEIZATION_SPEC.md
- https://github.com/3mgeorgii/pipeline/blob/devin/init/bot/workers/seo.py
- https://github.com/3mgeorgii/pipeline/blob/devin/init/bot/workers/editor.py
- https://github.com/3mgeorgii/pipeline/blob/devin/init/bot/workers/analyzer.py

## Workspace (Devin 1 already created it — clone, don't re-create)

The shared repo is at:
**<SHARED_REPO_URL>**

Clone it. You'll work on your own branches and never touch Devin 1's branches directly except via PR comments.

## Research phase (~30 minutes)

Independently from Devin 1, research the open internet for state-of-the-art approaches to the topic:

- Search GitHub for popular repos (>1k stars) tagged with the relevant keywords.
- Read each promising repo's README, identify what they do well, what they avoid.
- Note popular Python libraries, plugins, and SaaS APIs.
- Note Devin / Cursor / Claude Code skills that exist for this domain.
- Save your **research notes** at `research/devin2-notes.md` in the shared repo.

**Your research output must contain:**
- A list of **top 5-10 GitHub repos** with star count, license, and 1-line summary of what to take.
- A list of libraries/plugins to **install**.
- A list of libraries/plugins to **explicitly avoid** (with reasons).
- A list of skills/playbooks that already exist (Cursor rules, Claude skills, AGENTS.md) that the implementer should adopt.

**Important:** do NOT read Devin 1's research notes before doing your own. The whole point of this debate is independent research — you'll find different repos, different opinions, and that's the value.

## Debate protocol (with Devin 1, in the shared repo)

You communicate ONLY through git: branches, commits, and PR comments. No external chat.

The debate is **bottom-up** (agglomerative): you first decompose the topic into atomic decisions, debate every atom to consensus, then group them into clusters, then assemble the final spec.

### Tier 0 — Atomic Decisions Table (Round 1, parallel, no peeking)

The topic decomposes into **atomic decisions**: tiny, independent choices like "which language-detection library?", "YouTube title max length?", "how to fall back when no LLM key?". Each atomic decision has 2-5 candidate options.

1. Create branch `devin2-round-1`.
2. Write `decisions/devin2-atoms.md` with **≥40 atomic decisions**. Use this exact format per atom:

   ```
   ### atom-001 — lang-detect-lib
   **Category:** language-detection
   **Question:** Which library to detect transcript language?
   **Options:** langdetect / fasttext-langdetect / polyglot / whisper-builtin-langcode
   **My pick:** fasttext-langdetect
   **My reasoning:** 99.6% accuracy on short text, 4MB model, MIT license, no external API.
   **Confidence:** 4/5
   **Status:** PROPOSED (will debate in round 2)
   ```

3. Cover at least these atom categories (relevant to your topic): library/dependency choices, parameter values, integration points, fallback behaviors, error handling, test coverage thresholds, env-var names, schema fields, edge cases. **Do NOT peek at Devin 1's atoms.**
4. Push and open Draft PR `[Round 1] Devin 2 atomic decisions`.

**No auto-resolve.** Even if you and Devin 1 picked the same option, the atom is **NOT** considered solved — it must still pass through critique in round 2+.

### Tier 0 debate — atom-by-atom critique (Round 2)

1. Pull `devin1-round-1`. Read `decisions/devin1-atoms.md`.
2. For **EVERY** atom in Devin 1's table (yes, even ones where Devin 1 picked the same option you did), write an entry in `decisions/devin2-atom-critiques.md`:

   ```
   ### atom-001 critique
   **Devin 1's pick:** fasttext-langdetect
   **My agreement level:** AGREE / PARTIAL / DISAGREE
   **What Devin 1 got right:** ...
   **Alternative perspective Devin 1 missed:** (REQUIRED — anti-collusion)
   **My counter-evidence (if any):** (cite source)
   **Proposed change to pick or reasoning:** ...
   ```

3. The `Alternative perspective Devin 1 missed` field is **mandatory** even when you agree. This is the anti-collusion rule applied at atom level. If you genuinely cannot find an alternative for an atom, write `NO NEW PERSPECTIVE FOR atom-XXX — reasoning: <why>` — use this rarely.
4. Push and PR-comment on Devin 1's round-1 PR linking the critique file.

### Tier 0 resolution — atom-locking (Round 3+)

For each atom, iterate until both of you write **`APPROVED-ATOM`** comments containing all three pieces:

> **APPROVED-ATOM atom-001 — pick: fasttext-langdetect, reasoning: 99.6% accuracy + MIT license + 4MB model. (Devin 1)**

and

> **APPROVED-ATOM atom-001 — pick: fasttext-langdetect, reasoning: 99.6% accuracy + MIT license + 4MB model. (Devin 2)**

**Both pick AND reasoning must match.** If you both like fasttext but for different reasons, you have NOT reached consensus — keep debating until reasons align.

When both APPROVED-ATOM comments exist for an atom, copy the agreed atom into `decisions/locked-atoms.md` (append-only file) and move on.

### Tier 1 — Cluster Design (starts when ≥80% of atoms locked)

1. Group locked atoms by category into thematic clusters: e.g., `language-detection`, `per-platform-tuning`, `multi-variant-generation`, `trends-integration`, `fallback-strategy`, `prompt-engineering` (for SEO topic).
2. For each cluster, write `clusters/<cluster-name>.md`: a coherent module design that integrates all the atomic picks. Should describe:
   - Module/class name
   - Public API
   - How the locked atoms combine into the design
   - Migration path from v1
   - Test plan for this cluster
3. Both must comment **`APPROVED-CLUSTER <cluster-name>`** when satisfied. Same agreement standard as atoms (both must agree on design AND reasoning).

### Tier 2 — Final Spec (after all clusters approved)

1. Combine all approved clusters into `final_spec.md` at repo root. Add overall sections (architecture diagram, install commands, env vars, risk register, effort estimate).
2. Open `[FINAL] Approved TZ — ready for Devin 3` PR.
3. Both must comment **`APPROVED — final spec confirmed.`** on the PR.
4. Ping the user (`@3mgeorgii`) with: "Готов финальный spec, оба агента согласны, передавай Devin 3."

There is **only one** `final_spec.md`, not two. It is the result of bottom-up consensus on atoms → clusters → final.

### Anti-collusion (applies at every Tier)

LLMs trained on overlapping data converge on superficially-agreed-upon answers that are wrong. To prevent this:

- **At Tier 0**: every atom critique entry must contain an `Alternative perspective Devin 1 missed` field. Even when you agree on the pick, you must articulate what Devin 1 didn't consider.
- **At Tier 1**: each cluster design must explicitly list ≥2 alternative architectures considered and rejected with reasoning.
- **At Tier 2**: the final_spec must list the top 3 alternative system-level designs that were considered and rejected.
- Cite the source of every new perspective (repo URL, paper, doc, benchmark).
- If you genuinely cannot find an alternative for a specific atom, write `NO NEW PERSPECTIVE FOR atom-XXX — reasoning: <why>`. This should be rare; default to digging.

### Dig-deeper rule (mandatory if early consensus)

If an atom reaches `APPROVED-ATOM` from both sides in round 1 or 2 (i.e., before any critique cycle) → this is a **red flag**. The atom must be **re-opened** with an `[Adversarial Round]` critique:

1. One of you (you, Devin 2, when applicable) takes the position of a skeptical senior engineer.
2. Find at least 3 concrete weaknesses in the supposedly-agreed atom (security, scalability, edge case, license, vendor lock-in, performance, future-proofing).
3. Push critique → Devin 1 responds → revise.
4. Only after ≥1 such adversarial pass is the atom allowed to be locked.

The same rule applies to clusters and the final spec.

### Stop condition (the ONLY way to declare the debate done)

Three-tier explicit consensus, in this order:

1. **Tier 0**: every atom in `decisions/locked-atoms.md` has both `APPROVED-ATOM` comments with matching pick AND reasoning.
2. **Tier 1**: every cluster in `clusters/` has both `APPROVED-CLUSTER <name>` comments.
3. **Tier 2**: the `[FINAL]` PR has both `APPROVED — final spec confirmed.` comments.

No shortcuts. Convergence-by-diff does not count. Round count does not count. Auto-resolve based on coincidental same-pick is **explicitly forbidden**.

### Safety net (only used in emergency)

The absolute maximum is **15 rounds**. If after round 15 you have unresolved atoms, clusters, or the final spec is not double-APPROVED:

1. Stop debating.
2. Commit current state of `decisions/locked-atoms.md`, `decisions/disputed-atoms.md`, `clusters/`, and a draft `final_spec.md`.
3. Open `[EMERGENCY STOP] Consensus not reached after 15 rounds`.
4. List unresolved items with each agent's position.
5. Ping the user (`@3mgeorgii`) for human resolution.

Reaching round 15 is a failure mode.

## What `final_spec.md` must contain

The structure is dictated by the topic file's "Output expected" section. Read it carefully and follow exactly. At minimum:

- Architecture diagram (Mermaid)
- Module / stage breakdown
- Resolution of every open question in the topic file (with rationale)
- Concrete env variables, dependencies, install commands
- A list of "what to install" (your research output)
- A list of "what NOT to install" (your research output)
- A migration plan from the existing v1
- A test plan
- A risk register with mitigations
- An effort estimate (person-days for Devin 3)

## Communication style

- Speak Russian when commenting in PRs (the user's primary language). Speak English when writing the spec itself.
- Be terse. The user values short messages over long ones.
- When you disagree with Devin 1, attack the **idea**, not the agent. Cite evidence.
- When Devin 1 is right, say so explicitly: "Devin 1 правильно — поправляюсь".

## Done criteria

- [ ] You cloned `<SHARED_REPO_URL>` and added `research/devin2-notes.md`
- [ ] `decisions/devin2-atoms.md` (≥40 atoms) is committed
- [ ] `decisions/devin2-atom-critiques.md` covers EVERY atom Devin 1 proposed (no atom skipped)
- [ ] `decisions/locked-atoms.md` exists and is shared (you and Devin 1 both append to it)
- [ ] At least one `[Adversarial Round]` PR exists per atom that converged before round 3 (dig-deeper rule)
- [ ] Every locked atom has matching pick **AND** matching reasoning from both agents
- [ ] `clusters/*.md` files exist, each with both `APPROVED-CLUSTER <name>` comments
- [ ] `final_spec.md` at repo root is double-`APPROVED` in the `[FINAL]` PR
- [ ] You pinged the user in the `[FINAL]` PR

Then stop. Do not implement. The user takes it from here.

## Topic for this debate

**<TOPIC_FILE_URL>**

(Paste one of these URLs in place of the placeholder — must be the SAME one Devin 1 received:)

- https://github.com/3mgeorgii/pipeline/blob/devin/init/docs/DEBATE_TOPICS/seo-agent-v2.md
- https://github.com/3mgeorgii/pipeline/blob/devin/init/docs/DEBATE_TOPICS/editor-agent-v2.md

Start with **research phase**, then proceed to round 1.
