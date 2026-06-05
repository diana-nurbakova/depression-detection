


# Task 2 — LLM Prompts (Verbatim Extraction)

> **Scope.** This document contains the exact strings sent to `meta-llama/Llama-3.3-70B-Instruct` (via HuggingFace InferenceClient) when building the Theory-of-Mind training features for eRisk 2026 Task 2. They are reproduced as-rendered (i.e., after Python triple-quoted-string concatenation, after `\` line-continuations are collapsed, and after `.format()` substitution of the symptom block).
>
> **Source files** (single source of truth):
> - System / user templates: [src/erisk_task2/tom/prompts.py](../src/erisk_task2/tom/prompts.py)
> - Symptom Variant C generation: [src/erisk_task2/tom/prompts.py:22-34](../src/erisk_task2/tom/prompts.py#L22-L34) and [src/erisk_task2/features/layer1.py:48-71](../src/erisk_task2/features/layer1.py#L48-L71)
> - Call sites + pre-format: [src/erisk_task2/tom/tom_module.py:60-62, 155-204](../src/erisk_task2/tom/tom_module.py#L60-L204)
> - Thread transcript format: [src/erisk_task2/formatting/thread_formatter.py](../src/erisk_task2/formatting/thread_formatter.py)
>
> **What was *not* used at submission time.** The live-pipeline run defaults to embedding-based ToM Option A, so none of these prompts hit the LLM during the actual eRisk submission. They were used only during the **Colab training-feature build** to compute `tom_features.npz` (see §6 of [task2_solution_description.md](task2_solution_description.md) on the train/test mismatch).
>
> **Line endings.** The source uses LF (`\n`). Python `"""\` at the start of a triple-quoted string suppresses the first newline; a trailing `\` before the closing `"""` likewise suppresses the final newline. Mid-string `\` line-continuations collapse `\<LF>` into a single line. The strings reproduced below are the *post-collapse* text. The `{{`/`}}` escapes used in the Python source decode to literal `{`/`}` (Python `str.format` convention).
>
> **No few-shot examples and no chain-of-thought primer.** None of the prompts contain few-shot examples, assistant-turn primers, role-play preambles, or step-by-step / "think before answering" instructions. The only output discipline is the directive **`Respond with valid JSON only.`** at the end of the first sentence of each system prompt. JSON parsing is done downstream by `_extract_json()` in both `llm_client.py` and `hf_client.py`, which tolerates ```` ```json ```` code fences and locates the outermost `{ ... }` block; on parse failure the call is logged and the corresponding 21-d sub-vector is left at zero — there is no second LLM call with adjusted temperature or instructions.

---

## 1. BDI-II Variant C — Symptom Definitions (inlined into every system prompt)

Generated at module-load time by joining 21 lines with `"\n"`. Texts come from `SYMPTOM_DESCRIPTIONS` in [layer1.py:49-71](../src/erisk_task2/features/layer1.py#L49-L71). The block is substituted into every prompt's `{symptom_definitions}` placeholder via `str.format` once, at `ToMModule.__init__` time, so the bytes are identical across all LLM calls (this is what enables Ollama KV-cache hits in the local-backend variant).

```
1-Sadness: Persistent feelings of sorrow, unhappiness, or emotional pain.
2-Pessimism: Discouragement and hopelessness about the future.
3-Past failure: Feeling like a failure, seeing many disappointments.
4-Loss of pleasure: Reduced enjoyment of activities, hobbies, social life.
5-Guilty feelings: Excessive guilt, self-blame for bad things.
6-Punishment feelings: Expectation of punishment, sense that bad things are deserved.
7-Self-dislike: Self-criticism, disappointment in oneself as a person.
8-Self-criticalness: Harsh self-judgment for all faults and mistakes.
9-Suicidal thoughts: Thoughts of ending one's life, death wishes.
10-Crying: Increased tearfulness, uncontrollable emotional outbursts.
11-Agitation: Restlessness, irritability, inability to stay still or relax.
12-Loss of interest: Social withdrawal, apathy, not caring about things.
13-Indecisiveness: Difficulty making decisions, putting off choices.
14-Worthlessness: Profound sense of having no value, being useless.
15-Loss of energy: Fatigue, everything takes extra effort.
16-Sleep changes: Insomnia, oversleeping, or disrupted sleep patterns.
17-Irritability: Short temper, easily frustrated or angered.
18-Appetite changes: Eating much more or less, weight gain or loss.
19-Concentration difficulty: Brain fog, difficulty focusing, forgetfulness.
20-Tiredness/fatigue: Constant exhaustion, lack of motivation due to tiredness.
21-Loss of interest in sex: Reduced libido, no sexual desire.
```

(Variant A — names only without descriptions — is also defined in the codebase but is not the default; `config.task2.yaml:17` sets `variant: "C"`.)

---

## 2. Prompt 1 — Self-View System Prompt (Option C)

**Used when.** `ToMModule._assess_llm` is called and the target user has at least one own text in the thread (`thread.has_target_text == True`). Source: [tom_module.py:163-172](../src/erisk_task2/tom/tom_module.py#L163-L172).

**System message (verbatim, post-collapse, post-`{symptom_definitions}`-substitution):**

```text
You are a clinical psychologist analyzing social media text for indicators of depression. Assess the user's mental state based SOLELY on their own writings. Report only symptoms where you find textual evidence. Respond with valid JSON only.

BDI-II Symptom Definitions:
1-Sadness: Persistent feelings of sorrow, unhappiness, or emotional pain.
2-Pessimism: Discouragement and hopelessness about the future.
3-Past failure: Feeling like a failure, seeing many disappointments.
4-Loss of pleasure: Reduced enjoyment of activities, hobbies, social life.
5-Guilty feelings: Excessive guilt, self-blame for bad things.
6-Punishment feelings: Expectation of punishment, sense that bad things are deserved.
7-Self-dislike: Self-criticism, disappointment in oneself as a person.
8-Self-criticalness: Harsh self-judgment for all faults and mistakes.
9-Suicidal thoughts: Thoughts of ending one's life, death wishes.
10-Crying: Increased tearfulness, uncontrollable emotional outbursts.
11-Agitation: Restlessness, irritability, inability to stay still or relax.
12-Loss of interest: Social withdrawal, apathy, not caring about things.
13-Indecisiveness: Difficulty making decisions, putting off choices.
14-Worthlessness: Profound sense of having no value, being useless.
15-Loss of energy: Fatigue, everything takes extra effort.
16-Sleep changes: Insomnia, oversleeping, or disrupted sleep patterns.
17-Irritability: Short temper, easily frustrated or angered.
18-Appetite changes: Eating much more or less, weight gain or loss.
19-Concentration difficulty: Brain fog, difficulty focusing, forgetfulness.
20-Tiredness/fatigue: Constant exhaustion, lack of motivation due to tiredness.
21-Loss of interest in sex: Reduced libido, no sexual desire.

Output format:
{
  "active_symptoms": {
    "<symptom_name>": {"score": <1-3>, "evidence": "<brief quote or paraphrase>"},
    ...
  },
  "depression_probability": <0.0-1.0>,
  "overall_impression": "<1-2 sentences>"
}

Severity scale: 1=mild/possible, 2=moderate/clear, 3=severe/strong.
If no indicators found, return:
{"active_symptoms": {}, "depression_probability": 0.0, "overall_impression": "No indicators observed."}
```

**User message template (verbatim, before `.format` substitution):**

```text
The following texts were written by a single user in an online discussion thread:

{target_user_texts}
```

`{target_user_texts}` is built at call time as `"\n\n".join(thread.target_texts)` — i.e., every text the target wrote in the current thread, joined by blank lines, in chronological order ([tom_module.py:164-165](../src/erisk_task2/tom/tom_module.py#L164-L165)).

**Expected JSON output schema (as written into the system prompt):**

| Field | Type | Range / Allowed values |
| --- | --- | --- |
| `active_symptoms` | object (dict) | keys = symptom names (from the 21 above); values = `{ "score": int, "evidence": str }` |
| `active_symptoms.<name>.score` | integer | **1 / 2 / 3** (1 = mild / possible, 2 = moderate / clear, 3 = severe / strong) |
| `active_symptoms.<name>.evidence` | string | brief quote or paraphrase from the target's own text |
| `depression_probability` | float | **0.0 – 1.0** |
| `overall_impression` | string | 1–2 sentences |

**Empty-finding fallback (specified inline in the prompt):**

```json
{"active_symptoms": {}, "depression_probability": 0.0, "overall_impression": "No indicators observed."}
```

**Chain-of-thought.** No explicit reasoning section is requested; the prompt asks for **valid JSON only**, suppressing CoT in the response. There is no "think step-by-step" or "reasoning:" field.

---

## 3. Prompt 2a — Observer-View System Prompt, Independent (Option C, default)

**Used when.** `chained=False` (the default — `config.task2.yaml:45`, [tom_module.py:188-193](../src/erisk_task2/tom/tom_module.py#L188-L193)). Also the only observer prompt used when the target had no own text (silent thread).

**System message (verbatim):**

```text
You are a clinical psychologist observing an online discussion. Assess how OTHER PEOPLE in the conversation perceive and respond to a specific target user marked with [TARGET]. Focus on what the community's reactions reveal about the target user's mental state. Respond with valid JSON only.

BDI-II Symptom Definitions:
1-Sadness: Persistent feelings of sorrow, unhappiness, or emotional pain.
2-Pessimism: Discouragement and hopelessness about the future.
3-Past failure: Feeling like a failure, seeing many disappointments.
4-Loss of pleasure: Reduced enjoyment of activities, hobbies, social life.
5-Guilty feelings: Excessive guilt, self-blame for bad things.
6-Punishment feelings: Expectation of punishment, sense that bad things are deserved.
7-Self-dislike: Self-criticism, disappointment in oneself as a person.
8-Self-criticalness: Harsh self-judgment for all faults and mistakes.
9-Suicidal thoughts: Thoughts of ending one's life, death wishes.
10-Crying: Increased tearfulness, uncontrollable emotional outbursts.
11-Agitation: Restlessness, irritability, inability to stay still or relax.
12-Loss of interest: Social withdrawal, apathy, not caring about things.
13-Indecisiveness: Difficulty making decisions, putting off choices.
14-Worthlessness: Profound sense of having no value, being useless.
15-Loss of energy: Fatigue, everything takes extra effort.
16-Sleep changes: Insomnia, oversleeping, or disrupted sleep patterns.
17-Irritability: Short temper, easily frustrated or angered.
18-Appetite changes: Eating much more or less, weight gain or loss.
19-Concentration difficulty: Brain fog, difficulty focusing, forgetfulness.
20-Tiredness/fatigue: Constant exhaustion, lack of motivation due to tiredness.
21-Loss of interest in sex: Reduced libido, no sexual desire.

Output format:
{
  "perceived_symptoms": {
    "<symptom_name>": {"score": <1-3>, "observer_signal": "<what in others' responses suggests this>"},
    ...
  },
  "observer_concern_level": <0-3>,
  "community_response_type": "<concern|support|advice|normalization|casual|mixed>",
  "depression_probability": <0.0-1.0>,
  "key_observation": "<1-2 sentences>"
}
```

**User message template (verbatim):**

```text
{formatted_thread}
```

`{formatted_thread}` is the priority-truncated transcript built by `format_thread()` — see §5 below. The thread carries the `[TARGET]` markers the system prompt asks the model to attend to.

**Expected JSON output schema:**

| Field | Type | Range / Allowed values |
| --- | --- | --- |
| `perceived_symptoms` | object (dict) | keys = symptom names; values = `{ "score": int, "observer_signal": str }` |
| `perceived_symptoms.<name>.score` | integer | **1 / 2 / 3** (severity inferred from observer behaviour, same scale as self-view) |
| `perceived_symptoms.<name>.observer_signal` | string | what specifically in others' responses suggests this symptom |
| `observer_concern_level` | integer | **0 – 3** (downstream code normalises by /3 → ToM slot 45) |
| `community_response_type` | string | one of: `concern`, `support`, `advice`, `normalization`, `casual`, `mixed` (downstream encoding: `concern→1.0, support→0.8, advice→0.6, mixed→0.5, normalization→0.3, casual→0.0` → ToM slot 46) |
| `depression_probability` | float | **0.0 – 1.0** |
| `key_observation` | string | 1–2 sentences |

**Chain-of-thought.** None requested. "Respond with valid JSON only." in the opening paragraph.

---

## 4. Prompt 2b — Observer-View System Prompt, Chained (defined but not used at the default config)

**Used when.** `chained=True` and the self-view call already succeeded. The default config sets `chained: false`, so 2b never fired in the training-feature build. Included here for completeness because it is what `extract_tom_features` would parse if the chained variant were re-enabled.

**System message (verbatim):**

```text
You are a clinical psychologist comparing two perspectives on a social media user: how they present themselves versus how others perceive them. The user is marked with [TARGET]. Respond with valid JSON only.

BDI-II Symptom Definitions:
1-Sadness: Persistent feelings of sorrow, unhappiness, or emotional pain.
2-Pessimism: Discouragement and hopelessness about the future.
3-Past failure: Feeling like a failure, seeing many disappointments.
4-Loss of pleasure: Reduced enjoyment of activities, hobbies, social life.
5-Guilty feelings: Excessive guilt, self-blame for bad things.
6-Punishment feelings: Expectation of punishment, sense that bad things are deserved.
7-Self-dislike: Self-criticism, disappointment in oneself as a person.
8-Self-criticalness: Harsh self-judgment for all faults and mistakes.
9-Suicidal thoughts: Thoughts of ending one's life, death wishes.
10-Crying: Increased tearfulness, uncontrollable emotional outbursts.
11-Agitation: Restlessness, irritability, inability to stay still or relax.
12-Loss of interest: Social withdrawal, apathy, not caring about things.
13-Indecisiveness: Difficulty making decisions, putting off choices.
14-Worthlessness: Profound sense of having no value, being useless.
15-Loss of energy: Fatigue, everything takes extra effort.
16-Sleep changes: Insomnia, oversleeping, or disrupted sleep patterns.
17-Irritability: Short temper, easily frustrated or angered.
18-Appetite changes: Eating much more or less, weight gain or loss.
19-Concentration difficulty: Brain fog, difficulty focusing, forgetfulness.
20-Tiredness/fatigue: Constant exhaustion, lack of motivation due to tiredness.
21-Loss of interest in sex: Reduced libido, no sexual desire.

Output format:
{
  "perceived_symptoms": {
    "<symptom_name>": {"score": <1-3>, "observer_signal": "<evidence>"},
    ...
  },
  "observer_concern_level": <0-3>,
  "community_response_type": "<concern|support|advice|normalization|casual|mixed>",
  "depression_probability": <0.0-1.0>,
  "perspective_gap": {
    "self_higher": ["<symptoms user rates higher than observers>"],
    "observer_higher": ["<symptoms observers detect but user doesn't express>"],
    "alignment": "<aligned|user_minimizes|user_exaggerates|mixed>"
  },
  "insight_assessment": "<1-2 sentences>"
}
```

**User message template (verbatim):**

```text
A previous assessment of this user's OWN writings found:
{self_view_json}

Full conversation thread:
{formatted_thread}
```

`{self_view_json}` is the raw JSON of Prompt 1's parsed response, re-serialised via `json.dumps(self_view)` at call time ([tom_module.py:180-185](../src/erisk_task2/tom/tom_module.py#L180-L185)).

---

## 5. Thread Transcript Schema (`format_thread`)

Not a prompt per se, but it is the entire `user` message for Prompts 2a and 2b. Reproducing the exact wrapping ensures inputs can be reconstructed. Source: [thread_formatter.py:43-150](../src/erisk_task2/formatting/thread_formatter.py#L43-L150).

### 5.1 Token budget and char-conversion heuristic

| Parameter | Value | Source |
| --- | --- | --- |
| `max_tokens` | **2000** | [config/task2.yaml:67](../config/task2.yaml#L67); default in `format_thread(max_tokens=2000)` |
| `CHARS_PER_TOKEN` | **4** (conservative) | [thread_formatter.py:19](../src/erisk_task2/formatting/thread_formatter.py#L19) |
| Effective char budget | `2000 × 4 = 8000` chars, minus header (`=== THREAD …`) and footer (`=== END THREAD ===`) | [thread_formatter.py:49, 101](../src/erisk_task2/formatting/thread_formatter.py#L49) |
| `truncate_chars` (P3 cap) | **100** | [config/task2.yaml:68](../config/task2.yaml#L68); default in `format_thread(truncate_chars=100)` |

### 5.2 Priority classes and truncation rules

| Priority | Membership | Treatment when budget allows | Treatment when budget exhausted |
| --- | --- | --- | --- |
| **P1** | Target's own posts/comments **and** comments whose `parent_id` is the target's submission or any target comment (= direct replies to target) | Full text. | If even P1 overflows, the offending P1 line is character-truncated to `budget - used` chars (only if `> 50` chars remain) with a trailing `"..."`, and iteration stops. |
| **P2** | Other nodes inside the same reply branch that contains the target (i.e., upstream/sibling posts on branches reaching the target) | Full text. | Omitted entirely; remaining P2 count is added to `posts_omitted`. |
| **P3** | Non-target posts whose parent is in a target-branch (one hop further out) | Truncated to first `truncate_chars = 100` chars + `"... [+{N} more chars]"` suffix. Posts shorter than 100 chars are kept full. | Omitted entirely. |
| **P4** | Posts in branches that have no target participation at all | **Always omitted.** | (same) |

The priority classifier walks the **chronologically flattened** node list (submission first, then comments sorted by `created_utc`). Submissions outside the target branch are skipped at the loop level — only their title appears in the header.

### 5.3 Per-node format (`_format_node`)

Each kept line is one of:

```text
[POST] <author>[TARGET]: <body>
```

(when the node is the submission; the `[TARGET]` suffix is present iff the submission author is the target).

```text
[REPLY to POST by <parent_author>[TARGET]] <author>[TARGET]: <body>
```

(when the node is a direct comment on the submission; trailing `\n` is appended to each line).

```text
[REPLY to <parent_author>[TARGET]] <author>[TARGET]: <body>
```

(when the node is a reply to another comment; the `[TARGET]` markers on either side are present only when that party is the target).

```text
[COMMENT] <author>[TARGET]: <body>
```

(orphan: parent not present in this thread; rare).

If `node.body` is empty, the body becomes the literal placeholder `[no text]`. Each formatted line ends with a newline.

### 5.4 Envelope (header / footer)

```text
=== THREAD (Round <N>) ===
Title: <thread.title  or "[no title]">

<P1 lines>
<P2 lines>
<P3 lines>

=== END THREAD ===
```

`<N>` is `thread.round_number` (0-indexed round).

### 5.5 Silent-thread short-circuit

When `thread.has_target_text` is False (target did not contribute any post or comment), the formatter bypasses the priority loop and emits exactly:

```text
=== THREAD (Round <N>) ===
Title: <thread.title  or "[no title]">

[TARGET did not contribute text in this thread]
=== END THREAD ===
```

In this case Prompt 1 (self-view) is **not** sent — `_assess_llm` skips it and sets `self_view = None` — and only Prompt 2a is issued, using the placeholder transcript above. See [tom_module.py:161-176](../src/erisk_task2/tom/tom_module.py#L161-L176).

---

## 6. Few-Shot Examples and Reasoning Primers

**None.** No few-shot examples, no `assistant:` priming turns, and no chain-of-thought instructions appear in any of the four prompts (Prompts 1, 2a, 2b, 4) defined in [prompts.py](../src/erisk_task2/tom/prompts.py). The only output-shaping mechanism is:

1. The directive **"Respond with valid JSON only."** in the opening sentence of every system prompt.
2. An inline `Output format:` JSON template within the system prompt, with field names and value-range placeholders (e.g., `<0.0-1.0>`, `<1-3>`).
3. For Prompt 1, an explicit empty-finding fallback string the model is told to return verbatim when no indicators are found.
4. Downstream tolerance: `_extract_json()` ([llm_client.py:115-146](../src/erisk_task2/tom/llm_client.py#L115-L146), [hf_client.py:118-149](../src/erisk_task2/tom/hf_client.py#L118-L149)) tries direct parse → markdown-fenced JSON → outermost `{...}` substring, in that order.

If a model emits reasoning before the JSON, `_extract_json`'s third fallback (find `{`...`}` boundaries) typically still recovers the object. If it fails, the call is logged at WARNING level and the corresponding 21-d block is left at its default zero value. There is **no retry with adjusted instructions or lowered temperature** — see [docs/task2_solution_description.md §6.2](task2_solution_description.md#62-option-c-llm-based-dual-mentalizing-used-to-build-the-training-features).

---

## 7. Reference: Prompt 4 (Response-Category Classifier, Option B only)

For completeness — Prompt 4 is only invoked under ToM `method="option_b"` (response-category classification), which is implemented but not selected by the default config and was not used in the training-feature build.

**System message (verbatim):**

```text
Classify the reply into exactly one category reflecting what it reveals about the replier's perception of the original poster. Respond with the category label only, nothing else.

Categories:
CONCERN - Expresses worry about the user's wellbeing
ADVICE - Suggests help, therapy, or coping strategies
EMOTIONAL_SUPPORT - Empathy, validation, comfort
NORMALIZATION - Suggests the situation is normal
SHARED_EXPERIENCE - Relates personal similar experience
PRACTICAL_SUPPORT - Offers specific resources
CASUAL - No mental health signal
```

**User message template:**

```text
Target user wrote: "{target_text}"
Someone replied: "{reply_text}"
```

`{target_text}` and `{reply_text}` are each truncated to 200 chars before substitution ([tom_module.py:131-138](../src/erisk_task2/tom/tom_module.py#L131-L138)). Output is plain text (no JSON), matched case-insensitively against the 7 category labels.

---

## 8. Reference: Prompt 3 (Severity Assessor, defined but unused)

Prompt 3 (BDI-II total-score estimator) is defined in [prompts.py:148-177](../src/erisk_task2/tom/prompts.py#L148-L177) but no call site in the Task 2 pipeline references it. Reproduced for completeness.

**System message (verbatim):**

```text
You are a depression screening tool. Given a user's accumulated social media posts, estimate depression severity on the BDI-II scale (0-63). Be calibrated: most social media users are NOT depressed (base rate ~10%). Respond with valid JSON only.

BDI-II severity categories:
  Minimal: 0-13
  Mild: 14-19
  Moderate: 20-28
  Severe: 29-63

{symptom_definitions}

Output format:
{
  "bdi_total": <0-63>,
  "severity": "<minimal|mild|moderate|severe>",
  "trajectory": "<stable|worsening|improving|fluctuating>",
  "active_symptoms": ["<symptom1>", "<symptom2>", ...],
  "confidence": <0.0-1.0>
}
```

**User message template:**

```text
Posts by a single user, chronological order. Total: {n_posts} posts over {current_round} rounds.

{accumulated_texts}
```

---

## 9. Appendix B — Call-shape & Runtime Details

Eight points covering exactly what is sent at runtime, beyond the prompt strings reproduced above.

### 9.1 Message shape

**Two messages per LLM call. No assistant primer.** No few-shot turns. No tool-use turns.

- **HF Inference path** ([hf_client.py:58-71](../src/erisk_task2/tom/hf_client.py#L58-L71)):

  ```python
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user",   "content": user_prompt},
  ]
  client.chat_completion(messages=messages, temperature=temp, max_tokens=2048)
  ```

- **Ollama path** ([llm_client.py:50-67](../src/erisk_task2/tom/llm_client.py#L50-L67)) — `/api/generate` endpoint (not `/api/chat`):

  ```python
  payload = {
      "model":      "llama3.3:70b",
      "system":     system_prompt,
      "prompt":     user_prompt,
      "stream":     False,
      "options":    {"num_ctx": 8192, "temperature": temp},
      "keep_alive": "24h",
  }
  ```

Both Prompt 1 (self-view) and Prompt 2a (observer-view) are sent through this same 2-part shape. Prompt 2b's chained mode (defined but inactive at default config) would also stay 2-part — the previous self-view JSON is inlined into the *user* message, not added as a separate `assistant` turn.

### 9.2 Per-call temperature

**Both calls use the same `temperature = 0.1`.** [tom_module.py:166-193](../src/erisk_task2/tom/tom_module.py#L166-L193) invokes `self.llm_client.generate_json(sys, user)` with **no `temperature` kwarg**; the client defaults to `self.temperature` set from config at init time. There is no per-prompt override (self vs observer) and no per-attempt override (initial vs retry).

### 9.3 Server-side JSON enforcement

**None.** No `response_format`, no `json_object`/`json_schema` mode, no grammar, no GBNF, no tool-use JSON schema. Confirmed by `grep response_format|json_object|json_schema|grammar` over `src/erisk_task2/tom/` → zero matches. JSON validity is policed only by:

1. The prompt-level directive "Respond with valid JSON only." (every system prompt).
2. The inline `Output format:` JSON template (every system prompt).
3. Client-side recovery in `_extract_json()` ([llm_client.py:115-146](../src/erisk_task2/tom/llm_client.py#L115-L146), [hf_client.py:118-149](../src/erisk_task2/tom/hf_client.py#L118-L149)): direct `json.loads` → ```` ```json ```` fence strip → outermost `{...}` slice.

### 9.4 Output-token cap

Set at the API call site, not in the prompt.

| Backend | Cap | Source |
| --- | --- | --- |
| HF Inference | `max_tokens = 2048` | [hf_client.py:26, 70](../src/erisk_task2/tom/hf_client.py#L26) |
| Ollama | **Not set** (`num_predict` absent from `options`) → model default | [llm_client.py:50-60](../src/erisk_task2/tom/llm_client.py#L50-L60) |

The prompts do not ask the model for a length budget.

### 9.5 Batching policy

**Sequential, synchronous, per (subject, round, thread).**

Per `_assess_llm` call ([tom_module.py:155-204](../src/erisk_task2/tom/tom_module.py#L155-L204)):

```text
if thread.has_target_text:
    self_view = LLM(Prompt 1, target's own texts)        # blocking HTTP call #1
observer_view = LLM(Prompt 2a, formatted_thread)         # blocking HTTP call #2
```

→ up to two blocking calls per (subject, round). No batching across subjects, no batching across rounds, no concurrency primitives. (The encoder step *is* batched across users in a round — [pipeline.py:998-1015](../src/erisk_task2/pipeline.py#L998-L1015) — but that's sentence-transformer encoding, not LLM inference.)

System prompts are pre-formatted once in `ToMModule.__init__` ([tom_module.py:60-62](../src/erisk_task2/tom/tom_module.py#L60-L62)) and reused byte-identically across every subsequent call — enables Ollama KV-cache hits, no benefit on HF chat-completion.

### 9.6 Few-shot examples — privacy check

**No few-shot examples exist in any prompt body.** Hence no example posts, subject IDs, subreddit names, or PII baked into the prompts. The only user-derived text reaching the LLM is the **runtime substitution** of `{target_user_texts}` (Prompt 1) and `{formatted_thread}` (Prompt 2a) — those are the actual eRisk subject's posts at request time, not appendix content. Comment-thread authors appear by their reddit username in the formatted thread (no further pseudonymisation is applied beyond what the eRisk organisers already did upstream).

**Implication for publication.** This appendix as written contains no participant data and needs no redaction. If you publish a *sample* runtime call (e.g., to illustrate one fully populated 2a request), redact the `submission_id`/author handles in the thread transcript.

### 9.7 Retry policy

Same `(temperature, max_tokens, prompt)` on every attempt; no escalation strategy.

| Aspect | Value | Source |
| --- | --- | --- |
| `max_retries` | 3 (i.e., up to 2 retries after the initial call) | [hf_client.py:28, 64](../src/erisk_task2/tom/hf_client.py#L28), [llm_client.py:29, 64](../src/erisk_task2/tom/llm_client.py#L29) |
| Back-off | `time.sleep(2 ** attempt)` seconds (1 s, 2 s) | [hf_client.py:82-83](../src/erisk_task2/tom/hf_client.py#L82-L83), [llm_client.py:81-82](../src/erisk_task2/tom/llm_client.py#L81-L82) |
| Temperature on retry | unchanged (no temp→0 fallback) | same |
| Prompt on retry | unchanged (no stricter "ONLY emit JSON" suffix) | same |
| Outcome on full failure | empty string returned → `generate_json` returns `None` → 21-d sub-vector left at zero | [hf_client.py:84-86](../src/erisk_task2/tom/hf_client.py#L84-L86), [tom_module.py:169-176, 195-202](../src/erisk_task2/tom/tom_module.py#L169-L202) |

### 9.8 What slot [42] holds at inference (live mode)

The 47-d ToM block is documented as if it were always populated by Prompt 1's `depression_probability ∈ [0, 1]`. **In the live submission run, Option A is used and the value at slot [42] is the L2 norm of an unnormalised embedding mean** — a fundamentally different scale.

Trace ([pipeline.py:120-225, 326-362](../src/erisk_task2/pipeline.py#L120-L362), [pipeline.py:995-1036](../src/erisk_task2/pipeline.py#L995-L1036)):

1. Per round, `round_mean = encoder.encode(target_texts).mean(axis=0)` (1920-d).
2. `profile.self_view_history.append({"embedding": round_mean})`.
3. `other_texts` are gathered but the live precompute buffer contains only **target_texts + title** (no `other_comments`, explicitly skipped to avoid ~10× encoding overhead — [pipeline.py:995-996](../src/erisk_task2/pipeline.py#L995-L996)). The branch test `emb_offset + len(other_texts) <= len(precomputed)` fails → `profile.observer_view_history.append(None)`.
4. At feature assembly, `_compute_tom_a_features` produces:

| Slot | Live value | Mechanism |
| --- | --- | --- |
| `[0:21]` self-view LLM symptom scores | **0** | Option A doesn't emit per-symptom scores |
| `[21:42]` observer-view LLM symptom scores | **0** | same |
| **`[42]`** self depression_probability proxy | **`‖ mean_over_rounds( round_mean ) ‖₂`** — non-zero | mean → norm (in that order) over the per-round mean target-text embeddings; embeddings are not L2-normalised, so the value is unbounded above |
| `[43]` observer depression_probability proxy | **0** | observer_view_history is empty (other_comments not encoded live) |
| `[44]` insight_gap | **0** | requires both self and observer embeddings |
| `[45]` observer_concern_level / 3 | **0** | Option A doesn't set this |
| `[46]` community_response_type encoded | **0** | same |

**Important caveats for the paper:**

- **Order of operations**: `mean → norm`, not `norm → mean`. The cross-round average of the round-level mean embeddings is taken first, *then* its L2 norm — i.e., the magnitude of the centroid, not the average magnitude.
- **Embedding normalisation**: the three sentence transformers (mpnet-base-v2, MiniLM-L12-v2, distilroberta-v1) do not L2-normalise their outputs by default, so `‖·‖₂` is roughly a "how much accumulated semantic mass" proxy. It is **not** a depression probability, **not** in `[0, 1]`, and **not** the quantity the XGBoost classifier learned to expect (which was the LLM's `depression_probability ∈ [0, 1]` from Prompt 1).
- **Distributional mismatch at slot [42]**: at training, slot [42] was a calibrated `[0, 1]` probability from Llama-3.3-70B. At inference, it is an unnormalised real-valued embedding norm. The XGBoost leaf threshold learned around the former simply does not fire on the latter.
- Combined with `[0:42] = 0` and `[43:46] = 0`, this is the technical mechanism behind the **R0 ≈ R4** result reported in §6 / §18 of [task2_solution_description.md](task2_solution_description.md): only one ToM slot is non-zero live, and that one slot carries the wrong distribution.
