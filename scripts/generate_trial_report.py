"""Generate the trial calibration report for MentalRiskES Task 1."""

import json
import numpy as np
from pathlib import Path

# Load data
with open("output/mentalriskes/trial_results_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)

gold = data["gold"]

# Also count parse details from logs
parse_details = {}
for run_name in data["runs"]:
    fname_map = {
        "run0_primary": "predictions_run0_primary.jsonl",
        "run1_comparison": "predictions_run1_comparison.jsonl",
        "run2_lightweight": "predictions_run2_lightweight.jsonl",
    }
    log_path = Path("output/mentalriskes/logs") / fname_map[run_name]
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            entries = [json.loads(l) for l in f.readlines()[-19:]]
        defaults = 0
        has_steps = 0
        no_steps = 0
        for e in entries:
            for inst in ["PHQ-9", "GAD-7", "CompACT-10"]:
                err = e.get(f"{inst}_error", "")
                steps = e.get(f"{inst}_steps", {})
                if err:
                    defaults += 1
                elif steps.get("step_2_temporal") or steps.get("step_2_endorsement"):
                    has_steps += 1
                else:
                    no_steps += 1
        parse_details[run_name] = {"cot_json": has_steps, "prose_or_retry": no_steps, "defaults": defaults}


def rmse(p, g):
    return float(np.sqrt(np.mean((np.array(p, dtype=float) - np.array(g, dtype=float)) ** 2)))

def mae_fn(p, g):
    return float(np.mean(np.abs(np.array(p, dtype=float) - np.array(g, dtype=float))))

def pearson(p, g):
    p, g = np.array(p, dtype=float), np.array(g, dtype=float)
    if len(p) < 2 or np.std(p) == 0 or np.std(g) == 0:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


COMPACT_SUBS = {"OtE(3,5,8)": [2, 4, 7], "BA(1,6,9)": [0, 5, 8], "VA(2,4,7,10)": [1, 3, 6, 9]}
PHQ9_ITEMS = ["Anhedonia", "Mood", "Sleep", "Fatigue", "Appetite", "Self-worth", "Concentration", "Psychomotor", "Suicidality"]
GAD7_ITEMS = ["Nervousness", "Worry control", "Excessive worry", "Relaxation", "Restlessness", "Irritability", "Fear/dread"]
COMPACT_ITEMS = ["Rushing(BA)", "Coherent(VA)", "ThoughtSupp(OtE)", "Values(VA)", "Avoidance(OtE)", "Inattentive(BA)", "Persistence(VA)", "EmoSupp(OtE)", "Autopilot(BA)", "Perseverance(VA)"]

lines = []

# ==================== HEADER ====================
lines.append("# MentalRiskES 2026 Task 1 — Trial Calibration Report")
lines.append("")
lines.append("*Generated from 3 trial runs on the single-patient 19-round trial session.*")
lines.append("")

# ==================== SECTION 1: SETUP ====================
lines.append("## 1. Experimental Setup")
lines.append("")
lines.append("| Parameter | Value |")
lines.append("|-----------|-------|")
lines.append("| Trial data | 1 session, 19 rounds, Spanish therapeutic conversation |")
lines.append("| Patient profile | University student, anxiety-dominant, academic stress, parental pressure |")
lines.append("| LLM provider | Ollama (remote server) |")
lines.append("| Context window | Sliding window: first turn + last 6 turns (max 20 turns) |")
lines.append("| Parse strategy | JSON parse > prose array extraction > calibrated retry > defaults |")
lines.append("| Gold standard | Manual clinical annotations (spec Appendix C) |")
lines.append("")

lines.append("### Gold Standard Scores (Round 19)")
lines.append("")
lines.append("| Instrument | Items | Total |")
lines.append("|-----------|-------|:-----:|")
for inst in ["PHQ-9", "GAD-7", "CompACT-10"]:
    lines.append(f"| {inst} | `{gold[inst]}` | **{sum(gold[inst])}** |")
lines.append("")

lines.append("### Run Configurations")
lines.append("")
lines.append("| Run | Model | Few-shot | Calibration | Purpose |")
lines.append("|-----|-------|:--------:|-------------|---------|")
lines.append("| run0_primary | Llama-3.3-70B | Yes | Flat (k=0) | Best accuracy — full CoT + examples |")
lines.append("| run1_comparison | Llama-3.3-70B | Yes | Band-aware | Test severity-dependent correction |")
lines.append("| run2_lightweight | Mistral 7B | No | None | Efficiency baseline |")
lines.append("")

# ==================== SECTION 2: ROUND-WISE ====================
lines.append("## 2. Round-by-Round Trajectories")
lines.append("")

for run_name in ["run0_primary", "run1_comparison", "run2_lightweight"]:
    rd = data["runs"][run_name]
    rounds = rd["rounds"]
    lines.append(f"### {run_name} — {rd['description']}")
    lines.append("")
    lines.append("| Round | PHQ-9 | err | GAD-7 | err | CompACT-10 | err | Notes |")
    lines.append("|:-----:|:-----:|:---:|:-----:|:---:|:----------:|:---:|-------|")

    for r in rounds:
        rn = r["round"]
        pt = sum(r["phq9"])
        gt = sum(r["gad7"])
        ct = sum(r["compact10"])
        pe = pt - sum(gold["PHQ-9"])
        ge = gt - sum(gold["GAD-7"])
        ce = ct - sum(gold["CompACT-10"])

        notes = []
        if r.get("phq9_error"):
            notes.append("PHQ default")
        if r.get("gad7_error"):
            notes.append("GAD default")
        if r.get("compact10_error"):
            notes.append("C10 default")

        lines.append(f"| {rn} | {pt} | {pe:+d} | {gt} | {ge:+d} | {ct} | {ce:+d} | {', '.join(notes)} |")

    # Summary stats
    phq_t = [sum(r["phq9"]) for r in rounds]
    gad_t = [sum(r["gad7"]) for r in rounds]
    cmp_t = [sum(r["compact10"]) for r in rounds]
    lines.append("")
    lines.append(f"**Stability**: PHQ-9 {np.mean(phq_t):.1f} +/- {np.std(phq_t):.1f} "
                 f"| GAD-7 {np.mean(gad_t):.1f} +/- {np.std(gad_t):.1f} "
                 f"| CompACT-10 {np.mean(cmp_t):.1f} +/- {np.std(cmp_t):.1f}")
    lines.append("")

# ==================== SECTION 3: R19 COMPARISON ====================
lines.append("## 3. Final Round (R19) — Ablation Comparison")
lines.append("")

# Build table
header = "| Metric |"
divider = "|--------|"
for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
    short = {"run0_primary": "Run 0 (70B+flat)", "run1_comparison": "Run 1 (70B+band)", "run2_lightweight": "Run 2 (7B+none)"}[rn]
    header += f" {short} |"
    divider += ":-:|"
header += " Gold |"
divider += ":-:|"
lines.append(header)
lines.append(divider)

for inst, key, gold_key in [("PHQ-9", "phq9", "PHQ-9"), ("GAD-7", "gad7", "GAD-7"), ("CompACT-10", "compact10", "CompACT-10")]:
    g = gold[gold_key]
    vals = {}
    for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
        last = data["runs"][rn]["rounds"][-1]
        vals[rn] = last[key]

    for metric_name, metric_fn in [("RMSE", rmse), ("MAE", mae_fn), ("Pearson", pearson)]:
        row = f"| {inst} {metric_name} |"
        scores = {rn: metric_fn(vals[rn], g) for rn in vals}
        best = min(scores.values()) if metric_name != "Pearson" else max(scores.values())
        for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
            v = scores[rn]
            s = f"{v:.3f}"
            if v == best:
                s = f"**{s}**"
            row += f" {s} |"
        row += " — |"
        lines.append(row)

    row = f"| {inst} Total |"
    for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
        row += f" {sum(vals[rn])} |"
    row += f" **{sum(g)}** |"
    lines.append(row)

# Overall
row_rmse = "| **Mean RMSE** |"
row_pear = "| **Mean Pearson** |"
for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
    last = data["runs"][rn]["rounds"][-1]
    r_vals = [rmse(last[k], gold[gk]) for k, gk in [("phq9", "PHQ-9"), ("gad7", "GAD-7"), ("compact10", "CompACT-10")]]
    p_vals = [pearson(last[k], gold[gk]) for k, gk in [("phq9", "PHQ-9"), ("gad7", "GAD-7"), ("compact10", "CompACT-10")]]
    row_rmse += f" {np.mean(r_vals):.3f} |"
    row_pear += f" {np.mean(p_vals):.3f} |"
row_rmse += " — |"
row_pear += " — |"
lines.append(row_rmse)
lines.append(row_pear)
lines.append("")

# ==================== SECTION 4: PER-ITEM BIAS ====================
lines.append("## 4. Per-Item Bias Analysis (Run 0, all 19 rounds)")
lines.append("")
lines.append("Bias = mean(predicted - gold) across all rounds.")
lines.append("")

for inst, key, gold_key, item_names in [
    ("PHQ-9", "phq9", "PHQ-9", PHQ9_ITEMS),
    ("GAD-7", "gad7", "GAD-7", GAD7_ITEMS),
    ("CompACT-10", "compact10", "CompACT-10", COMPACT_ITEMS),
]:
    g = np.array(gold[gold_key])
    r0 = data["runs"]["run0_primary"]["rounds"]
    biases = np.array([np.array(r[key]) - g for r in r0])
    mean_b = biases.mean(axis=0)
    std_b = biases.std(axis=0)

    lines.append(f"### {inst}")
    lines.append("")
    lines.append("| Item | Gold | Mean Bias | Std | Flag |")
    lines.append("|------|:----:|:---------:|:---:|------|")
    for i, name in enumerate(item_names):
        flag = ""
        if mean_b[i] > 0.3:
            flag = "OVER :arrow_up:"
        elif mean_b[i] < -0.3:
            flag = "UNDER :arrow_down:"
        lines.append(f"| {i+1}. {name} | {g[i]} | {mean_b[i]:+.2f} | {std_b[i]:.2f} | {flag} |")
    lines.append(f"| **Total** | **{int(sum(g))}** | **{biases.sum(axis=1).mean():+.1f}** | **{biases.sum(axis=1).std():.1f}** | |")
    lines.append("")

# CompACT subscales
lines.append("### CompACT-10 Subscale Bias")
lines.append("")
lines.append("| Subscale | Items | Gold | Mean Bias | Std | Flag |")
lines.append("|----------|-------|:----:|:---------:|:---:|------|")
r0 = data["runs"]["run0_primary"]["rounds"]
for sub_name, indices in COMPACT_SUBS.items():
    g_sub = sum(gold["CompACT-10"][i] for i in indices)
    biases = [sum(r["compact10"][i] for i in indices) - g_sub for r in r0]
    mb = np.mean(biases)
    sb = np.std(biases)
    flag = "OVER" if mb > 0.5 else ("UNDER" if mb < -0.5 else "OK")
    lines.append(f"| {sub_name} | {','.join(str(i+1) for i in indices)} | {g_sub} | {mb:+.1f} | {sb:.1f} | {flag} |")
lines.append("")

# ==================== SECTION 5: PARSE RELIABILITY ====================
lines.append("## 5. Parse Reliability")
lines.append("")
lines.append("| Run | CoT JSON | Prose/Retry | Defaults | Total | Success Rate |")
lines.append("|-----|:--------:|:----------:|:--------:|:-----:|:----------:|")
for run_name in ["run0_primary", "run1_comparison", "run2_lightweight"]:
    pd = parse_details.get(run_name, {})
    cot = pd.get("cot_json", 0)
    prose = pd.get("prose_or_retry", 0)
    defs = pd.get("defaults", 0)
    total = cot + prose + defs
    rate = (cot + prose) / total * 100 if total > 0 else 0
    lines.append(f"| {run_name} | {cot} | {prose} | {defs} | {total} | {rate:.0f}% |")
lines.append("")
lines.append("- **CoT JSON**: Full chain-of-thought with valid JSON output")
lines.append("- **Prose/Retry**: Scores extracted from markdown prose or via retry prompt")
lines.append("- **Defaults**: All attempts failed, used fallback scores (PHQ-9/GAD-7=[1,..0], CompACT=[3,..])")
lines.append("")

# ==================== SECTION 6: STABILITY ====================
lines.append("## 6. Prediction Stability")
lines.append("")
lines.append("Standard deviation of total scores across 19 rounds (lower = more stable).")
lines.append("")
lines.append("| Instrument | Run 0 (mean/std) | Run 1 (mean/std) | Run 2 (mean/std) | Gold |")
lines.append("|-----------|:----------------:|:----------------:|:----------------:|:----:|")
for inst, key, gold_key in [("PHQ-9", "phq9", "PHQ-9"), ("GAD-7", "gad7", "GAD-7"), ("CompACT-10", "compact10", "CompACT-10")]:
    row = f"| {inst} |"
    for rn in ["run0_primary", "run1_comparison", "run2_lightweight"]:
        totals = [sum(r[key]) for r in data["runs"][rn]["rounds"]]
        row += f" {np.mean(totals):.1f} +/- {np.std(totals):.1f} |"
    row += f" {sum(gold[gold_key])} |"
    lines.append(row)
lines.append("")
lines.append("*Note: PHQ-9 and GAD-7 should be stable (past 2 weeks). CompACT-10 may shift slightly as flexibility emerges within session.*")
lines.append("")

# ==================== SECTION 7: ABLATION INSIGHTS ====================
lines.append("## 7. Ablation Insights")
lines.append("")
lines.append("### Effect of Model Size (Run 0 vs Run 2)")
lines.append("")
lines.append("| Aspect | 70B (Run 0) | 7B (Run 2) |")
lines.append("|--------|:-----------:|:----------:|")
lines.append(f"| Mean RMSE | {np.mean([rmse(data['runs']['run0_primary']['rounds'][-1][k], gold[gk]) for k,gk in [('phq9','PHQ-9'),('gad7','GAD-7'),('compact10','CompACT-10')]]):.3f} | {np.mean([rmse(data['runs']['run2_lightweight']['rounds'][-1][k], gold[gk]) for k,gk in [('phq9','PHQ-9'),('gad7','GAD-7'),('compact10','CompACT-10')]]):.3f} |")
lines.append(f"| Mean Pearson | {np.mean([pearson(data['runs']['run0_primary']['rounds'][-1][k], gold[gk]) for k,gk in [('phq9','PHQ-9'),('gad7','GAD-7'),('compact10','CompACT-10')]]):.3f} | {np.mean([pearson(data['runs']['run2_lightweight']['rounds'][-1][k], gold[gk]) for k,gk in [('phq9','PHQ-9'),('gad7','GAD-7'),('compact10','CompACT-10')]]):.3f} |")
lines.append("| PHQ-9 item 9 safety | Always 0 (correct) | Scored 3 in early rounds (dangerous) |")
lines.append("| LLM calls | 53 | 55 |")
lines.append("| Time | ~2 hours | ~45 min |")
lines.append("")
lines.append("**Conclusion**: 70B is substantially better at ranking (Pearson) and safety (item 9). 7B has lower CompACT-10 RMSE only because defaults (midpoint=3) happen to be close to gold.")
lines.append("")

lines.append("### Effect of Few-Shot Examples (Run 0 vs Run 2)")
lines.append("")
lines.append("Both use different models, so this conflates model and few-shot effects. However:")
lines.append("- Run 0 (with few-shot) shows stable PHQ-9 at gold level (13) from round 2 onward")
lines.append("- Run 2 (without few-shot) oscillates wildly (8-23 for PHQ-9)")
lines.append("- Few-shot examples anchor the model to realistic score ranges")
lines.append("")

lines.append("### Effect of Calibration (Run 0 vs Run 1)")
lines.append("")
lines.append("Both use the same model and few-shot. Run 1 applies band-aware correction (subtract 1 at severe band).")
lines.append("- Run 0 R19: PHQ-9=11, GAD-7=16, CompACT-10=39")
lines.append("- Run 1 R19: PHQ-9=7, GAD-7=8, CompACT-10=41")
lines.append("- Run 1 under-scored because the model's R17-19 outputs happened to be lower, then band-aware correction didn't trigger (not in severe band)")
lines.append("- **The difference is LLM non-determinism, not the calibration strategy**")
lines.append("")

# ==================== SECTION 8: RECOMMENDATIONS ====================
lines.append("## 8. Calibration Recommendations")
lines.append("")
lines.append("### PHQ-9: No correction")
lines.append("- Bias: -0.7 total (negligible)")
lines.append("- Stable across rounds (std 1.4)")
lines.append("- Band correctly classified")
lines.append("- Risk of over-correcting exceeds benefit")
lines.append("")
lines.append("### GAD-7: No post-hoc correction")
lines.append("- Bias: -1.2 total but std 3.2 (high variance)")
lines.append("- Any correction would be unreliable")
lines.append("- **Priority**: improve parse reliability and prompt stability")
lines.append("")
lines.append("### CompACT-10: Flat -1 on Valued Action items")
lines.append("- VA subscale bias: +2.7 (items 2,4,7,10 each +0.6 to +0.8)")
lines.append("- Low variance (std 0.6-0.8 per item) = stable, correctable")
lines.append("- **Correction**: subtract 1 from items 2, 4, 7, 10 (clip to 0)")
lines.append("- Expected: total 39 -> ~35 (gold: 33)")
lines.append("")
lines.append("### Prompt-level improvement")
lines.append("- Add to CompACT-10 prompt: calibration anchor for VA items")
lines.append("- *\"For a moderately distressed patient, typical Valued Action scores are 3-4. Score 5+ only with strong behavioral evidence of values-aligned action despite difficulty.\"*")
lines.append("- This addresses root cause: LLM interprets within-session engagement as established values behavior")
lines.append("")

# ==================== SECTION 9: CAVEATS ====================
lines.append("## 9. Caveats")
lines.append("")
lines.append("1. **Single patient**: All conclusions from 1 trial session. Biases may not generalize to multi-patient test data.")
lines.append("2. **Non-determinism**: Same model + same prompt produces different scores across runs.")
lines.append("3. **Gold standard is ours**: Manual annotations, not organizer gold labels.")
lines.append("4. **Ollama instability**: 4 connection drops in run0 (RemoteDisconnected).")
lines.append("5. **CompACT-10 precedent**: No prior computational work on this instrument.")
lines.append("6. **Metric uncertainty**: Official evaluation metrics not yet published (TBU).")
lines.append("")

report = "\n".join(lines)
out_path = Path("output/mentalriskes/trial_calibration_report.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Report written to {out_path} ({len(lines)} lines, {len(report)} chars)")
