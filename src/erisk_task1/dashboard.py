"""Streamlit dashboard for Task 1 run analysis.

Launch with: streamlit run src/erisk_task1/dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNS_DIR = Path("runs/task1/personas")
BDI_ITEMS = {
    1: "Sadness", 2: "Pessimism", 3: "Past failure", 4: "Loss of pleasure",
    5: "Guilty feelings", 6: "Punishment feelings", 7: "Self-dislike",
    8: "Self-criticalness", 9: "Suicidal thoughts", 10: "Crying",
    11: "Agitation", 12: "Loss of interest", 13: "Indecisiveness",
    14: "Worthlessness", 15: "Loss of energy", 16: "Sleep changes",
    17: "Irritability", 18: "Appetite changes", 19: "Concentration difficulty",
    20: "Tiredness/fatigue", 21: "Loss of interest in sex",
}

SEVERITY_COLORS = {
    "minimal": "#4CAF50",
    "mild": "#FFC107",
    "moderate": "#FF9800",
    "severe": "#F44336",
}

STATE_COLORS = {
    "SCORED": "#1976D2",
    "NO_EVIDENCE": "#9E9E9E",
    "EVIDENCE_OF_ABSENCE": "#66BB6A",
}

GOLDEN_SCORES = {
    "Maria": 40, "Marco": 38, "Elena": 35, "Linda": 28,
    "Laura": 23, "James": 22, "Alex": 15, "Gabriel": 13,
    "Ethan": 12, "Priya": 7, "Maya": 6, "Noah": 5,
}

GOLDEN_KEY_SYMPTOMS = {
    "Maria": ["Sadness", "Self-criticalness", "Loss of interest", "Tiredness or fatigue"],
    "Marco": ["Past failure", "Agitation", "Loss of interest", "Concentration difficulty"],
    "Elena": ["Pessimism", "Crying", "Tiredness or fatigue", "Loss of interest"],
    "Linda": ["Guilty feelings", "Pessimism", "Indecisiveness", "Tiredness or fatigue"],
    "Laura": ["Sadness", "Worthlessness", "Tiredness or fatigue", "Concentration difficulty"],
    "James": ["Loss of energy", "Worthlessness", "Loss of interest", "Indecisiveness"],
    "Alex": ["Concentration difficulty", "Irritability", "Changes in sleeping pattern", "Changes in appetite"],
    "Gabriel": ["Irritability", "Self-criticalness", "Changes in appetite", "Self-dislike"],
    "Ethan": ["Loss of pleasure", "Loss of interest", "Changes in sleeping pattern", "Indecisiveness"],
    "Priya": ["Agitation", "Changes in sleeping pattern", "Self-criticalness", "Loss of pleasure"],
    "Maya": ["Agitation", "Self-criticalness", "Tiredness or fatigue"],
    "Noah": ["Self-dislike", "Loss of energy", "Irritability", "Changes in sleeping pattern"],
}

BOUNDARY_PERSONAS = {"Ethan", "Gabriel", "Alex", "Linda"}

TALKDEP_DIR = Path("data/TalkDep/persona-development/conversation_generation/final_conversations")


def score_to_band(total: int) -> str:
    if total <= 13:
        return "minimal"
    elif total <= 19:
        return "mild"
    elif total <= 28:
        return "moderate"
    else:
        return "severe"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def discover_personas(runs_dir: str) -> list[str]:
    """Find all persona directories."""
    p = Path(runs_dir)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir() and d.name.startswith("persona")])


@st.cache_data
def load_conversation(runs_dir: str, persona: str, run_id: int) -> list[dict] | None:
    path = Path(runs_dir) / persona / f"conversation_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_results(runs_dir: str, persona: str, run_id: int) -> list[dict] | None:
    path = Path(runs_dir) / persona / f"results_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_internal(runs_dir: str, persona: str, run_id: int) -> dict | None:
    path = Path(runs_dir) / persona / f"internal_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_talkdep_conversation(name: str) -> str | None:
    fname = f"{name.lower()}-final-conversation.txt"
    fpath = TALKDEP_DIR / fname
    if not fpath.exists():
        return None
    return fpath.read_text(encoding="utf-8")


def parse_item_scores(internal: dict) -> pd.DataFrame:
    """Parse item_scores from internal JSON into a DataFrame."""
    rows = []
    for key, val in internal.get("item_scores", {}).items():
        # key format: "13_indecisiveness" or "1_sadness"
        parts = key.split("_", 1)
        item_id = int(parts[0])
        item_name = BDI_ITEMS.get(item_id, parts[1] if len(parts) > 1 else f"Item {item_id}")
        rows.append({
            "Item ID": item_id,
            "Item": item_name,
            "Score": val.get("score"),
            "Confidence": val.get("confidence", 0.0),
            "State": val.get("state", ""),
            "Source": val.get("source", ""),
            "Evidence": val.get("evidence", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Item ID").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Tab: Conversation Viewer
# ---------------------------------------------------------------------------

def render_conversation_tab(runs_dir: str, persona: str, run_id: int):
    conv = load_conversation(runs_dir, persona, run_id)
    if conv is None:
        st.warning("No conversation data found.")
        return

    st.subheader(f"Conversation — {persona}, Run {run_id}")

    # Stats
    user_turns = [t for t in conv if t.get("role") == "user"]
    assistant_turns = [t for t in conv if t.get("role") == "assistant"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total turns", len(conv))
    col2.metric("Interviewer messages", len(user_turns))
    col3.metric("Persona responses", len(assistant_turns))

    # Word counts
    persona_words = sum(len(t.get("message", "").split()) for t in assistant_turns)
    interviewer_words = sum(len(t.get("message", "").split()) for t in user_turns)
    col1.metric("Persona words", persona_words)
    col2.metric("Interviewer words", interviewer_words)
    col3.metric("Avg persona response", round(persona_words / max(len(assistant_turns), 1), 1))

    st.divider()

    # Chat display
    for turn in conv:
        role = turn.get("role", "")
        msg = turn.get("message", "")
        turn_num = turn.get("turn", "")

        if role == "user":
            with st.chat_message("user", avatar="🎤"):
                st.caption(f"Turn {turn_num} — Interviewer")
                st.write(msg)
        else:
            with st.chat_message("assistant", avatar="🧑"):
                st.caption(f"Turn {turn_num} — Persona")
                st.write(msg)


# ---------------------------------------------------------------------------
# Tab: BDI-II Scoring
# ---------------------------------------------------------------------------

def render_scoring_tab(runs_dir: str, persona: str, run_id: int):
    internal = load_internal(runs_dir, persona, run_id)
    if internal is None:
        st.warning("No internal scoring data found.")
        return

    results = load_results(runs_dir, persona, run_id)

    st.subheader(f"BDI-II Scoring — {persona}, Run {run_id}")

    # Summary metrics
    bdi_score = internal.get("bdi-score", 0)
    band = internal.get("severity_band", score_to_band(bdi_score))
    key_symptoms = internal.get("key-symptoms", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("BDI-II Total", bdi_score)
    col2.markdown(
        f"**Severity:** <span style='color:{SEVERITY_COLORS.get(band, '#999')};font-size:1.2em'>"
        f"{band.upper()}</span>",
        unsafe_allow_html=True,
    )
    col3.markdown("**Top-4 Symptoms:**\n" + "\n".join(f"- {s}" for s in key_symptoms))

    # Correction info
    correction = internal.get("correction")
    if correction:
        with st.expander("Score Correction Details"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Raw Total", correction.get("raw_total", "—"))
            cc2.metric("Strategy", correction.get("correction_strategy", "—"))
            cc3.metric("Delta", correction.get("correction_delta", "—"))

    # Scoring metadata
    meta = internal.get("scoring_metadata", {})
    if meta:
        with st.expander("Scoring Pipeline"):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Pass 1", meta.get("pass1_total", "—"))
            mc2.metric("Pass 2", meta.get("pass2_total", "—"))
            mc3.metric("Final", meta.get("final_total", "—"))
            mc4.metric("Conv. Turns", meta.get("conversation_turns", "—"))

    st.divider()

    # Item scores table
    df = parse_item_scores(internal)
    if df.empty:
        st.info("No item scores available.")
        return

    # Bar chart of scores
    df_scored = df[df["Score"].notna()].copy()
    df_scored["Score"] = df_scored["Score"].astype(int)

    if not df_scored.empty:
        fig = go.Figure()
        colors = [STATE_COLORS.get(s, "#999") for s in df_scored["State"]]
        fig.add_trace(go.Bar(
            x=df_scored["Item"],
            y=df_scored["Score"],
            marker_color=colors,
            text=df_scored["Score"],
            textposition="outside",
            hovertext=df_scored["Evidence"],
        ))
        fig.update_layout(
            title="BDI-II Item Scores",
            yaxis_title="Score (0-3)",
            yaxis_range=[0, 3.5],
            xaxis_tickangle=-45,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    if not df_scored.empty:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=df_scored["Score"].tolist() + [df_scored["Score"].iloc[0]],
            theta=df_scored["Item"].tolist() + [df_scored["Item"].iloc[0]],
            fill="toself",
            name="Scores",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
            title="BDI-II Symptom Profile",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Detailed table
    st.subheader("Item Details")

    def highlight_state(row):
        color = STATE_COLORS.get(row["State"], "")
        return [f"color: {color}"] * len(row) if color else [""] * len(row)

    display_df = df[["Item ID", "Item", "Score", "Confidence", "State", "Source"]].copy()
    st.dataframe(
        display_df.style.apply(highlight_state, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Evidence expander
    with st.expander("Evidence Details"):
        for _, row in df.iterrows():
            if row["Evidence"]:
                state_color = STATE_COLORS.get(row["State"], "#999")
                st.markdown(
                    f"**{row['Item ID']}. {row['Item']}** "
                    f"<span style='color:{state_color}'>[{row['State']}]</span> "
                    f"Score: {row['Score']}, Conf: {row['Confidence']:.2f}",
                    unsafe_allow_html=True,
                )
                st.caption(row["Evidence"])
                st.divider()


# ---------------------------------------------------------------------------
# Tab: Cross-Run Comparison
# ---------------------------------------------------------------------------

def render_cross_run_tab(runs_dir: str, persona: str):
    st.subheader(f"Cross-Run Comparison — {persona}")

    run_data = {}
    for run_id in [1, 2, 3]:
        internal = load_internal(runs_dir, persona, run_id)
        if internal:
            run_data[run_id] = internal

    if not run_data:
        st.warning("No run data found for this persona.")
        return

    # Summary comparison
    cols = st.columns(len(run_data))
    for i, (run_id, data) in enumerate(run_data.items()):
        bdi = data.get("bdi-score", 0)
        band = data.get("severity_band", score_to_band(bdi))
        with cols[i]:
            st.markdown(f"### Run {run_id}")
            st.metric("BDI-II Total", bdi)
            st.markdown(
                f"<span style='color:{SEVERITY_COLORS.get(band, '#999')};font-weight:bold'>"
                f"{band.upper()}</span>",
                unsafe_allow_html=True,
            )
            symptoms = data.get("key-symptoms", [])
            for s in symptoms:
                st.markdown(f"- {s}")

    st.divider()

    # Item-level comparison table
    all_items = set()
    for data in run_data.values():
        all_items.update(data.get("item_scores", {}).keys())

    rows = []
    for key in sorted(all_items):
        parts = key.split("_", 1)
        item_id = int(parts[0])
        item_name = BDI_ITEMS.get(item_id, parts[1] if len(parts) > 1 else key)
        row = {"Item ID": item_id, "Item": item_name}
        for run_id, data in run_data.items():
            item = data.get("item_scores", {}).get(key, {})
            score = item.get("score")
            row[f"Run {run_id} Score"] = score if score is not None else "—"
            row[f"Run {run_id} Conf"] = round(item.get("confidence", 0), 2)
            row[f"Run {run_id} State"] = item.get("state", "—")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Item ID").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Score comparison chart
    chart_data = []
    for run_id, data in run_data.items():
        for key, item in data.get("item_scores", {}).items():
            parts = key.split("_", 1)
            item_id = int(parts[0])
            score = item.get("score")
            if score is not None:
                chart_data.append({
                    "Item": BDI_ITEMS.get(item_id, key),
                    "Score": score,
                    "Run": f"Run {run_id}",
                })

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df, x="Item", y="Score", color="Run",
            barmode="group", title="Item Scores Across Runs",
        )
        fig.update_layout(xaxis_tickangle=-45, height=400, yaxis_range=[0, 3.5])
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Cross-Persona Overview
# ---------------------------------------------------------------------------

def render_overview_tab(runs_dir: str, run_id: int):
    st.subheader(f"Cross-Persona Overview — Run {run_id}")

    personas = discover_personas(runs_dir)
    if not personas:
        st.warning("No personas found.")
        return

    # Summary table
    summary_rows = []
    all_item_data = {}
    for persona in personas:
        internal = load_internal(runs_dir, persona, run_id)
        if internal is None:
            continue
        bdi = internal.get("bdi-score", 0)
        band = internal.get("severity_band", score_to_band(bdi))
        symptoms = internal.get("key-symptoms", [])
        meta = internal.get("scoring_metadata", {})

        summary_rows.append({
            "Persona": persona,
            "BDI-II": bdi,
            "Band": band,
            "Top Symptoms": ", ".join(symptoms),
            "Pass 1": meta.get("pass1_total", "—"),
            "Pass 2": meta.get("pass2_total", "—"),
            "Final": meta.get("final_total", "—"),
            "Turns": meta.get("conversation_turns", "—"),
        })

        # Collect item scores for heatmap
        item_scores = {}
        for key, val in internal.get("item_scores", {}).items():
            parts = key.split("_", 1)
            item_id = int(parts[0])
            item_scores[item_id] = val.get("score")
        all_item_data[persona] = item_scores

    if not summary_rows:
        st.warning("No data for this run.")
        return

    df_summary = pd.DataFrame(summary_rows)

    def color_band(val):
        color = SEVERITY_COLORS.get(val, "")
        return f"color: {color}; font-weight: bold" if color else ""

    st.dataframe(
        df_summary.style.map(color_band, subset=["Band"]),
        use_container_width=True,
        hide_index=True,
    )

    # BDI-II scores bar chart
    fig = go.Figure()
    colors = [SEVERITY_COLORS.get(r["Band"], "#999") for r in summary_rows]
    fig.add_trace(go.Bar(
        x=[r["Persona"] for r in summary_rows],
        y=[r["BDI-II"] for r in summary_rows],
        marker_color=colors,
        text=[r["BDI-II"] for r in summary_rows],
        textposition="outside",
    ))
    # Band thresholds
    for threshold, label in [(13, "Minimal/Mild"), (19, "Mild/Moderate"), (28, "Moderate/Severe")]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                      annotation_text=label, annotation_position="right")
    fig.update_layout(title="BDI-II Scores", yaxis_title="BDI-II Total", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    if all_item_data:
        heatmap_data = []
        for persona, items in all_item_data.items():
            for item_id in range(1, 22):
                score = items.get(item_id)
                heatmap_data.append({
                    "Persona": persona,
                    "Item": BDI_ITEMS.get(item_id, f"Item {item_id}"),
                    "Score": score if score is not None else -1,
                })
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot = heatmap_df.pivot(index="Item", columns="Persona", values="Score")
        # Sort items by ID
        item_order = [BDI_ITEMS[i] for i in range(1, 22)]
        pivot = pivot.reindex([i for i in item_order if i in pivot.index])

        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=3,
            labels=dict(color="Score"),
            title="BDI-II Item Heatmap Across Personas",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: TalkDep / Ablation
# ---------------------------------------------------------------------------

def render_talkdep_tab():
    st.subheader("TalkDep Personas — Golden Reference")

    # TalkDep overview table
    rows = []
    for name, total in sorted(GOLDEN_SCORES.items(), key=lambda x: -x[1]):
        band = score_to_band(total)
        symptoms = GOLDEN_KEY_SYMPTOMS.get(name, [])
        boundary = name in BOUNDARY_PERSONAS
        rows.append({
            "Persona": name,
            "BDI-II": total,
            "Band": band,
            "Boundary": "Yes" if boundary else "",
            "Key Symptoms": ", ".join(symptoms),
        })

    df = pd.DataFrame(rows)

    def color_band(val):
        color = SEVERITY_COLORS.get(val, "")
        return f"color: {color}; font-weight: bold" if color else ""

    st.dataframe(
        df.style.map(color_band, subset=["Band"]),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    fig = go.Figure()
    colors = [SEVERITY_COLORS.get(r["Band"], "#999") for r in rows]
    fig.add_trace(go.Bar(
        x=[r["Persona"] for r in rows],
        y=[r["BDI-II"] for r in rows],
        marker_color=colors,
        text=[r["BDI-II"] for r in rows],
        textposition="outside",
    ))
    for threshold, label in [(13, "Minimal/Mild"), (19, "Mild/Moderate"), (28, "Moderate/Severe")]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                      annotation_text=label, annotation_position="right")
    fig.update_layout(
        title="TalkDep Golden BDI-II Scores",
        yaxis_title="BDI-II Total", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Conversation viewer
    st.subheader("TalkDep Conversation Viewer")
    persona_name = st.selectbox(
        "Select TalkDep persona",
        list(GOLDEN_SCORES.keys()),
        key="talkdep_persona",
    )

    conv_text = load_talkdep_conversation(persona_name)
    if conv_text is None:
        st.warning(f"Conversation file not found for {persona_name}")
        return

    # Info bar
    total = GOLDEN_SCORES[persona_name]
    band = score_to_band(total)
    symptoms = GOLDEN_KEY_SYMPTOMS.get(persona_name, [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Golden BDI-II", total)
    col2.markdown(
        f"**Band:** <span style='color:{SEVERITY_COLORS.get(band, '#999')}'>"
        f"{band.upper()}</span>",
        unsafe_allow_html=True,
    )
    col3.markdown("**Key Symptoms:** " + ", ".join(symptoms))

    if persona_name in BOUNDARY_PERSONAS:
        st.info(f"{persona_name} is a **boundary persona** (near band edges).")

    st.divider()

    # Parse and display conversation
    for line in conv_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match lines like "**Therapist:** text" or "**Maria:** text"
        m = re.match(r"\*\*(\w+):\*\*\s*(.*)", line)
        if m:
            role_name = m.group(1)
            utterance = m.group(2).strip()
            if not utterance:
                continue
            if role_name == "Therapist":
                with st.chat_message("user", avatar="🎤"):
                    st.write(utterance)
            else:
                with st.chat_message("assistant", avatar="🧑"):
                    st.write(utterance)
        elif line.startswith("###") or line.startswith("Patient name"):
            st.markdown(f"**{line.strip('#').strip()}**")

    st.divider()

    # Ablation results loader
    st.subheader("Ablation Study Results")
    ablation_dir = Path("runs/ablation")
    summary_path = ablation_dir / "ablation_summary.json"

    if summary_path.exists():
        _render_ablation_results(summary_path, ablation_dir)
    else:
        st.info(
            "No ablation results found. Run the ablation study first:\n\n"
            "```bash\nunset VIRTUAL_ENV && uv run python -m erisk_task1.cli ablation --talkdep data/TalkDep\n```"
        )

        # Check for individual result files
        if ablation_dir.exists():
            result_files = sorted(ablation_dir.rglob("*.json"))
            if result_files:
                st.markdown("**Found individual result files:**")
                _render_ablation_from_files(result_files)


def _render_ablation_results(summary_path: Path, ablation_dir: Path):
    """Render ablation results from the summary JSON."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Aggregate metrics table
    agg_rows = []
    for r in data:
        agg_rows.append({
            "Config": r["config"],
            "DCHR": f"{r['dchr']:.1%}",
            "MAD": f"{r['mad']:.1f}",
            "ADODL": f"{r['adodl']:.3f}",
            "ASHR": f"{r.get('ashr_proxy', 0):.1%}",
            "Boundary Acc.": f"{r.get('boundary_accuracy', 0):.1%}",
            "N": r["n_personas"],
        })
    st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)

    # Metrics comparison chart
    metrics_df = pd.DataFrame([{
        "Config": r["config"],
        "DCHR": r["dchr"],
        "ADODL": r["adodl"],
        "ASHR": r.get("ashr_proxy", 0),
        "Boundary": r.get("boundary_accuracy", 0),
    } for r in data])

    fig = px.bar(
        metrics_df.melt(id_vars="Config", var_name="Metric", value_name="Value"),
        x="Config", y="Value", color="Metric", barmode="group",
        title="Ablation Metrics Comparison",
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Per-persona breakdown
    st.subheader("Per-Persona Breakdown")
    selected_config = st.selectbox(
        "Select configuration",
        [r["config"] for r in data],
        key="ablation_config",
    )

    config_data = next((r for r in data if r["config"] == selected_config), None)
    if config_data and "per_persona" in config_data:
        per_persona = config_data["per_persona"]

        # Predicted vs Golden chart
        fig = go.Figure()
        names = [p["name"] for p in per_persona]
        golden = [p["golden"] for p in per_persona]
        predicted = [p["predicted"] for p in per_persona]

        fig.add_trace(go.Bar(name="Golden", x=names, y=golden, marker_color="#2196F3"))
        fig.add_trace(go.Bar(name="Predicted", x=names, y=predicted, marker_color="#FF9800"))

        for threshold, label in [(13, "Minimal/Mild"), (19, "Mild/Moderate"), (28, "Moderate/Severe")]:
            fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                          annotation_text=label, annotation_position="right")
        fig.update_layout(
            barmode="group", title=f"Golden vs Predicted — {selected_config}",
            yaxis_title="BDI-II Total", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-persona table
        pp_df = pd.DataFrame(per_persona)

        def color_band_ok(val):
            if val is True:
                return "color: #4CAF50; font-weight: bold"
            elif val is False:
                return "color: #F44336; font-weight: bold"
            return ""

        st.dataframe(
            pp_df.style.map(color_band_ok, subset=["band_ok"]),
            use_container_width=True,
            hide_index=True,
        )


def _render_ablation_from_files(result_files: list[Path]):
    """Render ablation results from individual per-persona JSON files."""
    results_by_config = {}
    for f in result_files:
        if f.name == "ablation_summary.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            config = data.get("config", "unknown")
            results_by_config.setdefault(config, []).append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not results_by_config:
        return

    for config_name, results in sorted(results_by_config.items()):
        with st.expander(f"Config: {config_name} ({len(results)} personas)"):
            rows = []
            for r in sorted(results, key=lambda x: x.get("persona", "")):
                rows.append({
                    "Persona": r.get("persona", ""),
                    "Golden": r.get("golden_total", ""),
                    "Predicted": r.get("predicted_total", ""),
                    "Golden Band": r.get("golden_band", ""),
                    "Predicted Band": r.get("predicted_band", ""),
                    "Band OK": r.get("band_correct", ""),
                    "Deviation": r.get("deviation", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Multi-Run-Set Comparison
# ---------------------------------------------------------------------------

def discover_run_sets() -> dict[str, str]:
    """Find all available result sets (directories with persona subdirs)."""
    base = Path("runs/task1")
    if not base.exists():
        return {}
    sets = {}
    # Main personas dir
    if (base / "personas").exists():
        sets["personas (latest)"] = str(base / "personas")
    # Dated dirs
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("task1_results_") and not d.name.endswith(".zip"):
            # Check if it has persona subdirs
            has_personas = any(
                sd.is_dir() and sd.name.startswith("persona") for sd in d.iterdir()
            )
            if has_personas:
                sets[d.name] = str(d)
    return sets


def render_multi_run_set_tab():
    st.subheader("Multi-Result-Set Comparison")

    run_sets = discover_run_sets()
    if len(run_sets) < 2:
        st.info("Need at least 2 result sets to compare. Currently found: " +
                ", ".join(run_sets.keys()) if run_sets else "none")
        return

    selected = st.multiselect(
        "Select result sets to compare",
        list(run_sets.keys()),
        default=list(run_sets.keys())[:2],
    )

    if len(selected) < 2:
        st.info("Select at least 2 result sets.")
        return

    run_id = st.selectbox("Run number", [1, 2, 3], key="multi_run_id")

    # Build comparison data
    comparison = {}
    for set_name in selected:
        runs_dir = run_sets[set_name]
        personas = discover_personas(runs_dir)
        set_data = {}
        for persona in personas:
            internal = load_internal(runs_dir, persona, run_id)
            if internal:
                set_data[persona] = {
                    "BDI-II": internal.get("bdi-score", 0),
                    "Band": internal.get("severity_band", ""),
                    "Symptoms": internal.get("key-symptoms", []),
                }
        comparison[set_name] = set_data

    # Find common personas
    all_personas = set()
    for set_data in comparison.values():
        all_personas.update(set_data.keys())
    common = sorted(all_personas)

    if not common:
        st.warning("No common personas found.")
        return

    # Comparison table
    rows = []
    for persona in common:
        row = {"Persona": persona}
        for set_name in selected:
            data = comparison[set_name].get(persona, {})
            row[f"{set_name} BDI"] = data.get("BDI-II", "—")
            row[f"{set_name} Band"] = data.get("Band", "—")
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Chart
    chart_data = []
    for persona in common:
        for set_name in selected:
            data = comparison[set_name].get(persona, {})
            bdi = data.get("BDI-II")
            if bdi is not None:
                chart_data.append({
                    "Persona": persona,
                    "BDI-II": bdi,
                    "Result Set": set_name,
                })

    if chart_data:
        fig = px.bar(
            pd.DataFrame(chart_data),
            x="Persona", y="BDI-II", color="Result Set",
            barmode="group", title=f"BDI-II Scores Comparison (Run {run_id})",
        )
        for threshold, label in [(13, "Minimal/Mild"), (19, "Mild/Moderate"), (28, "Moderate/Severe")]:
            fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                          annotation_text=label, annotation_position="right")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Task 1 — Depression Interview Analysis",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Task 1 — Depression Interview Analysis")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")

        run_sets = discover_run_sets()
        if not run_sets:
            st.error("No run data found in runs/task1/")
            return

        runs_dir = st.selectbox(
            "Result Set",
            list(run_sets.keys()),
            format_func=lambda x: x,
        )
        runs_dir_path = run_sets[runs_dir]

        personas = discover_personas(runs_dir_path)
        if not personas:
            st.warning("No personas in this result set.")

        selected_persona = st.selectbox("Persona", personas) if personas else None
        selected_run = st.selectbox("Run", [1, 2, 3])

    # Tabs
    tab_names = [
        "Conversation",
        "BDI-II Scoring",
        "Cross-Run Comparison",
        "Cross-Persona Overview",
        "TalkDep & Ablation",
        "Multi-Result-Set",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if selected_persona:
            render_conversation_tab(runs_dir_path, selected_persona, selected_run)
        else:
            st.info("Select a persona from the sidebar.")

    with tabs[1]:
        if selected_persona:
            render_scoring_tab(runs_dir_path, selected_persona, selected_run)
        else:
            st.info("Select a persona from the sidebar.")

    with tabs[2]:
        if selected_persona:
            render_cross_run_tab(runs_dir_path, selected_persona)
        else:
            st.info("Select a persona from the sidebar.")

    with tabs[3]:
        render_overview_tab(runs_dir_path, selected_run)

    with tabs[4]:
        render_talkdep_tab()

    with tabs[5]:
        render_multi_run_set_tab()


if __name__ == "__main__":
    main()
