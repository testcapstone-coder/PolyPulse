"""PolyPulse: AI-powered prediction-market demand and historical intelligence.

Deploy this file as `app.py` on Streamlit Community Cloud. Three Hugging Face
models run on CPU and are loaded only when needed. Runtime order is:
validity gate → demand classifier → MiniLM historical retrieval.
No model training occurs in the app.
"""

from html import escape
from pathlib import Path
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demand_prediction import LABELS, load_classifier, load_published_config, predict_demand
from historical_search import clean_text, find_similar_markets, get_extractor, load_reference_data
from question_validity import (
    MODEL_ID as VALIDATOR_MODEL_ID,
    load_validator,
    validate_market_question,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPARISON_CSV = BASE_DIR / "results" / "model_comparison.csv"

CATEGORY_ICONS = {
    "Politics": "🗳️",
    "Cryptocurrency": "🪙",
    "Sports": "🏆",
    "Economics": "📈",
    "Technology": "💻",
    "Entertainment": "🎬",
    "Science": "🔬",
    "Other": "🧭",
}

DEMAND_ICONS = {
    "Low": "🌙",
    "Medium": "⚡",
    "High": "🔥",
}

st.set_page_config(
    page_title="PolyPulse | Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------

def inject_styles():
    """Minimal light visual system: editorial, restrained, and highly readable."""
    st.markdown(
        """
<style>
:root {
    --bg: #f7f8fb;
    --surface: #ffffff;
    --surface-2: #fbfcfe;
    --text: #172033;
    --muted: #6f7b90;
    --line: #e2e7ef;
    --line-strong: #d5dce8;
    --accent: #4659e8;
    --accent-dark: #3546c7;
    --accent-soft: #f0f3ff;
    --success: #1f7a57;
    --warning: #9d6500;
    --danger: #b13b3b;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: rgba(247, 248, 251, .94) !important;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.45rem;
    padding-bottom: 3.2rem;
}

/* ---------- text visibility ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stWidgetLabel"] *,
.stApp details *,
.stApp summary *,
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stToast"] * {
    color: var(--text) !important;
}

.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] * {
    color: var(--muted) !important;
    opacity: 1 !important;
}

.stApp a,
.stApp a:visited {
    color: var(--accent) !important;
}

/* ---------- brand header ---------- */
.pp-hero {
    padding: 0.6rem 0 1.2rem 0;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--line);
}

.pp-brandline {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .8rem;
}

.pp-brandmark {
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: var(--accent);
    color: #fff;
    font-weight: 800;
    font-size: .82rem;
    letter-spacing: -.02em;
}

.pp-brandname {
    color: var(--text);
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -.02em;
}

.pp-kicker {
    color: var(--accent);
    font-size: .73rem;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin-bottom: .5rem;
}

.pp-title {
    max-width: 860px;
    color: var(--text);
    font-size: clamp(2.15rem, 4vw, 3.35rem);
    line-height: 1.06;
    font-weight: 820;
    letter-spacing: -.045em;
    margin: 0;
}

.pp-subtitle {
    max-width: 760px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
    margin-top: .8rem;
}

/* ---------- top navigation ---------- */
div[data-testid="stRadio"] > label {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 1.55rem !important;
    padding: 0 !important;
    margin: .85rem 0 1.65rem 0 !important;
    border-bottom: 1px solid var(--line) !important;
    background: transparent !important;
    overflow-x: auto !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label {
    position: relative;
    display: flex !important;
    flex: 0 0 auto !important;
    align-items: center !important;
    min-height: 43px !important;
    padding: 0 0 .72rem 0 !important;
    border: 0 !important;
    background: transparent !important;
    cursor: pointer !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label p {
    margin: 0 !important;
    color: #738096 !important;
    font-size: .92rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--text) !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked)::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 2px;
    border-radius: 999px;
    background: var(--accent);
}

/* ---------- section hierarchy ---------- */
.pp-section-label {
    color: #8792a5;
    font-size: .69rem;
    font-weight: 800;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin: 0 0 .25rem 0;
}

.pp-section-intro {
    max-width: 740px;
    color: var(--muted);
    font-size: .95rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* ---------- containers ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 3px 12px rgba(32, 45, 72, .035);
}

/* ---------- inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
textarea,
input {
    background: #fff !important;
    color: var(--text) !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-color: var(--line-strong) !important;
    border-radius: 10px !important;
}

textarea::placeholder,
input::placeholder {
    color: #9aa5b6 !important;
    opacity: 1 !important;
}

/* ---------- buttons ---------- */
div.stButton > button {
    min-height: 42px;
    border-radius: 9px;
    border: 1px solid var(--line-strong);
    background: #fff;
    color: #2b3548;
    font-weight: 720;
    box-shadow: none;
}

div.stButton > button:hover {
    border-color: #b9c3d3;
    background: #fafbfe;
}

div.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}

div.stButton > button[kind="primary"] * {
    color: #fff !important;
}

div.stButton > button[kind="primary"]:hover {
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
}

/* ---------- nested result tabs ---------- */
div[data-baseweb="tab-list"] {
    gap: 1.25rem !important;
    padding: 0 !important;
    margin: .6rem 0 1rem 0 !important;
    border-bottom: 1px solid var(--line) !important;
    background: transparent !important;
    overflow-x: auto !important;
}

button[data-baseweb="tab"] {
    min-width: max-content !important;
    min-height: 40px !important;
    padding: 0 0 .7rem 0 !important;
    background: transparent !important;
    border: 0 !important;
    color: #7a8698 !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

button[data-baseweb="tab"] *,
button[data-baseweb="tab"] p {
    color: inherit !important;
    opacity: 1 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ---------- metrics ---------- */
[data-testid="stMetric"] {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #8792a5 !important;
    font-size: .76rem !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--text) !important;
    font-weight: 800 !important;
}

/* ---------- analysis snapshot ---------- */
.pp-analysis-shell {
    border: 1px solid #dfe5ef;
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    margin: .4rem 0 1.25rem 0;
    background: #fff;
    box-shadow: 0 4px 18px rgba(32, 45, 72, .035);
}

.pp-analysis-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: .8rem;
}

.pp-analysis-badge {
    color: var(--accent);
    font-size: .68rem;
    font-weight: 820;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.pp-analysis-question {
    color: var(--text);
    font-size: clamp(1.2rem, 2vw, 1.48rem);
    line-height: 1.36;
    font-weight: 790;
    letter-spacing: -.025em;
    margin-bottom: .3rem;
}

.pp-analysis-event {
    color: var(--muted);
    font-size: .86rem;
    margin-bottom: 1.15rem;
}

.pp-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--line);
}

.pp-kpi {
    min-width: 0;
    padding: 1rem 1rem .15rem 0;
    background: transparent;
}

.pp-kpi + .pp-kpi {
    border-left: 1px solid var(--line);
    padding-left: 1rem;
}

.pp-kpi-label {
    color: #8792a5;
    font-size: .72rem;
    margin-bottom: .32rem;
}

.pp-kpi-value {
    color: var(--text);
    font-size: clamp(1rem, 1.8vw, 1.42rem);
    line-height: 1.18;
    font-weight: 810;
    letter-spacing: -.025em;
    overflow-wrap: anywhere;
}

.pp-demand-high { color: #b84b32; }
.pp-demand-medium { color: #936100; }
.pp-demand-low { color: #246da3; }

/* ---------- overview summary ---------- */
.pp-summary-card {
    border: 0;
    padding: .15rem 0 .1rem 0;
    background: transparent;
}

.pp-summary-headline {
    color: var(--text);
    font-size: 1.02rem;
    line-height: 1.5;
    font-weight: 780;
    margin-bottom: .5rem;
}

.pp-summary-copy {
    color: #566276;
    font-size: .94rem;
    line-height: 1.65;
}

.pp-summary-stat {
    color: var(--accent);
    font-weight: 800;
}

.pp-summary-note {
    color: #8a95a6;
    margin-top: .85rem;
    padding-top: .75rem;
    border-top: 1px solid var(--line);
    font-size: .78rem;
}

/* ---------- historical list ---------- */
.pp-history-list {
    display: flex;
    flex-direction: column;
    gap: .5rem;
    margin: .45rem 0 .85rem 0;
}

.pp-history-row {
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr) 100px 130px;
    gap: .9rem;
    align-items: center;
    padding: .8rem .1rem;
    border-bottom: 1px solid var(--line);
    background: transparent;
}

.pp-history-row:last-child {
    border-bottom: 0;
}

.pp-history-rank {
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: .76rem;
    font-weight: 820;
}

.pp-history-question {
    color: var(--text);
    font-size: .91rem;
    line-height: 1.42;
    font-weight: 700;
}

.pp-history-meta {
    color: #8b96a8;
    font-size: .74rem;
    margin-top: .2rem;
}

.pp-history-stat-label {
    color: #929cad;
    font-size: .66rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .15rem;
}

.pp-history-stat-value {
    color: #334056;
    font-size: .88rem;
    font-weight: 780;
}

.pp-history-similarity {
    color: var(--accent);
}

.pp-chart-shell {
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .55rem .7rem .2rem .7rem;
    background: #fff;
    margin-top: .4rem;
}

/* ---------- explore cards ---------- */
.pp-card-title {
    color: var(--text);
    font-weight: 760;
    font-size: .96rem;
    line-height: 1.45;
    margin-bottom: .15rem;
}

.pp-card-copy,
.pp-example-cta-hint {
    color: var(--muted);
    font-size: .78rem;
}

.pp-tag {
    display: inline-flex;
    align-items: center;
    padding: .22rem .48rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: #647087;
    font-size: .68rem;
    font-weight: 740;
    letter-spacing: .02em;
}

/* ---------- tables / expanders ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 11px;
    overflow: hidden;
    background: #fff;
}

details {
    background: #fff !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
}

.stAlert {
    border-radius: 10px;
}

/* ---------- footer ---------- */
.pp-footer {
    color: #909bad;
    margin-top: 2.6rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-size: .78rem;
    text-align: center;
}

@media (max-width: 820px) {
    .pp-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .pp-kpi:nth-child(3) {
        border-left: 0;
        padding-left: 0;
        border-top: 1px solid var(--line);
        margin-top: .8rem;
        padding-top: .8rem;
    }

    .pp-kpi:nth-child(4) {
        border-top: 1px solid var(--line);
        margin-top: .8rem;
        padding-top: .8rem;
    }

    .pp-history-row {
        grid-template-columns: 38px minmax(0, 1fr);
    }

    .pp-history-row > div:nth-child(3),
    .pp-history-row > div:nth-child(4) {
        grid-column: 2;
    }
}

@media (max-width: 640px) {
    .pp-title {
        font-size: 2.1rem;
    }

    .pp-kpi-grid {
        grid-template-columns: 1fr;
    }

    .pp-kpi + .pp-kpi {
        border-left: 0;
        border-top: 1px solid var(--line);
        padding-left: 0;
        margin-top: .7rem;
        padding-top: .75rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    html = (
        '<div class="pp-hero">'
        '<div class="pp-brandline">'
        '<span class="pp-brandmark">P</span>'
        '<span class="pp-brandname">PolyPulse</span>'
        '</div>'
        '<div class="pp-kicker">Pre-launch market intelligence</div>'
        '<div class="pp-title">Understand expected demand before a prediction market opens.</div>'
        '<div class="pp-subtitle">'
        'Validate the proposition, estimate demand, and compare it with historical '
        'Polymarket markets in one focused workflow.'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_label(text):
    st.markdown(
        f'<div class="pp-section-label">{escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached model/resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def validity_resources():
    """Cache the fine-tuned binary question-validity model for this app process."""
    return load_validator()


@st.cache_resource(show_spinner=False)
def classifier_resources():
    """Cache the fine-tuned DeBERTa demand model and its published config."""
    classifier = load_classifier()
    config = load_published_config()
    return classifier, config


@st.cache_resource(show_spinner=False)
def historical_resources():
    """Cache historical records, precomputed embeddings, and MiniLM."""
    markets, embeddings, metadata = load_reference_data(str(DATA_DIR))
    extractor = get_extractor(
        metadata["model_id"], metadata.get("model_revision"), device=-1
    )
    return markets, embeddings, metadata, extractor


# ---------------------------------------------------------------------------
# Analysis orchestration
# ---------------------------------------------------------------------------

def analyse(question, title):
    """Validate first; only valid questions proceed to demand and history."""
    results = {
        "question": question,
        "event_title": title,
        "validity": None,
        "demand": None,
        "history": None,
        "validity_error": None,
        "demand_error": None,
        "history_error": None,
        "validity_seconds": None,
        "demand_seconds": None,
        "history_seconds": None,
        "history_index_count": None,
    }

    try:
        with st.spinner("Checking question validity…"):
            validator = validity_resources()
            start = perf_counter()
            results["validity"] = validate_market_question(
                question, title, validator=validator
            )
            results["validity_seconds"] = perf_counter() - start
    except Exception as exc:
        results["validity_error"] = f"{type(exc).__name__}: {exc}"
        return results

    if not results["validity"]["is_valid"]:
        return results

    try:
        with st.spinner("Predicting expected market demand…"):
            classifier, config = classifier_resources()
            start = perf_counter()
            results["demand"] = predict_demand(
                question, title, classifier=classifier, config=config
            )
            results["demand_seconds"] = perf_counter() - start
    except Exception as exc:
        results["demand_error"] = f"{type(exc).__name__}: {exc}"

    try:
        with st.spinner("Finding historical comparables…"):
            _, _, metadata, _ = historical_resources()
            results["history_index_count"] = int(metadata["record_count"])

            min_similarity = float(
                metadata.get(
                    "min_similarity_default",
                    metadata.get("default_min_similarity", 0.52),
                )
            )
            min_comparables = int(
                metadata.get(
                    "min_comparables_default",
                    metadata.get("default_min_comparables", 5),
                )
            )
            stats_top_n = max(
                10,
                min_comparables,
                int(metadata.get("statistics_top_n_default", 30)),
            )

            start = perf_counter()
            results["history"] = find_similar_markets(
                question,
                event_title=title,
                data_dir=str(DATA_DIR),
                top_k=10,
                stats_top_n=stats_top_n,
                min_similarity=min_similarity,
                min_comparables=min_comparables,
                include_topic=True,
            )
            results["history_seconds"] = perf_counter() - start
    except Exception as exc:
        results["history_error"] = f"{type(exc).__name__}: {exc}"

    return results


def money(value):
    return f"${float(value):,.0f}"


def compact_volume_band(value):
    """Shorten historical-band copy so it remains readable in a KPI card."""
    if not value:
        return "Not published"

    value = str(value).strip()

    if value.lower().startswith("below "):
        return "< " + value[6:].strip()

    if value.lower().endswith(" or above"):
        return "≥ " + value[:-9].strip()

    if " to " in value:
        left, right = value.split(" to ", 1)
        return f"{left.strip()}–{right.strip()}"

    return value


def topic_from_history(history):
    if not isinstance(history, dict):
        return "—"
    topic = history.get("topic_suggestion")
    if isinstance(topic, dict) and topic.get("topic"):
        return str(topic["topic"])
    return "—"


def comparable_count(history):
    if not isinstance(history, dict):
        return None
    value = history.get("n_comparables")
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# Results UI
# ---------------------------------------------------------------------------

def show_result_header(results):
    """Render the analysis snapshot as safe inline HTML, not indented Markdown."""
    demand = results.get("demand")
    history = results.get("history")

    demand_label = demand["label"] if demand else "Unavailable"
    demand_value = demand_label.upper() if demand else "Unavailable"
    demand_class = (
        f"pp-demand-{demand_label.lower()}"
        if demand_label in {"Low", "Medium", "High"}
        else ""
    )

    volume_band = compact_volume_band(
        demand.get("volume_band") if demand else None
    )
    n_comp = comparable_count(history)
    topic = topic_from_history(history)
    comparables = f"{n_comp:,}" if n_comp is not None else "—"

    event_text = (
        f"Event: {results['event_title']}"
        if results.get("event_title")
        else "No event title provided"
    )

    html = (
        '<div class="pp-analysis-shell">'
        '<div class="pp-analysis-topline">'
        '<span class="pp-analysis-badge">Analysis snapshot</span>'
        '</div>'
        f'<div class="pp-analysis-question">{escape(str(results["question"]))}</div>'
        f'<div class="pp-analysis-event">{escape(event_text)}</div>'
        '<div class="pp-kpi-grid">'
        '<div class="pp-kpi">'
        '<div class="pp-kpi-label">Expected demand</div>'
        f'<div class="pp-kpi-value {demand_class}">{escape(demand_value)}</div>'
        '</div>'
        '<div class="pp-kpi">'
        '<div class="pp-kpi-label">Historical volume band</div>'
        f'<div class="pp-kpi-value">{escape(volume_band)}</div>'
        '</div>'
        '<div class="pp-kpi">'
        '<div class="pp-kpi-label">Comparable markets</div>'
        f'<div class="pp-kpi-value">{escape(comparables)}</div>'
        '</div>'
        '<div class="pp-kpi">'
        '<div class="pp-kpi-label">Suggested topic</div>'
        f'<div class="pp-kpi-value">{escape(topic)}</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_demand(results, show_title=True):
    if show_title:
        st.subheader("Demand prediction")

    pred = results.get("demand")
    if pred is None:
        st.warning(
            "Demand prediction is currently unavailable. Historical evidence may still be available."
        )
        if results.get("demand_error"):
            with st.expander("Technical details"):
                st.code(results["demand_error"])
        return

    label = pred["label"]
    icon = DEMAND_ICONS.get(label, "📊")
    top_score = float(pred["model_scores"].get(label, 0.0))

    headline_col, band_col = st.columns(2)
    headline_col.metric(
        f"{icon} Predicted demand",
        label.upper(),
    )
    band_col.metric(
        "Historical demand band",
        pred.get("volume_band") or "Not published",
    )

    st.markdown("#### Model score distribution")
    for class_label in LABELS:
        score = float(pred["model_scores"][class_label])
        prefix = "●" if class_label == label else "○"
        st.progress(
            min(max(score, 0.0), 1.0),
            text=f"{prefix} {class_label}: {score:.1%}",
        )

    adjustment_note = (
        "Published validation decision adjustments are applied."
        if pred.get("adjustments_available")
        else "No published validation adjustment is applied."
    )
    elapsed = results.get("demand_seconds")
    runtime = f" Inference: {elapsed:.2f}s." if elapsed is not None else ""

    st.caption(
        f"Top model score: {top_score:.1%}. Scores are classification scores, "
        f"not calibrated probabilities. {adjustment_note}{runtime}"
    )


def show_history(results, show_title=True):
    if show_title:
        st.subheader("Historical evidence")

    history = results.get("history")
    if history is None:
        st.warning(
            "Historical comparison is currently unavailable. Demand prediction may still be shown."
        )
        if results.get("history_error"):
            with st.expander("Technical details"):
                st.code(results["history_error"])
        return

    matches = history.get("matches") or []
    displayed_count = len(matches)
    stats_count = int(history.get("n_comparables", displayed_count))
    total_qualified = int(history.get("n_above_threshold", stats_count))
    threshold = float(history.get("minimum_similarity", 0.52))
    index_count = results.get(
        "history_index_count",
        history.get("n_reference_markets"),
    )
    topic = topic_from_history(history)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Suggested topic", topic)
    col2.metric("Comparables used", f"{stats_count:,}")
    col3.metric("Similarity threshold", f"{threshold:.0%}")
    col4.metric(
        "Indexed markets",
        f"{int(index_count):,}" if index_count is not None else "—",
    )

    if not matches:
        st.info(
            history.get("message")
            or "No historical markets meet the configured similarity threshold."
        )
        return

    # ------------------------------------------------------------------
    # Closest historical markets: compact ranked cards instead of a dense table.
    # ------------------------------------------------------------------
    st.markdown("#### Closest historical markets")
    st.caption(
        "The strongest semantic matches to the proposed market, ranked by cosine similarity."
    )

    top_preview = matches[:5]
    rows_html = []

    for rank, market in enumerate(top_preview, start=1):
        question = escape(str(market.get("question", "")))
        similarity = 100 * float(market.get("similarity", 0.0))
        volume = escape(money(market.get("volume_usd", 0.0)))
        topic_name = clean_text(market.get("topic", ""))
        event_title = clean_text(market.get("event_title", ""))

        secondary = []
        if topic_name:
            secondary.append(topic_name)
        if event_title:
            secondary.append(event_title)
        meta = " · ".join(secondary) if secondary else "Historical market"

        rows_html.append(
            '<div class="pp-history-row">'
            f'<div class="pp-history-rank">#{rank}</div>'
            '<div>'
            f'<div class="pp-history-question">{question}</div>'
            f'<div class="pp-history-meta">{escape(meta)}</div>'
            '</div>'
            '<div>'
            '<div class="pp-history-stat-label">Similarity</div>'
            f'<div class="pp-history-stat-value pp-history-similarity">{similarity:.1f}%</div>'
            '</div>'
            '<div>'
            '<div class="pp-history-stat-label">Observed volume</div>'
            f'<div class="pp-history-stat-value">{volume}</div>'
            '</div>'
            '</div>'
        )

    st.markdown(
        '<div class="pp-history-list">' + "".join(rows_html) + "</div>",
        unsafe_allow_html=True,
    )

    all_match_rows = []
    for rank, market in enumerate(matches, start=1):
        all_match_rows.append(
            {
                "Rank": rank,
                "Historical question": market["question"],
                "Similarity": 100 * float(market["similarity"]),
                "Observed volume (USD)": float(market["volume_usd"]),
                "Topic": clean_text(market.get("topic", "")),
            }
        )

    with st.expander(
        f"View all {displayed_count} displayed comparable markets",
        expanded=False,
    ):
        st.dataframe(
            pd.DataFrame(all_match_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank",
                    format="#%d",
                    width="small",
                ),
                "Historical question": st.column_config.TextColumn(
                    "Historical question",
                    width="large",
                ),
                "Similarity": st.column_config.NumberColumn(
                    "Similarity",
                    format="%.1f%%",
                    help="Cosine similarity to the proposed market.",
                    step=0.1,
                ),
                "Observed volume (USD)": st.column_config.NumberColumn(
                    "Observed volume",
                    format="$%.0f",
                ),
                "Topic": st.column_config.TextColumn(
                    "Topic",
                    width="small",
                ),
            },
        )

    elapsed = results.get("history_seconds")
    detail_text = (
        f"Showing the top {displayed_count} of {total_qualified:,} qualifying historical markets."
    )
    if elapsed is not None:
        detail_text += f" Search inference: {elapsed:.2f}s."
    st.caption(detail_text)

    # ------------------------------------------------------------------
    # Historical trading-volume range.
    # ------------------------------------------------------------------
    st.markdown("#### Historical trading-volume range")
    if history.get("status") != "ok":
        st.info(
            history.get("message")
            or "There are too few comparable markets to report a historical volume range."
        )
        return

    vol1, vol2, vol3 = st.columns(3)
    vol1.metric("Median observed volume", money(history["median_volume_usd"]))
    vol2.metric("25th percentile", money(history["p25_volume_usd"]))
    vol3.metric("75th percentile", money(history["p75_volume_usd"]))

    chart_df = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Market": f"#{rank}",
                "Question": str(market.get("question", "")),
                "Observed volume (USD)": float(market.get("volume_usd", 0.0)),
                "Similarity": 100 * float(market.get("similarity", 0.0)),
            }
            for rank, market in enumerate(matches, start=1)
        ]
    )

    base = alt.Chart(chart_df).encode(
        y=alt.Y(
            "Market:N",
            sort=[f"#{i}" for i in range(1, displayed_count + 1)],
            title="Comparable market rank",
            axis=alt.Axis(
                labelColor="#526078",
                titleColor="#334155",
                labelPadding=8,
                ticks=False,
                domain=False,
            ),
        ),
        x=alt.X(
            "Observed volume (USD):Q",
            title="Observed USD volume",
            axis=alt.Axis(
                format="$,.0f",
                labelColor="#526078",
                titleColor="#334155",
                gridColor="#e8edf4",
                domain=False,
            ),
        ),
        tooltip=[
            alt.Tooltip("Market:N", title="Rank"),
            alt.Tooltip("Question:N", title="Historical question"),
            alt.Tooltip(
                "Observed volume (USD):Q",
                title="Observed volume",
                format="$,.0f",
            ),
            alt.Tooltip("Similarity:Q", title="Similarity", format=".1f"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
        size=24,
        color="#6aaee8",
    )

    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        color="#334155",
        fontSize=11,
    ).encode(
        text=alt.Text(
            "Observed volume (USD):Q",
            format="$,.0f",
        )
    )

    chart = (
        (bars + labels)
        .properties(
            height=max(300, 38 * displayed_count),
            background="#ffffff",
        )
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="sans-serif",
            titleFont="sans-serif",
            titleFontWeight=600,
        )
    )

    st.markdown('<div class="pp-chart-shell">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        f"Bars show observed USD volume for the top {displayed_count} displayed matches. "
        f"Percentiles are calculated from the {stats_count} nearest qualified markets. "
        "Historical volumes are observations, not forecasts."
    )



def show_summary(results, show_title=True):
    if show_title:
        st.subheader("Executive summary")

    demand = results.get("demand")
    history = results.get("history")

    if demand and history and history.get("status") == "ok":
        label = demand["label"].upper()
        headline = (
            f"PolyPulse indicates {label} expected demand for this proposed market."
        )
        copy = (
            f"The {int(history['n_comparables']):,} nearest qualified historical markets "
            f"had a median observed trading volume of "
            f'<span class="pp-summary-stat">{escape(money(history["median_volume_usd"]))}</span>. '
            f"The middle 50% ranged from "
            f'<span class="pp-summary-stat">{escape(money(history["p25_volume_usd"]))}</span> '
            f"to "
            f'<span class="pp-summary-stat">{escape(money(history["p75_volume_usd"]))}</span>.'
        )
    elif demand and history:
        headline = (
            f"PolyPulse indicates {demand['label'].upper()} expected demand."
        )
        copy = (
            "Historical retrieval found too few sufficiently similar markets "
            "to report a reliable descriptive volume range."
        )
    elif demand:
        headline = (
            f"PolyPulse indicates {demand['label'].upper()} expected demand."
        )
        copy = "Historical evidence is currently unavailable."
    elif history and history.get("status") == "ok":
        headline = (
            "Historical evidence is available, but demand classification is unavailable."
        )
        copy = (
            f"The {int(history['n_comparables']):,} comparable markets had a median "
            f"observed volume of "
            f'<span class="pp-summary-stat">{escape(money(history["median_volume_usd"]))}</span>.'
        )
    elif history:
        headline = "The market could not be fully assessed."
        copy = (
            "Demand classification is unavailable and there are too few historical "
            "comparables to report a volume range."
        )
    else:
        headline = "No analysis result is currently available."
        copy = "Please run the market analysis again."

    html = (
        '<div class="pp-summary-card">'
        f'<div class="pp-summary-headline">{escape(headline)}</div>'
        f'<div class="pp-summary-copy">{copy}</div>'
        '<div class="pp-summary-note">'
        'Decision-support only. PolyPulse estimates market demand; it does not '
        'predict the event outcome or recommend launching a market.'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_analysis_results(result):
    show_result_header(result)

    overview_tab, demand_tab, history_tab = st.tabs(
        ["Overview", "Demand", "Historical evidence"]
    )

    with overview_tab:
        with st.container(border=True):
            section_label("Executive view")
            show_summary(result, show_title=False)

    with demand_tab:
        with st.container(border=True):
            section_label("Forecast")
            show_demand(result, show_title=False)

    with history_tab:
        with st.container(border=True):
            section_label("Evidence")
            show_history(result, show_title=False)


# ---------------------------------------------------------------------------
# Popular questions
# ---------------------------------------------------------------------------

def popular_questions_table(
    markets,
    embeddings,
    min_similarity=0.52,
    min_comparables=5,
):
    """Find real, high-volume reference questions per indexed topic."""
    columns = [
        "Category",
        "Market question",
        "Event title",
        "Similar markets",
        "Median historical volume (USD)",
    ]
    if markets.empty:
        return pd.DataFrame(columns=columns)

    topics = (
        markets["topic"].fillna("Other").astype(str)
        if "topic" in markets.columns
        else pd.Series("Other", index=markets.index)
    )

    candidates = pd.DataFrame(
        {
            "index": np.arange(len(markets)),
            "Category": topics.to_numpy(),
            "Volume": markets["volume_usd"].to_numpy(dtype=float),
        }
    )
    candidates = (
        candidates.sort_values("Volume", ascending=False, kind="stable")
        .groupby("Category", sort=True)
        .head(2)
    )

    rows = []
    for _, candidate in candidates.iterrows():
        idx = int(candidate["index"])
        query_vector = np.asarray(embeddings[idx], dtype=np.float32)
        similarities = np.asarray(
            embeddings @ query_vector,
            dtype=np.float32,
        )
        qualified = np.flatnonzero(similarities >= min_similarity)
        qualified = qualified[qualified != idx]

        if len(qualified) < min_comparables:
            continue

        nearest = qualified[
            np.argsort(-similarities[qualified], kind="stable")[:10]
        ]
        volumes = markets.iloc[nearest]["volume_usd"].to_numpy(dtype=float)

        rows.append(
            {
                "Category": str(candidate["Category"]),
                "Market question": str(markets.iloc[idx]["question"]),
                "Event title": clean_text(
                    markets.iloc[idx].get("event_title", "")
                ),
                "Similar markets": int(len(qualified)),
                "Median historical volume (USD)": float(np.median(volumes)),
            }
        )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["Similar markets", "Market question"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_popular_questions():
    """Load historical data/vectors without downloading the HF pipelines."""
    markets, embeddings, metadata = load_reference_data(str(DATA_DIR))
    table = popular_questions_table(
        markets,
        embeddings,
        min_similarity=float(
            metadata.get(
                "min_similarity_default",
                metadata.get("default_min_similarity", 0.52),
            )
        ),
        min_comparables=int(
            metadata.get(
                "min_comparables_default",
                metadata.get("default_min_comparables", 5),
            )
        ),
    )
    return (
        table,
        int(metadata["record_count"]),
        float(
            metadata.get(
                "min_similarity_default",
                metadata.get("default_min_similarity", 0.52),
            )
        ),
    )


def load_example_into_form(question, event_title):
    """Load a historical example and navigate directly to Market Analysis."""
    st.session_state["market_question"] = str(question)
    st.session_state["market_event_title"] = str(event_title or "")
    st.session_state.pop("polypulse_analysis", None)
    st.session_state["top_nav"] = "Market Analysis"
    st.session_state["example_loaded_notice"] = True


def show_popular_questions():
    section_label("Explore")
    st.subheader("Explore historical markets")
    st.markdown(
        '<div class="pp-section-intro">'
        'Browse real historical markets with strong coverage in the reference index. '
        'Choose one to load it directly into Market Analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading historical examples…"):
            table, total_markets, similarity_threshold = load_popular_questions()
    except Exception as exc:
        st.warning("Historical examples are currently unavailable.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    met1, met2, met3 = st.columns(3)
    met1.metric("🗂️ Indexed markets", f"{total_markets:,}")
    met2.metric("✨ Ready-to-try examples", len(table))
    met3.metric("🎯 Similarity floor", f"{similarity_threshold:.0%}")

    if table.empty:
        st.info("No sample questions meet the current historical-coverage rules.")
        return

    raw_categories = list(dict.fromkeys(table["Category"]))
    category_options = {"All categories": None}
    for category in raw_categories:
        icon = CATEGORY_ICONS.get(str(category), "📌")
        category_options[str(category)] = category

    selected_label = st.selectbox(
        "Filter by category",
        list(category_options),
        key="popular_category_filter",
    )
    selected_category = category_options[selected_label]
    filtered = (
        table
        if selected_category is None
        else table.loc[table["Category"] == selected_category]
    )

    st.markdown("#### Featured examples")
    preview = filtered.head(6).reset_index(drop=True)
    card_columns = st.columns(2)

    for idx, row in preview.iterrows():
        with card_columns[idx % 2]:
            with st.container(border=True):
                category = str(row["Category"])
                icon = CATEGORY_ICONS.get(category, "📌")
                st.markdown(
                    f'<span class="pp-tag">{escape(category)}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="pp-card-title">{escape(str(row["Market question"]))}</div>',
                    unsafe_allow_html=True,
                )
                if clean_text(row["Event title"]):
                    st.caption(f"Event: {row['Event title']}")

                c1, c2 = st.columns(2)
                c1.metric(
                    "Similar markets",
                    f"{int(row['Similar markets']):,}",
                )
                c2.metric(
                    "Median volume",
                    money(row["Median historical volume (USD)"]),
                )

                st.markdown(
                    '<div class="pp-example-cta-hint">Load this question into the analysis form</div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Use in Market Analysis",
                    key=f"use_example_{selected_category}_{idx}_{row['Market question'][:30]}",
                    type="primary",
                    use_container_width=True,
                    on_click=load_example_into_form,
                    args=(row["Market question"], row["Event title"]),
                )

    with st.expander("View all matching examples"):
        display_table = filtered.copy()
        display_table["Category"] = display_table["Category"].map(
            lambda value: f"{CATEGORY_ICONS.get(str(value), '📌')} {value}"
        )
        st.dataframe(
            display_table.drop(columns=["Event title"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Category": st.column_config.TextColumn(
                    "Category",
                    width="medium",
                ),
                "Market question": st.column_config.TextColumn(
                    "Market question",
                    width="large",
                ),
                "Similar markets": st.column_config.NumberColumn(
                    "Similar markets",
                    format="%d",
                    help="Other indexed markets above the similarity threshold.",
                ),
                "Median historical volume (USD)": st.column_config.NumberColumn(
                    "Median historical volume",
                    format="$%.0f",
                ),
            },
        )

    st.caption(
        "Examples come directly from the historical index. 'Similar markets' measures "
        "semantic coverage, not search popularity."
    )


# ---------------------------------------------------------------------------
# Model stack
# ---------------------------------------------------------------------------

def model_card(label, icon, title, role, model_name):
    with st.container(border=True):
        section_label(label)
        st.markdown(f"### {title}")
        st.write(role)
        st.caption(model_name)


def show_performance():
    section_label("Architecture")
    st.subheader("Model architecture")
    st.markdown(
        '<div class="pp-section-intro">'
        'Three specialised models handle validation, demand classification and historical '
        'retrieval independently, keeping the workflow interpretable and modular.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_card(
            "Stage 1 · Model 3",
            "",
            "Question validity",
            "Rejects inputs that are not clear, resolvable prediction-market propositions.",
            "2610gnam05/nli-deberta-v3-small-demand",
        )
    with c2:
        model_card(
            "Stage 2 · Model 1",
            "",
            "Demand prediction",
            "Classifies expected pre-launch demand as Low, Medium or High.",
            "2610gnam05/polypulse-deberta-v3-base-demand",
        )
    with c3:
        model_card(
            "Stage 3 · Model 2",
            "",
            "Historical retrieval",
            "Finds semantically similar historical markets and observed trading volumes.",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    st.markdown("#### Pipeline")
    st.info(
        "Market question  →  Validity gate  →  Demand prediction  →  Historical evidence"
    )

    with st.expander("Model 1 · Demand prediction"):
        st.write(
            "**Fine-tuned model:** "
            "[DeBERTa-v3-base demand classifier]"
            "(https://huggingface.co/2610gnam05/polypulse-deberta-v3-base-demand)"
        )
        st.write(
            "**Pipeline:** `text-classification` · CPU · Low / Medium / High"
        )
        if COMPARISON_CSV.exists():
            comparison = pd.read_csv(COMPARISON_CSV, dtype=str).fillna("")
            if (
                not comparison.empty
                and "test_accuracy" in comparison.columns
                and comparison["test_accuracy"].str.strip().any()
            ):
                st.dataframe(
                    comparison,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption(
                    "The model-comparison file is present but final test metrics are not populated."
                )

    with st.expander("Model 2 · Historical retrieval"):
        st.write(
            "**Pretrained model:** "
            "[all-MiniLM-L6-v2]"
            "(https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)"
        )
        st.write(
            "**Pipeline:** `feature-extraction` · CPU · 384-dimensional embeddings"
        )
        try:
            import json

            with open(DATA_DIR / "metadata.json", encoding="utf-8") as handle:
                meta = json.load(handle)

            m1, m2 = st.columns(2)
            m1.metric("Indexed historical markets", f"{meta['record_count']:,}")
            m2.metric(
                "Reference cutoff",
                str(meta.get("reference_cutoff_utc", "Unavailable")),
            )
        except Exception as exc:
            st.caption(f"Historical metadata unavailable: {exc}")

    with st.expander("Model 3 · Question validity"):
        st.write(
            f"**Fine-tuned model:** [nli-deberta-v3-small-demand]"
            f"(https://huggingface.co/{VALIDATOR_MODEL_ID})"
        )
        st.write(
            "**Base model:** `cross-encoder/nli-deberta-v3-small` · "
            "binary `INVALID` / `VALID` classification · CPU"
        )
        st.caption(
            "This model executes first at runtime. Only questions passing the published "
            "validation threshold continue to Model 1 and Model 2."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

inject_styles()
render_hero()

if st.session_state.pop("example_loaded_notice", False):
    st.toast("Example loaded in Market Analysis.")

NAV_OPTIONS = [
    "Market Analysis",
    "Explore Questions",
    "Model Stack",
]

if "top_nav" not in st.session_state:
    st.session_state["top_nav"] = NAV_OPTIONS[0]

current_page = st.radio(
    "Navigation",
    NAV_OPTIONS,
    key="top_nav",
    horizontal=True,
    label_visibility="collapsed",
)

if current_page == "Market Analysis":
    section_label("Analyse")
    st.subheader("Analyse a proposed market")
    st.markdown(
        '<div class="pp-section-intro">'
        'Enter the proposition as it would appear to users. PolyPulse validates it first, '
        'then runs demand prediction and historical comparison.'
        '</div>',
        unsafe_allow_html=True,
    )

    def clear_previous_analysis():
        st.session_state.pop("polypulse_analysis", None)

    def clear_everything():
        clear_previous_analysis()
        st.session_state["market_question"] = ""
        st.session_state["market_event_title"] = ""

    with st.container(border=True):
        question = st.text_area(
            "Market question",
            placeholder="Will Bitcoin exceed $150,000 by December 2027?",
            max_chars=1200,
            height=105,
            key="market_question",
            help=(
                "Use a clear prediction-market proposition with an outcome that can "
                "be objectively resolved."
            ),
        )

        event_title = st.text_input(
            "Event title",
            placeholder="Optional — e.g. Bitcoin Price Predictions",
            max_chars=300,
            key="market_event_title",
            help="Optional context. The market question remains the primary input.",
        )

        with st.expander("What makes a good market question?"):
            st.write(
                "A strong input identifies a specific future event or measurable outcome "
                "and is written so the final result can be objectively determined."
            )
            st.caption(
                "Example: “Will Bitcoin exceed $150,000 by December 2027?”"
            )

        analyse_col, clear_col, spacer_col = st.columns([2, 1, 4])
        with analyse_col:
            submitted = st.button(
                "Analyse Market",
                type="primary",
                use_container_width=True,
                on_click=clear_previous_analysis,
            )
        with clear_col:
            st.button(
                "Clear",
                use_container_width=True,
                on_click=clear_everything,
            )

    results_placeholder = st.empty()

    if submitted:
        results_placeholder.empty()

        if not question.strip():
            st.error("Enter a market question before analysing.")
        else:
            st.session_state["polypulse_analysis"] = analyse(
                question.strip(),
                event_title.strip(),
            )

    if "polypulse_analysis" in st.session_state:
        result = st.session_state["polypulse_analysis"]

        with results_placeholder.container():
            validity = result.get("validity")

            if validity is None:
                st.error(
                    "Question validation is temporarily unavailable. Please try again."
                )
            elif not validity.get("is_valid", False):
                st.error(
                    "Please enter a clear prediction-market proposition, "
                    "with an outcome that can be objectively resolved."
                )
            else:
                section_label("Results")
                st.markdown("## Analysis")
                show_analysis_results(result)

elif current_page == "Explore Questions":
    show_popular_questions()

elif current_page == "Model Stack":
    show_performance()

st.markdown(
    """
    <div class="pp-footer">
        PolyPulse · Pre-launch prediction-market demand intelligence ·
        Historical evidence is descriptive, not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
