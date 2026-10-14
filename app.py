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
    """Light, high-contrast visual system with clear tab and section separation."""
    st.markdown(
        """
<style>
:root {
    --pp-bg: #f4f7fb;
    --pp-surface: #ffffff;
    --pp-surface-soft: #f8faff;
    --pp-border: #d9e2ef;
    --pp-border-strong: #c6d3e4;
    --pp-text: #182235;
    --pp-muted: #677489;
    --pp-primary: #3f5bd8;
    --pp-primary-dark: #2f46b4;
    --pp-primary-soft: #eef2ff;
    --pp-blue-soft: #edf7ff;
    --pp-success: #16794c;
    --pp-warning: #a96800;
    --pp-danger: #b73838;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--pp-bg) !important;
    color: var(--pp-text) !important;
}

[data-testid="stHeader"] {
    background: rgba(244, 247, 251, 0.94) !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    color: var(--pp-text) !important;
}

/* ------------------------------------------------------------------
   Text visibility guardrails.
   Streamlit/BaseWeb can retain dark-theme text colors inside nested
   components even after the page background is forced to light mode.
   These explicit selectors keep every user-facing text element readable.
   ------------------------------------------------------------------ */

.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] em,
.stApp [data-testid="stMarkdownContainer"] code,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp label,
.stApp summary,
.stApp details,
.stApp [role="radiogroup"],
.stApp [role="option"],
.stApp [data-baseweb="select"],
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="popover"] *,
.stApp [data-testid="stExpander"] *,
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stToast"] * {
    color: var(--pp-text) !important;
}

/* Captions and secondary explanatory copy remain visibly muted, never hidden. */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp small {
    color: var(--pp-muted) !important;
    opacity: 1 !important;
}

/* Form control text, placeholders and dropdown values. */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea,
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] div {
    color: var(--pp-text) !important;
    -webkit-text-fill-color: var(--pp-text) !important;
}

.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #8b98ac !important;
    -webkit-text-fill-color: #8b98ac !important;
    opacity: 1 !important;
}

/* Buttons: force nested text/icon visibility as BaseWeb may style descendants. */
.stApp div.stButton > button *,
.stApp button[kind="secondary"] *,
.stApp button[kind="tertiary"] * {
    color: #243047 !important;
    opacity: 1 !important;
}

.stApp div.stButton > button[kind="primary"] *,
.stApp button[kind="primary"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Tabs: explicitly style every descendant so titles such as
   "Explore Questions" stay visible in both selected and unselected states. */
.stApp button[data-baseweb="tab"],
.stApp button[data-baseweb="tab"] *,
.stApp button[data-baseweb="tab"] p,
.stApp button[data-baseweb="tab"] span,
.stApp button[data-baseweb="tab"] div {
    color: #4d5b70 !important;
    -webkit-text-fill-color: #4d5b70 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.stApp button[data-baseweb="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"] *,
.stApp button[data-baseweb="tab"][aria-selected="true"] p,
.stApp button[data-baseweb="tab"][aria-selected="true"] span,
.stApp button[data-baseweb="tab"][aria-selected="true"] div {
    color: #263fbc !important;
    -webkit-text-fill-color: #263fbc !important;
    opacity: 1 !important;
}

/* Expander summary and alert text can otherwise inherit theme colors. */
.stApp details summary,
.stApp details summary *,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary * {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
}


/* Top-level app navigation: session-state controlled radio styled as tabs.
   Unlike st.tabs, this can be changed programmatically from a button callback. */
div[data-testid="stRadio"] > label {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    padding: 6px !important;
    margin: 0.55rem 0 1.15rem 0 !important;
    border: 1px solid #d4deec !important;
    border-radius: 14px !important;
    background: #eaf0f8 !important;
    overflow-x: auto !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    flex: 0 0 auto !important;
    align-items: center !important;
    min-height: 42px !important;
    padding: 0.48rem 1rem !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    cursor: pointer !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #4d5b70 !important;
    font-weight: 760 !important;
    white-space: nowrap !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: #ffffff !important;
    border-color: #d7e0ed !important;
    box-shadow: 0 2px 9px rgba(45, 65, 116, 0.10) !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #263fbc !important;
}

/* Hide native radio circles inside top navigation only. */
div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Example-market CTA: make the action clearly look like a button. */
.pp-example-cta-hint {
    color: #68758a;
    font-size: 0.78rem;
    margin: 0.25rem 0 0.45rem 0;
}


/* Metrics */
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stMetricDelta"] * {
    color: #68758a !important;
    opacity: 1 !important;
}

.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] * {
    color: #1e293b !important;
    opacity: 1 !important;
}

/* Links remain clearly identifiable on light surfaces. */
.stApp a,
.stApp a:visited {
    color: #3156c8 !important;
    text-decoration-color: rgba(49, 86, 200, 0.35) !important;
}

.block-container {
    max-width: 1220px;
    padding-top: 1.7rem;
    padding-bottom: 3.5rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #182235 !important;
}

.stApp p,
.stApp label {
    color: #334155;
}

p, label, span, div {
    color: inherit;
}

/* Hero */
.pp-hero {
    border: 1px solid #d7e1f2;
    border-radius: 24px;
    padding: 28px 30px 26px 30px;
    margin-bottom: 1.15rem;
    background:
        radial-gradient(circle at 90% 10%, rgba(91, 103, 255, 0.15), transparent 32%),
        radial-gradient(circle at 8% 100%, rgba(56, 173, 255, 0.12), transparent 34%),
        linear-gradient(145deg, #ffffff, #f7f9ff);
    box-shadow: 0 12px 34px rgba(42, 57, 89, 0.07);
}

.pp-kicker {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #5d6a80;
    margin-bottom: 0.45rem;
}

.pp-title {
    color: #16213a;
    font-size: clamp(2.15rem, 5vw, 3.55rem);
    line-height: 1.02;
    font-weight: 820;
    letter-spacing: -0.045em;
    margin: 0;
}

.pp-subtitle {
    max-width: 780px;
    color: #5c697e;
    font-size: 1.03rem;
    line-height: 1.58;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

.pp-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.pp-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.34rem;
    border: 1px solid #d9e2ef;
    border-radius: 999px;
    padding: 0.38rem 0.68rem;
    font-size: 0.82rem;
    font-weight: 700;
    color: #48566d;
    background: #ffffff;
}

/* Section labels */
.pp-section-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #77839a;
    margin: 0.15rem 0 0.22rem 0;
}

/* Native bordered Streamlit containers become clear white cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--pp-surface) !important;
    border: 1px solid var(--pp-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 5px 18px rgba(42, 57, 89, 0.04);
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
textarea,
input {
    background: #ffffff !important;
    color: #182235 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-color: #cfd9e7 !important;
}

textarea::placeholder,
input::placeholder {
    color: #96a2b5 !important;
    opacity: 1 !important;
}

/* Primary/secondary buttons */
div.stButton > button {
    border-radius: 11px;
    min-height: 44px;
    font-weight: 720;
    border: 1px solid #cbd6e5;
    background: #ffffff;
    color: #243047;
}

div.stButton > button[kind="primary"] {
    background: var(--pp-primary) !important;
    border-color: var(--pp-primary) !important;
    color: #ffffff !important;
}

div.stButton > button[kind="primary"]:hover {
    background: var(--pp-primary-dark) !important;
    border-color: var(--pp-primary-dark) !important;
}

/* Tabs: visibly separated segmented controls */
div[data-baseweb="tab-list"] {
    gap: 5px !important;
    padding: 5px !important;
    margin: 0.55rem 0 1rem 0 !important;
    border: 1px solid #d4deec !important;
    border-radius: 14px !important;
    background: #eaf0f8 !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin;
}

button[data-baseweb="tab"] {
    min-height: 42px !important;
    min-width: max-content !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.05rem !important;
    color: #4d5b70 !important;
    font-weight: 720 !important;
    background: transparent !important;
    border: 0 !important;
    overflow: visible !important;
    white-space: nowrap !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #263fbc !important;
    background: #ffffff !important;
    box-shadow: 0 2px 9px rgba(45, 65, 116, 0.10) !important;
}

button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    color: inherit !important;
}

/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid #dde5f0;
    border-radius: 14px;
    padding: 0.88rem 0.95rem;
    background: #ffffff;
    min-width: 0;
}

[data-testid="stMetricLabel"] {
    color: #6c788b !important;
    font-size: 0.82rem;
}

[data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-weight: 790;
}

/* Analysis snapshot */
.pp-analysis-shell {
    border: 1px solid #cbd7f5;
    border-radius: 22px;
    padding: 24px;
    margin: 0.35rem 0 1.2rem 0;
    background:
        radial-gradient(circle at 93% 8%, rgba(88, 96, 235, 0.13), transparent 36%),
        linear-gradient(145deg, #f7f9ff, #edf3ff);
    box-shadow: 0 12px 32px rgba(55, 72, 128, 0.08);
}

.pp-analysis-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.8rem;
}

.pp-analysis-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    padding: 0.34rem 0.62rem;
    border-radius: 999px;
    border: 1px solid #c6d2f4;
    background: #ffffff;
    color: #4054b8;
    font-size: 0.75rem;
    font-weight: 820;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.pp-analysis-question {
    color: #182235;
    font-size: clamp(1.18rem, 2.2vw, 1.52rem);
    line-height: 1.35;
    font-weight: 790;
    letter-spacing: -0.018em;
    margin: 0 0 0.3rem 0;
}

.pp-analysis-event {
    color: #68758a;
    font-size: 0.89rem;
    margin-bottom: 1.2rem;
}

.pp-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.72rem;
}

.pp-kpi {
    min-width: 0;
    border: 1px solid #d7e0ef;
    border-radius: 15px;
    padding: 0.95rem 1rem;
    background: rgba(255, 255, 255, 0.90);
    box-shadow: 0 3px 10px rgba(48, 62, 102, 0.04);
}

.pp-kpi-label {
    color: #6c788b;
    font-size: 0.74rem;
    line-height: 1.25;
    margin-bottom: 0.34rem;
}

.pp-kpi-value {
    color: #1d293b;
    font-size: clamp(1.08rem, 2vw, 1.58rem);
    line-height: 1.15;
    font-weight: 810;
    letter-spacing: -0.025em;
    overflow-wrap: anywhere;
}

.pp-kpi-value.pp-demand-high { color: #c54a2e; }
.pp-kpi-value.pp-demand-medium { color: #9a6500; }
.pp-kpi-value.pp-demand-low { color: #176fb2; }

/* Overview summary */
.pp-summary-card {
    border: 1px solid #d6e0f0;
    border-radius: 16px;
    padding: 1.15rem 1.25rem;
    background: #ffffff;
    box-shadow: 0 4px 16px rgba(42, 57, 89, 0.04);
}

.pp-summary-headline {
    color: #1c2940;
    font-size: 1.02rem;
    line-height: 1.5;
    font-weight: 780;
    margin-bottom: 0.56rem;
}

.pp-summary-copy {
    color: #526078;
    font-size: 0.95rem;
    line-height: 1.64;
}

.pp-summary-stat {
    color: #253eaa;
    font-weight: 800;
}

.pp-summary-note {
    color: #7a8799;
    margin-top: 0.8rem;
    padding-top: 0.72rem;
    border-top: 1px solid #e1e8f1;
    font-size: 0.79rem;
}


/* Historical-comparable list */
.pp-history-list {
    display: flex;
    flex-direction: column;
    gap: 0.62rem;
    margin: 0.35rem 0 0.75rem 0;
}

.pp-history-row {
    display: grid;
    grid-template-columns: 54px minmax(0, 1fr) 112px 145px;
    gap: 0.9rem;
    align-items: center;
    padding: 0.88rem 0.95rem;
    border: 1px solid #dce5f0;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 2px 9px rgba(42, 57, 89, 0.035);
}

.pp-history-rank {
    width: 38px;
    height: 38px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    background: #eef2ff;
    color: #4054b8;
    font-size: 0.80rem;
    font-weight: 820;
}

.pp-history-question {
    color: #1e293b;
    font-size: 0.93rem;
    line-height: 1.4;
    font-weight: 720;
}

.pp-history-meta {
    color: #7a8799;
    font-size: 0.77rem;
    margin-top: 0.22rem;
}

.pp-history-stat-label {
    color: #8390a4;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.16rem;
}

.pp-history-stat-value {
    color: #24324a;
    font-size: 0.92rem;
    font-weight: 790;
}

.pp-history-similarity {
    color: #3156c8;
}

.pp-chart-shell {
    border: 1px solid #dce5f0;
    border-radius: 16px;
    padding: 0.75rem 0.85rem 0.35rem 0.85rem;
    background: #ffffff;
    box-shadow: 0 3px 12px rgba(42, 57, 89, 0.04);
    margin-top: 0.35rem;
}

@media (max-width: 800px) {
    .pp-history-row {
        grid-template-columns: 46px minmax(0, 1fr);
    }

    .pp-history-row > div:nth-child(3),
    .pp-history-row > div:nth-child(4) {
        grid-column: 2;
        display: inline-block;
    }
}


/* Alerts, tables, expanders */
.stAlert {
    border-radius: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #dbe4ef;
    border-radius: 13px;
    overflow: hidden;
    background: #ffffff;
}

details {
    background: #ffffff !important;
    border-color: #dbe4ef !important;
    border-radius: 12px !important;
}

/* Progress */
div[data-testid="stProgress"] > div > div {
    border-radius: 999px;
}

/* Footer */
.pp-footer {
    color: #7a8799;
    margin-top: 2.8rem;
    padding-top: 1.1rem;
    border-top: 1px solid #dbe3ed;
    font-size: 0.82rem;
    text-align: center;
}

@media (max-width: 720px) {
    .pp-hero {
        padding: 22px 20px;
        border-radius: 19px;
    }

    .pp-title {
        font-size: 2.25rem;
    }

    .pp-analysis-shell {
        padding: 18px;
        border-radius: 18px;
    }

    .pp-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .pp-kpi-value {
        font-size: 1.12rem;
    }

    button[data-baseweb="tab"] {
        padding: 0.5rem 0.62rem !important;
        font-size: 0.86rem !important;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    html = (
        '<div class="pp-hero">'
        '<div class="pp-kicker">Prediction-market intelligence</div>'
        '<div class="pp-title">PolyPulse</div>'
        '<div class="pp-subtitle">'
        'Estimate pre-launch trading demand, benchmark a proposed market against '
        'historical Polymarket observations, and surface the evidence behind the result.'
        '</div>'
        '<div class="pp-pill-row">'
        '<span class="pp-pill">🛡️ Validity first</span>'
        '<span class="pp-pill">📈 Demand forecast</span>'
        '<span class="pp-pill">🧭 Historical comparables</span>'
        '<span class="pp-pill">⚙️ Hugging Face models</span>'
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
        '<span class="pp-analysis-badge">✦ Analysis snapshot</span>'
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
        ["⚡ Overview", "📈 Demand", "🧭 Historical evidence"]
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
    st.session_state["top_nav"] = "📊 Market Analysis"
    st.session_state["example_loaded_notice"] = True


def show_popular_questions():
    section_label("Explore")
    st.subheader("Start from a real historical market")
    st.write(
        "Browse high-volume historical questions with strong coverage in the current "
        "reference index. Pick one to load it directly into Market Analysis."
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
    category_options = {"🌐 All categories": None}
    for category in raw_categories:
        icon = CATEGORY_ICONS.get(str(category), "📌")
        category_options[f"{icon} {category}"] = category

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
                st.caption(f"{icon} {category}")
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
                    "📊 Use in Market Analysis",
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
        st.markdown(f"### {icon} {title}")
        st.write(role)
        st.caption(model_name)


def show_performance():
    section_label("Architecture")
    st.subheader("Three-stage model pipeline")
    st.write(
        "PolyPulse separates input validation, demand classification and historical "
        "retrieval so that each model has one clear responsibility."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_card(
            "Stage 1 · Model 3",
            "🛡️",
            "Question validity",
            "Rejects inputs that are not clear, resolvable prediction-market propositions.",
            "2610gnam05/nli-deberta-v3-small-demand",
        )
    with c2:
        model_card(
            "Stage 2 · Model 1",
            "📈",
            "Demand prediction",
            "Classifies expected pre-launch demand as Low, Medium or High.",
            "2610gnam05/polypulse-deberta-v3-base-demand",
        )
    with c3:
        model_card(
            "Stage 3 · Model 2",
            "🧭",
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
    st.toast("Example loaded in Market Analysis.", icon="✅")

NAV_OPTIONS = [
    "📊 Market Analysis",
    "✨ Explore Questions",
    "🧠 Model Stack",
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

if current_page == "📊 Market Analysis":
    section_label("Analyse")
    st.subheader("Test a proposed prediction market")
    st.write(
        "Enter the market proposition exactly as a user would see it. PolyPulse first "
        "checks that it is a valid, resolvable prediction-market proposition."
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

elif current_page == "✨ Explore Questions":
    show_popular_questions()

elif current_page == "🧠 Model Stack":
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
