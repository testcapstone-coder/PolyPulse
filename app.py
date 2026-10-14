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
    """PolyPulse premium light UI — restrained, legible, and product-led."""
    st.markdown(
        """
<style>
:root{
    --bg:#f5f7fb;
    --surface:#ffffff;
    --surface-soft:#fafbfe;
    --text:#172033;
    --muted:#6f7b8f;
    --subtle:#96a0b1;
    --line:#e1e6ef;
    --line-strong:#d3dae6;
    --accent:#4f46e5;
    --accent-dark:#3f37c9;
    --accent-soft:#f0efff;
    --accent-softer:#f8f7ff;
    --high:#0f766e;
    --medium:#b7791f;
    --low:#64748b;
    --danger:#b42318;
    --shadow:0 12px 32px rgba(24,34,53,.055);
}

html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:var(--bg)!important;
    color:var(--text)!important;
}
[data-testid="stHeader"]{background:rgba(245,247,251,.96)!important;}
.block-container{
    max-width:1180px;
    padding-top:1.35rem;
    padding-bottom:3rem;
    font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

/* Global text safety */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6{
    color:var(--text)!important;
    letter-spacing:-.025em;
}
.stApp p,.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stWidgetLabel"] *,
.stApp details *, .stApp summary *,
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stToast"] *{
    color:var(--text)!important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *{
    color:var(--muted)!important;
    opacity:1!important;
}
.stApp a,.stApp a:visited{color:var(--accent)!important;}

/* Brand / hero */
.pp-hero{
    position:relative;
    overflow:hidden;
    border:1px solid var(--line);
    border-radius:20px;
    background:
        radial-gradient(circle at 92% 4%, rgba(79,70,229,.10), transparent 31%),
        linear-gradient(145deg,#fff 0%,#fcfcff 70%,#f7f6ff 100%);
    padding:1.45rem 1.55rem 1.5rem;
    box-shadow:0 6px 22px rgba(24,34,53,.035);
    margin-bottom:.9rem;
}
.pp-hero-top{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:1rem;
    margin-bottom:1.1rem;
}
.pp-brand{
    display:flex;
    align-items:center;
    gap:.68rem;
}
.pp-logo{
    width:34px;height:34px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    color:#fff;font-weight:850;font-size:.84rem;
    background:linear-gradient(145deg,#5b54ee,#3f37c9);
    box-shadow:0 6px 16px rgba(79,70,229,.20);
}
.pp-brand-name{
    color:var(--text);
    font-size:1rem;
    font-weight:820;
    letter-spacing:-.02em;
}
.pp-hero-meta{
    color:#7b8598;
    font-size:.74rem;
    font-weight:650;
    padding:.34rem .56rem;
    border:1px solid var(--line);
    background:rgba(255,255,255,.78);
    border-radius:999px;
}
.pp-kicker{
    color:var(--accent);
    font-size:.70rem;
    font-weight:820;
    letter-spacing:.11em;
    text-transform:uppercase;
    margin-bottom:.5rem;
}
.pp-title{
    color:var(--text);
    max-width:840px;
    font-size:clamp(2.05rem,4vw,3.2rem);
    line-height:1.06;
    font-weight:835;
    letter-spacing:-.05em;
    margin:0;
}
.pp-subtitle{
    color:var(--muted);
    max-width:760px;
    font-size:.99rem;
    line-height:1.67;
    margin-top:.78rem;
}

/* Programmatic top navigation */
div[data-testid="stRadio"]>label{display:none!important;}
div[data-testid="stRadio"] div[role="radiogroup"]{
    display:grid!important;
    grid-template-columns:repeat(3,1fr)!important;
    gap:5px!important;
    padding:5px!important;
    margin:.8rem 0 1.6rem!important;
    border:1px solid var(--line)!important;
    border-radius:13px!important;
    background:#edf1f6!important;
    overflow:visible!important;
}
div[data-testid="stRadio"] div[role="radiogroup"]>label{
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    min-height:42px!important;
    padding:.48rem .7rem!important;
    border:1px solid transparent!important;
    border-radius:9px!important;
    background:transparent!important;
    cursor:pointer!important;
}
div[data-testid="stRadio"] div[role="radiogroup"]>label>div:first-child{display:none!important;}
div[data-testid="stRadio"] div[role="radiogroup"]>label p{
    margin:0!important;
    color:#6d788b!important;
    font-size:.90rem!important;
    font-weight:720!important;
    white-space:nowrap!important;
}
div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked){
    background:#fff!important;
    border-color:#dde3ec!important;
    box-shadow:0 2px 9px rgba(24,34,53,.07)!important;
}
div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked) p{
    color:var(--text)!important;
}

/* Page hierarchy */
.pp-section-label{
    color:#8b96a8;
    font-size:.67rem;
    font-weight:820;
    letter-spacing:.11em;
    text-transform:uppercase;
    margin:0 0 .26rem;
}
.pp-page-intro{
    color:var(--muted);
    max-width:760px;
    font-size:.94rem;
    line-height:1.62;
    margin-bottom:1rem;
}

/* Streamlit containers */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--surface)!important;
    border:1px solid var(--line)!important;
    border-radius:15px!important;
    box-shadow:0 4px 15px rgba(24,34,53,.035);
}

/* Inputs */
div[data-baseweb="input"]>div,
div[data-baseweb="textarea"]>div,
textarea,input{
    background:#fff!important;
    color:var(--text)!important;
}
div[data-baseweb="input"]>div,
div[data-baseweb="textarea"]>div{
    border-color:var(--line-strong)!important;
    border-radius:10px!important;
}
textarea::placeholder,input::placeholder{
    color:#9aa5b6!important;
    opacity:1!important;
}
.stApp [data-testid="stWidgetLabel"] p{
    font-size:.82rem!important;
    font-weight:700!important;
}

/* Buttons */
div.stButton>button{
    min-height:42px;
    border-radius:9px;
    border:1px solid var(--line-strong);
    background:#fff;
    color:#2b3548;
    font-weight:730;
    box-shadow:none;
}
div.stButton>button:hover{
    border-color:#bcc6d5;
    background:#fafbfe;
}
div.stButton>button[kind="primary"]{
    background:var(--accent)!important;
    border-color:var(--accent)!important;
    color:#fff!important;
    box-shadow:0 5px 14px rgba(79,70,229,.14);
}
div.stButton>button[kind="primary"] *{color:#fff!important;}
div.stButton>button[kind="primary"]:hover{
    background:var(--accent-dark)!important;
    border-color:var(--accent-dark)!important;
}

/* Nested tabs */
div[data-baseweb="tab-list"]{
    gap:1.35rem!important;
    padding:0!important;
    margin:.55rem 0 1rem!important;
    border-bottom:1px solid var(--line)!important;
    background:transparent!important;
    overflow-x:auto!important;
}
button[data-baseweb="tab"]{
    min-width:max-content!important;
    min-height:40px!important;
    padding:0 0 .7rem!important;
    border:0!important;
    background:transparent!important;
    color:#7b8698!important;
    font-weight:710!important;
}
button[data-baseweb="tab"] *,button[data-baseweb="tab"] p{
    color:inherit!important;
    opacity:1!important;
}
button[data-baseweb="tab"][aria-selected="true"]{
    color:var(--text)!important;
    border-bottom:2px solid var(--accent)!important;
}

/* Market form workspace */
.pp-form-note{
    display:flex;
    align-items:flex-start;
    gap:.65rem;
    border:1px solid #e7e5ff;
    background:var(--accent-softer);
    border-radius:11px;
    padding:.72rem .8rem;
    margin:.15rem 0 .85rem;
}
.pp-form-note-dot{
    width:8px;height:8px;border-radius:50%;
    background:var(--accent);
    margin-top:.32rem;
    flex:0 0 auto;
}
.pp-form-note-copy{
    color:#646f84;
    font-size:.78rem;
    line-height:1.48;
}
.pp-workflow{
    height:100%;
    border:1px solid var(--line);
    border-radius:15px;
    background:linear-gradient(180deg,#fff,#fbfcff);
    padding:1rem;
}
.pp-workflow-title{
    color:var(--text);
    font-size:.84rem;
    font-weight:790;
    margin-bottom:.75rem;
}
.pp-flow-step{
    display:grid;
    grid-template-columns:26px 1fr;
    gap:.62rem;
    padding:.62rem 0;
}
.pp-flow-step+.pp-flow-step{border-top:1px solid var(--line);}
.pp-flow-index{
    width:24px;height:24px;border-radius:7px;
    display:flex;align-items:center;justify-content:center;
    background:var(--accent-soft);
    color:var(--accent);
    font-size:.67rem;
    font-weight:830;
}
.pp-flow-title{
    color:var(--text);
    font-size:.79rem;
    font-weight:740;
    line-height:1.34;
}
.pp-flow-copy{
    color:var(--muted);
    font-size:.72rem;
    line-height:1.45;
    margin-top:.1rem;
}
.pp-prelaunch{
    color:#8b96a8;
    font-size:.69rem;
    line-height:1.45;
    padding-top:.72rem;
    margin-top:.45rem;
    border-top:1px solid var(--line);
}

/* Analysis hero */
.pp-analysis-shell{
    overflow:hidden;
    border:1px solid #dfe4ee;
    border-radius:17px;
    background:#fff;
    box-shadow:var(--shadow);
    margin:.45rem 0 1rem;
}
.pp-analysis-head{
    padding:1.2rem 1.35rem 1.05rem;
    background:
        radial-gradient(circle at 96% 0%,rgba(79,70,229,.10),transparent 34%),
        linear-gradient(180deg,#fff,#fcfcff);
}
.pp-analysis-eyebrow{
    color:var(--accent);
    font-size:.66rem;
    font-weight:830;
    letter-spacing:.10em;
    text-transform:uppercase;
    margin-bottom:.5rem;
}
.pp-analysis-question{
    color:var(--text);
    font-size:clamp(1.2rem,2vw,1.5rem);
    line-height:1.36;
    font-weight:790;
    letter-spacing:-.028em;
}
.pp-analysis-event{
    color:var(--muted);
    font-size:.82rem;
    margin-top:.3rem;
}
.pp-analysis-grid{
    display:grid;
    grid-template-columns:1.25fr .8fr .8fr .8fr;
    border-top:1px solid var(--line);
}
.pp-analysis-primary,.pp-analysis-stat{
    min-width:0;
    padding:.95rem 1.1rem 1rem;
}
.pp-analysis-stat{
    border-left:1px solid var(--line);
}
.pp-stat-label{
    color:#8b96a8;
    font-size:.68rem;
    margin-bottom:.25rem;
}
.pp-demand-value{
    font-size:1.65rem;
    line-height:1.05;
    font-weight:850;
    letter-spacing:-.035em;
}
.pp-demand-high{color:var(--high);}
.pp-demand-medium{color:var(--medium);}
.pp-demand-low{color:var(--low);}
.pp-stat-value{
    color:var(--text);
    font-size:1.02rem;
    line-height:1.2;
    font-weight:790;
    letter-spacing:-.02em;
    overflow-wrap:anywhere;
}

/* Overview */
.pp-summary-card{
    padding:.1rem 0;
    background:transparent;
}
.pp-summary-headline{
    color:var(--text);
    font-size:1rem;
    font-weight:780;
    line-height:1.52;
    margin-bottom:.48rem;
}
.pp-summary-copy{
    color:#566276;
    font-size:.93rem;
    line-height:1.67;
}
.pp-summary-stat{
    color:var(--accent);
    font-weight:800;
}
.pp-summary-note{
    color:#8c97a8;
    font-size:.75rem;
    margin-top:.82rem;
    padding-top:.7rem;
    border-top:1px solid var(--line);
}

/* Demand */
.pp-demand-facts{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:.75rem;
    margin-bottom:1rem;
}
.pp-fact{
    border:1px solid var(--line);
    border-radius:11px;
    background:var(--surface-soft);
    padding:.78rem .85rem;
}
.pp-fact-label{
    color:#8b96a8;
    font-size:.67rem;
    margin-bottom:.22rem;
}
.pp-fact-value{
    color:var(--text);
    font-size:.96rem;
    font-weight:790;
}
.pp-score-list{
    display:flex;
    flex-direction:column;
    gap:.68rem;
    margin-top:.3rem;
}
.pp-score-head{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:.28rem;
}
.pp-score-name{
    color:var(--text);
    font-size:.80rem;
    font-weight:720;
}
.pp-score-value{
    color:#68758a;
    font-size:.76rem;
    font-weight:760;
}
.pp-score-track{
    height:8px;
    border-radius:999px;
    background:#edf0f5;
    overflow:hidden;
}
.pp-score-fill{
    height:100%;
    border-radius:999px;
    background:#b8c1cf;
}
.pp-score-fill.is-active{
    background:linear-gradient(90deg,#5b54ee,#4f46e5);
}

/* Historical evidence */
.pp-history-summary{
    display:flex;
    flex-wrap:wrap;
    gap:.65rem 1.15rem;
    padding:.15rem 0 .95rem;
    border-bottom:1px solid var(--line);
    margin-bottom:.9rem;
}
.pp-history-summary-item{}
.pp-history-summary-label{
    color:#929cad;
    font-size:.64rem;
    text-transform:uppercase;
    letter-spacing:.06em;
}
.pp-history-summary-value{
    color:var(--text);
    font-size:.88rem;
    font-weight:780;
    margin-top:.12rem;
}
.pp-history-list{
    display:flex;
    flex-direction:column;
    gap:.42rem;
    margin:.4rem 0 .8rem;
}
.pp-history-row{
    display:grid;
    grid-template-columns:40px minmax(0,1fr) 92px 120px;
    gap:.85rem;
    align-items:center;
    padding:.72rem .02rem;
    border-bottom:1px solid var(--line);
}
.pp-history-row:last-child{border-bottom:0;}
.pp-history-rank{
    width:30px;height:30px;
    border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    background:var(--accent-soft);
    color:var(--accent);
    font-size:.71rem;
    font-weight:820;
}
.pp-history-question{
    color:var(--text);
    font-size:.88rem;
    line-height:1.42;
    font-weight:700;
}
.pp-history-meta{
    color:#8d98a9;
    font-size:.70rem;
    margin-top:.16rem;
}
.pp-history-stat-label{
    color:#98a1b0;
    font-size:.61rem;
    text-transform:uppercase;
    letter-spacing:.06em;
    margin-bottom:.12rem;
}
.pp-history-stat-value{
    color:#334056;
    font-size:.84rem;
    font-weight:780;
}
.pp-history-similarity{color:var(--accent);}
.pp-chart-shell{
    border:1px solid var(--line);
    border-radius:12px;
    padding:.55rem .65rem .18rem;
    background:#fff;
    margin-top:.4rem;
}

/* Explore */
.pp-explore-strip{
    display:flex;
    flex-wrap:wrap;
    gap:1rem 1.4rem;
    padding:.72rem .85rem;
    border:1px solid var(--line);
    border-radius:12px;
    background:#fff;
    margin-bottom:1rem;
}
.pp-explore-label{
    color:#929cad;
    font-size:.63rem;
    text-transform:uppercase;
    letter-spacing:.06em;
}
.pp-explore-value{
    color:var(--text);
    font-size:.90rem;
    font-weight:790;
    margin-top:.12rem;
}
.pp-tag{
    display:inline-flex;
    align-items:center;
    border-radius:999px;
    background:var(--accent-soft);
    color:var(--accent);
    font-size:.63rem;
    font-weight:760;
    padding:.19rem .42rem;
}
.pp-card-title{
    color:var(--text);
    font-size:.94rem;
    line-height:1.44;
    font-weight:760;
    margin:.5rem 0 .2rem;
}
.pp-card-detail{
    color:var(--muted);
    font-size:.72rem;
    line-height:1.42;
}
.pp-card-stats{
    display:flex;
    gap:1rem;
    margin:.75rem 0 .55rem;
    padding-top:.65rem;
    border-top:1px solid var(--line);
}
.pp-card-stat-label{
    color:#929cad;
    font-size:.61rem;
    text-transform:uppercase;
    letter-spacing:.05em;
}
.pp-card-stat-value{
    color:var(--text);
    font-size:.82rem;
    font-weight:780;
    margin-top:.1rem;
}
.pp-cta-hint{
    color:#8994a5;
    font-size:.69rem;
    margin:.2rem 0 .4rem;
}

/* Model architecture */
.pp-pipeline{
    display:grid;
    grid-template-columns:1fr 42px 1fr 42px 1fr;
    align-items:stretch;
    margin:1rem 0 1.15rem;
}
.pp-stage{
    border:1px solid var(--line);
    border-radius:13px;
    background:#fff;
    padding:.95rem;
}
.pp-stage-num{
    color:var(--accent);
    font-size:.64rem;
    font-weight:820;
    letter-spacing:.08em;
    text-transform:uppercase;
}
.pp-stage-title{
    color:var(--text);
    font-size:.93rem;
    font-weight:790;
    margin:.34rem 0 .18rem;
}
.pp-stage-copy{
    color:var(--muted);
    font-size:.75rem;
    line-height:1.48;
}
.pp-stage-arrow{
    display:flex;
    align-items:center;
    justify-content:center;
    color:#a3adbd;
    font-size:1.15rem;
}

/* Tables / expanders / alerts */
[data-testid="stMetric"]{
    background:transparent!important;
    border:0!important;
    padding:0!important;
}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{
    color:#8b96a8!important;
    font-size:.72rem!important;
}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{
    color:var(--text)!important;
    font-weight:800!important;
}
div[data-testid="stDataFrame"]{
    border:1px solid var(--line);
    border-radius:11px;
    overflow:hidden;
    background:#fff;
}
details{
    background:#fff!important;
    border-color:var(--line)!important;
    border-radius:10px!important;
}
.stAlert{border-radius:10px;}

/* Footer */
.pp-footer{
    color:#929cad;
    margin-top:2.5rem;
    padding-top:.95rem;
    border-top:1px solid var(--line);
    font-size:.74rem;
    text-align:center;
}

@media(max-width:860px){
    div[data-testid="stRadio"] div[role="radiogroup"]{
        grid-template-columns:1fr!important;
    }
    .pp-analysis-grid{grid-template-columns:1fr 1fr;}
    .pp-analysis-stat:nth-child(3){border-left:0;border-top:1px solid var(--line);}
    .pp-analysis-stat:nth-child(4){border-top:1px solid var(--line);}
    .pp-history-row{grid-template-columns:36px minmax(0,1fr);}
    .pp-history-row>div:nth-child(3),.pp-history-row>div:nth-child(4){grid-column:2;}
    .pp-pipeline{grid-template-columns:1fr;}
    .pp-stage-arrow{transform:rotate(90deg);height:34px;}
}
@media(max-width:640px){
    .pp-hero{padding:1.1rem;}
    .pp-hero-meta{display:none;}
    .pp-title{font-size:2rem;}
    .pp-analysis-grid{grid-template-columns:1fr;}
    .pp-analysis-stat{border-left:0!important;border-top:1px solid var(--line);}
    .pp-demand-facts{grid-template-columns:1fr;}
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    html = (
        '<div class="pp-hero">'
        '<div class="pp-hero-top">'
        '<div class="pp-brand"><span class="pp-logo">P</span>'
        '<span class="pp-brand-name">PolyPulse</span></div>'
        '<span class="pp-hero-meta">3-model pre-launch pipeline</span>'
        '</div>'
        '<div class="pp-kicker">Prediction-market intelligence</div>'
        '<div class="pp-title">Estimate demand before the market opens.</div>'
        '<div class="pp-subtitle">'
        'Validate the proposition, predict expected trading demand, and ground the '
        'result in comparable historical Polymarket markets.'
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
    demand = results.get("demand")
    history = results.get("history")

    demand_label = demand["label"] if demand else "Unavailable"
    demand_value = demand_label.upper() if demand else "Unavailable"
    demand_class = (
        f"pp-demand-{demand_label.lower()}"
        if demand_label in {"Low", "Medium", "High"}
        else ""
    )
    volume_band = compact_volume_band(demand.get("volume_band") if demand else None)
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
        '<div class="pp-analysis-head">'
        '<div class="pp-analysis-eyebrow">Market analysis</div>'
        f'<div class="pp-analysis-question">{escape(str(results["question"]))}</div>'
        f'<div class="pp-analysis-event">{escape(event_text)}</div>'
        '</div>'
        '<div class="pp-analysis-grid">'
        '<div class="pp-analysis-primary">'
        '<div class="pp-stat-label">Expected demand</div>'
        f'<div class="pp-demand-value {demand_class}">{escape(demand_value)}</div>'
        '</div>'
        '<div class="pp-analysis-stat">'
        '<div class="pp-stat-label">Historical volume band</div>'
        f'<div class="pp-stat-value">{escape(volume_band)}</div>'
        '</div>'
        '<div class="pp-analysis-stat">'
        '<div class="pp-stat-label">Comparable markets</div>'
        f'<div class="pp-stat-value">{escape(comparables)}</div>'
        '</div>'
        '<div class="pp-analysis-stat">'
        '<div class="pp-stat-label">Suggested topic</div>'
        f'<div class="pp-stat-value">{escape(topic)}</div>'
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
    band = pred.get("volume_band") or "Not published"
    top_score = float(pred["model_scores"].get(label, 0.0))
    elapsed = results.get("demand_seconds")

    facts_html = (
        '<div class="pp-demand-facts">'
        '<div class="pp-fact"><div class="pp-fact-label">Predicted demand</div>'
        f'<div class="pp-fact-value">{escape(label.upper())}</div></div>'
        '<div class="pp-fact"><div class="pp-fact-label">Historical demand band</div>'
        f'<div class="pp-fact-value">{escape(str(band))}</div></div>'
        '</div>'
    )
    st.markdown(facts_html, unsafe_allow_html=True)

    st.markdown("#### Model score distribution")
    bars = []
    for class_label in LABELS:
        score = float(pred["model_scores"][class_label])
        active = " is-active" if class_label == label else ""
        width = max(0.0, min(100.0, score * 100.0))
        bars.append(
            '<div>'
            '<div class="pp-score-head">'
            f'<span class="pp-score-name">{escape(class_label)}</span>'
            f'<span class="pp-score-value">{score:.1%}</span>'
            '</div>'
            '<div class="pp-score-track">'
            f'<div class="pp-score-fill{active}" style="width:{width:.2f}%"></div>'
            '</div>'
            '</div>'
        )
    st.markdown(
        '<div class="pp-score-list">' + "".join(bars) + "</div>",
        unsafe_allow_html=True,
    )

    adjustment_note = (
        "Published validation decision adjustments applied."
        if pred.get("adjustments_available")
        else "No published validation decision adjustment applied."
    )
    runtime = f" Inference: {elapsed:.2f}s." if elapsed is not None else ""
    st.caption(
        f"Top score: {top_score:.1%}. Scores are classification scores, not calibrated probabilities. "
        f"{adjustment_note}{runtime}"
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
    index_count = results.get("history_index_count", history.get("n_reference_markets"))
    topic = topic_from_history(history)

    summary_html = (
        '<div class="pp-history-summary">'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Suggested topic</div>'
        f'<div class="pp-history-summary-value">{escape(topic)}</div></div>'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Comparables used</div>'
        f'<div class="pp-history-summary-value">{stats_count:,}</div></div>'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Similarity floor</div>'
        f'<div class="pp-history-summary-value">{threshold:.0%}</div></div>'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Indexed markets</div>'
        f'<div class="pp-history-summary-value">{int(index_count):,}</div></div>'
        '</div>'
    ) if index_count is not None else (
        '<div class="pp-history-summary">'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Suggested topic</div>'
        f'<div class="pp-history-summary-value">{escape(topic)}</div></div>'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Comparables used</div>'
        f'<div class="pp-history-summary-value">{stats_count:,}</div></div>'
        '<div class="pp-history-summary-item"><div class="pp-history-summary-label">Similarity floor</div>'
        f'<div class="pp-history-summary-value">{threshold:.0%}</div></div>'
        '</div>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    if not matches:
        st.info(
            history.get("message")
            or "No historical markets meet the configured similarity threshold."
        )
        return

    st.markdown("#### Closest historical markets")
    st.caption("Top semantic matches, ordered by similarity to the proposed market.")

    rows = []
    for rank, market in enumerate(matches[:5], start=1):
        question = escape(str(market.get("question", "")))
        similarity = 100 * float(market.get("similarity", 0.0))
        volume = escape(money(market.get("volume_usd", 0.0)))
        topic_name = clean_text(market.get("topic", ""))
        event_title = clean_text(market.get("event_title", ""))
        meta_parts = [x for x in [topic_name, event_title] if x]
        meta = " · ".join(meta_parts) if meta_parts else "Historical market"

        rows.append(
            '<div class="pp-history-row">'
            f'<div class="pp-history-rank">{rank}</div>'
            '<div>'
            f'<div class="pp-history-question">{question}</div>'
            f'<div class="pp-history-meta">{escape(meta)}</div>'
            '</div>'
            '<div><div class="pp-history-stat-label">Similarity</div>'
            f'<div class="pp-history-stat-value pp-history-similarity">{similarity:.1f}%</div></div>'
            '<div><div class="pp-history-stat-label">Observed volume</div>'
            f'<div class="pp-history-stat-value">{volume}</div></div>'
            '</div>'
        )

    st.markdown(
        '<div class="pp-history-list">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )

    all_rows = [
        {
            "Rank": rank,
            "Historical question": market["question"],
            "Similarity": 100 * float(market["similarity"]),
            "Observed volume (USD)": float(market["volume_usd"]),
            "Topic": clean_text(market.get("topic", "")),
        }
        for rank, market in enumerate(matches, start=1)
    ]
    with st.expander(f"View all {displayed_count} displayed comparable markets"):
        st.dataframe(
            pd.DataFrame(all_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
                "Historical question": st.column_config.TextColumn(
                    "Historical question", width="large"
                ),
                "Similarity": st.column_config.NumberColumn(
                    "Similarity", format="%.1f%%", step=0.1
                ),
                "Observed volume (USD)": st.column_config.NumberColumn(
                    "Observed volume", format="$%.0f"
                ),
                "Topic": st.column_config.TextColumn("Topic", width="small"),
            },
        )

    elapsed = results.get("history_seconds")
    detail = f"Top {displayed_count} of {total_qualified:,} qualifying historical markets shown."
    if elapsed is not None:
        detail += f" Search inference: {elapsed:.2f}s."
    st.caption(detail)

    st.markdown("#### Historical trading-volume range")
    if history.get("status") != "ok":
        st.info(
            history.get("message")
            or "There are too few comparable markets to report a historical volume range."
        )
        return

    p25 = float(history["p25_volume_usd"])
    median = float(history["median_volume_usd"])
    p75 = float(history["p75_volume_usd"])

    c1, c2, c3 = st.columns(3)
    c1.metric("25th percentile", money(p25))
    c2.metric("Median", money(median))
    c3.metric("75th percentile", money(p75))

    chart_df = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Label": f"#{rank} · {str(market.get('question',''))[:36]}",
                "Question": str(market.get("question", "")),
                "Observed volume (USD)": float(market.get("volume_usd", 0.0)),
                "Similarity": 100 * float(market.get("similarity", 0.0)),
            }
            for rank, market in enumerate(matches, start=1)
        ]
    )

    sort_order = chart_df["Label"].tolist()
    bars = alt.Chart(chart_df).mark_bar(
        cornerRadiusTopRight=5,
        cornerRadiusBottomRight=5,
        size=20,
        color="#7771ea",
    ).encode(
        y=alt.Y(
            "Label:N",
            sort=sort_order,
            title=None,
            axis=alt.Axis(
                labelColor="#566276",
                labelLimit=300,
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
                labelColor="#69758a",
                titleColor="#566276",
                gridColor="#edf0f5",
                domain=False,
            ),
        ),
        tooltip=[
            alt.Tooltip("Rank:Q", title="Rank", format=".0f"),
            alt.Tooltip("Question:N", title="Historical question"),
            alt.Tooltip("Observed volume (USD):Q", title="Observed volume", format="$,.0f"),
            alt.Tooltip("Similarity:Q", title="Similarity", format=".1f"),
        ],
    )

    median_rule = alt.Chart(
        pd.DataFrame({"median": [median]})
    ).mark_rule(
        color="#4f46e5",
        strokeDash=[5, 4],
        strokeWidth=1.5,
    ).encode(x="median:Q")

    median_label = alt.Chart(
        pd.DataFrame({"median": [median], "label": ["Median"]})
    ).mark_text(
        align="left",
        baseline="top",
        dx=5,
        dy=4,
        color="#4f46e5",
        fontSize=11,
        fontWeight=600,
    ).encode(x="median:Q", text="label:N")

    chart = (
        (bars + median_rule + median_label)
        .properties(
            height=max(300, 35 * displayed_count),
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
        f"Bars show observed volume for the {displayed_count} displayed matches. "
        f"The dashed line is the median calculated from the {stats_count} nearest qualified markets."
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
    st.session_state["top_nav"] = "📊 Market Analysis"
    st.session_state["example_loaded_notice"] = True


def show_popular_questions():
    section_label("Explore")
    st.subheader("Explore historical markets")
    st.markdown(
        '<div class="pp-page-intro">'
        'Start from a real historical market with strong coverage in the reference index. '
        'Select one to load it directly into Market Analysis.'
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

    strip = (
        '<div class="pp-explore-strip">'
        '<div><div class="pp-explore-label">Indexed markets</div>'
        f'<div class="pp-explore-value">{total_markets:,}</div></div>'
        '<div><div class="pp-explore-label">Ready-to-try examples</div>'
        f'<div class="pp-explore-value">{len(table)}</div></div>'
        '<div><div class="pp-explore-label">Similarity floor</div>'
        f'<div class="pp-explore-value">{similarity_threshold:.0%}</div></div>'
        '</div>'
    )
    st.markdown(strip, unsafe_allow_html=True)

    if table.empty:
        st.info("No sample questions meet the current historical-coverage rules.")
        return

    raw_categories = list(dict.fromkeys(table["Category"]))
    category_options = {"All categories": None}
    for category in raw_categories:
        category_options[str(category)] = category

    selected_label = st.selectbox(
        "Category",
        list(category_options),
        key="popular_category_filter",
    )
    selected_category = category_options[selected_label]
    filtered = (
        table if selected_category is None
        else table.loc[table["Category"] == selected_category]
    )

    st.markdown("#### Featured examples")
    preview = filtered.head(6).reset_index(drop=True)
    card_columns = st.columns(2, gap="large")

    for idx, row in preview.iterrows():
        with card_columns[idx % 2]:
            with st.container(border=True):
                category = str(row["Category"])
                event = clean_text(row["Event title"])
                question = escape(str(row["Market question"]))
                st.markdown(
                    f'<span class="pp-tag">{escape(category)}</span>'
                    f'<div class="pp-card-title">{question}</div>'
                    + (
                        f'<div class="pp-card-detail">Event · {escape(event)}</div>'
                        if event else
                        '<div class="pp-card-detail">Historical Polymarket market</div>'
                    )
                    + '<div class="pp-card-stats">'
                    '<div><div class="pp-card-stat-label">Similar markets</div>'
                    f'<div class="pp-card-stat-value">{int(row["Similar markets"]):,}</div></div>'
                    '<div><div class="pp-card-stat-label">Median volume</div>'
                    f'<div class="pp-card-stat-value">{escape(money(row["Median historical volume (USD)"]))}</div></div>'
                    '</div>'
                    '<div class="pp-cta-hint">Load this question into the analysis workspace</div>',
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
        st.dataframe(
            display_table.drop(columns=["Event title"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Market question": st.column_config.TextColumn("Market question", width="large"),
                "Similar markets": st.column_config.NumberColumn(
                    "Similar markets", format="%d"
                ),
                "Median historical volume (USD)": st.column_config.NumberColumn(
                    "Median historical volume", format="$%.0f"
                ),
            },
        )

    st.caption(
        "Examples come directly from the historical index. Similar-market counts measure "
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
    st.subheader("Model stack")
    st.markdown(
        '<div class="pp-page-intro">'
        'Three specialised models handle proposition validity, demand classification, '
        'and historical retrieval. Each stage has a single responsibility.'
        '</div>',
        unsafe_allow_html=True,
    )

    pipeline = (
        '<div class="pp-pipeline">'
        '<div class="pp-stage">'
        '<div class="pp-stage-num">Stage 1 · Model 3</div>'
        '<div class="pp-stage-title">Question validity</div>'
        '<div class="pp-stage-copy">Rejects propositions that are unclear or cannot be objectively resolved.</div>'
        '</div>'
        '<div class="pp-stage-arrow">→</div>'
        '<div class="pp-stage">'
        '<div class="pp-stage-num">Stage 2 · Model 1</div>'
        '<div class="pp-stage-title">Demand prediction</div>'
        '<div class="pp-stage-copy">Classifies expected pre-launch demand as Low, Medium, or High.</div>'
        '</div>'
        '<div class="pp-stage-arrow">→</div>'
        '<div class="pp-stage">'
        '<div class="pp-stage-num">Stage 3 · Model 2</div>'
        '<div class="pp-stage-title">Historical evidence</div>'
        '<div class="pp-stage-copy">Finds comparable markets and their observed trading volumes.</div>'
        '</div>'
        '</div>'
    )
    st.markdown(pipeline, unsafe_allow_html=True)

    with st.expander("Model 1 · Demand prediction"):
        st.write(
            "**Fine-tuned model:** "
            "[DeBERTa-v3-base demand classifier]"
            "(https://huggingface.co/2610gnam05/polypulse-deberta-v3-base-demand)"
        )
        st.write("**Pipeline:** `text-classification` · CPU · Low / Medium / High")
        if COMPARISON_CSV.exists():
            comparison = pd.read_csv(COMPARISON_CSV, dtype=str).fillna("")
            if (
                not comparison.empty
                and "test_accuracy" in comparison.columns
                and comparison["test_accuracy"].str.strip().any()
            ):
                st.dataframe(comparison, hide_index=True, use_container_width=True)

    with st.expander("Model 2 · Historical retrieval"):
        st.write(
            "**Pretrained model:** "
            "[all-MiniLM-L6-v2]"
            "(https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)"
        )
        st.write("**Pipeline:** `feature-extraction` · CPU · 384-dimensional embeddings")
        try:
            import json
            with open(DATA_DIR / "metadata.json", encoding="utf-8") as handle:
                meta = json.load(handle)
            c1, c2 = st.columns(2)
            c1.metric("Indexed historical markets", f"{meta['record_count']:,}")
            c2.metric("Reference cutoff", str(meta.get("reference_cutoff_utc", "Unavailable")))
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
            "This model executes first. Only propositions that pass its published threshold "
            "continue to demand prediction and historical retrieval."
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
        '<div class="pp-page-intro">'
        'Enter the proposition exactly as it would appear to users. PolyPulse first validates '
        'that it can be objectively resolved, then estimates demand and retrieves historical evidence.'
        '</div>',
        unsafe_allow_html=True,
    )

    def clear_previous_analysis():
        st.session_state.pop("polypulse_analysis", None)

    def clear_everything():
        clear_previous_analysis()
        st.session_state["market_question"] = ""
        st.session_state["market_event_title"] = ""

    form_col, flow_col = st.columns([2.15, .85], gap="large")

    with form_col:
        with st.container(border=True):
            question = st.text_area(
                "Market question",
                placeholder="Will Bitcoin exceed $150,000 by December 2027?",
                max_chars=1200,
                height=112,
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

            st.markdown(
                '<div class="pp-form-note">'
                '<div class="pp-form-note-dot"></div>'
                '<div class="pp-form-note-copy">'
                'A strong proposition identifies a specific future event or measurable outcome '
                'and makes the resolution condition unambiguous.'
                '</div></div>',
                unsafe_allow_html=True,
            )

            analyse_col, clear_col, spacer_col = st.columns([2, 1, 3])
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

    with flow_col:
        workflow = (
            '<div class="pp-workflow">'
            '<div class="pp-workflow-title">Analysis flow</div>'
            '<div class="pp-flow-step"><div class="pp-flow-index">1</div><div>'
            '<div class="pp-flow-title">Validate</div>'
            '<div class="pp-flow-copy">Check that the proposition is clear and objectively resolvable.</div>'
            '</div></div>'
            '<div class="pp-flow-step"><div class="pp-flow-index">2</div><div>'
            '<div class="pp-flow-title">Predict demand</div>'
            '<div class="pp-flow-copy">Classify expected pre-launch demand as Low, Medium, or High.</div>'
            '</div></div>'
            '<div class="pp-flow-step"><div class="pp-flow-index">3</div><div>'
            '<div class="pp-flow-title">Retrieve evidence</div>'
            '<div class="pp-flow-copy">Find comparable historical markets and observed volumes.</div>'
            '</div></div>'
            '<div class="pp-prelaunch">Pre-launch analysis only · Post-launch volume is not used as a predictive input.</div>'
            '</div>'
        )
        st.markdown(workflow, unsafe_allow_html=True)

    results_placeholder = st.empty()

    if submitted:
        results_placeholder.empty()
        if not question.strip():
            st.error("Enter a market question before analysing.")
        else:
            st.session_state["polypulse_analysis"] = analyse(
                question.strip(), event_title.strip()
            )

    if "polypulse_analysis" in st.session_state:
        result = st.session_state["polypulse_analysis"]

        with results_placeholder.container():
            validity = result.get("validity")

            if validity is None:
                st.error("Question validation is temporarily unavailable. Please try again.")
            elif not validity.get("is_valid", False):
                st.error(
                    "Please enter a clear prediction-market proposition, "
                    "with an outcome that can be objectively resolved."
                )
            else:
                st.divider()
                section_label("Result")
                st.markdown("## Market analysis")
                show_analysis_results(result)

elif current_page == "Explore Questions":
    show_popular_questions()

elif current_page == "Model Stack":
    show_performance()

st.markdown(
    """
    <div class="pp-footer">
        PolyPulse · Pre-launch demand intelligence · Historical evidence is descriptive, not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
