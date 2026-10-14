"""PolyPulse: AI-powered prediction-market demand and historical intelligence.

Deploy this file as `app.py` on Streamlit Community Cloud. Three Hugging Face
models run on CPU and are loaded only when needed. Runtime order is:
validity gate → demand classifier → MiniLM historical retrieval.
No model training occurs in the app.
"""

from html import escape
from pathlib import Path
from time import perf_counter

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
    """Small, dependency-free design system for the Streamlit UI."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1220px;
            padding-top: 1.8rem;
            padding-bottom: 3.5rem;
        }

        .pp-hero {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 24px;
            padding: 28px 30px 26px 30px;
            margin-bottom: 1.1rem;
            background:
                radial-gradient(circle at 88% 8%, rgba(123, 97, 255, 0.18), transparent 30%),
                radial-gradient(circle at 8% 96%, rgba(36, 188, 255, 0.12), transparent 34%),
                rgba(127, 127, 127, 0.035);
        }

        .pp-kicker {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            opacity: 0.68;
            margin-bottom: 0.45rem;
        }

        .pp-title {
            font-size: clamp(2.15rem, 5vw, 3.55rem);
            line-height: 1.02;
            font-weight: 820;
            letter-spacing: -0.045em;
            margin: 0;
        }

        .pp-subtitle {
            max-width: 760px;
            font-size: 1.03rem;
            line-height: 1.55;
            opacity: 0.76;
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
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 999px;
            padding: 0.38rem 0.68rem;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(127, 127, 127, 0.055);
        }

        .pp-section-label {
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            opacity: 0.58;
            margin: 0.15rem 0 0.2rem 0;
        }

        .pp-result-question {
            font-size: 1.2rem;
            font-weight: 680;
            line-height: 1.42;
            margin: 0.2rem 0 0.35rem 0;
        }

        .pp-card-title {
            font-weight: 720;
            font-size: 0.98rem;
            margin-bottom: 0.22rem;
        }

        .pp-card-copy {
            opacity: 0.72;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .pp-footer {
            margin-top: 2.8rem;
            padding-top: 1.1rem;
            border-top: 1px solid rgba(128,128,128,.18);
            opacity: 0.58;
            font-size: 0.82rem;
            text-align: center;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            background: rgba(127, 127, 127, 0.035);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            opacity: 0.72;
        }

        [data-testid="stMetricValue"] {
            font-weight: 760;
        }

        div.stButton > button {
            border-radius: 12px;
            min-height: 44px;
            font-weight: 690;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            border-radius: 12px;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .stAlert {
            border-radius: 14px;
        }

        @media (max-width: 720px) {
            .pp-hero {
                padding: 22px 20px;
                border-radius: 19px;
            }
            .pp-title {
                font-size: 2.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="pp-hero">
            <div class="pp-kicker">Prediction-market intelligence</div>
            <div class="pp-title">PolyPulse</div>
            <div class="pp-subtitle">
                Estimate pre-launch trading demand, benchmark a proposed market against
                historical Polymarket observations, and surface the evidence behind the result.
            </div>
            <div class="pp-pill-row">
                <span class="pp-pill">🛡️ Validity first</span>
                <span class="pp-pill">📈 Demand forecast</span>
                <span class="pp-pill">🧭 Historical comparables</span>
                <span class="pp-pill">⚙️ Hugging Face models</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    with st.container(border=True):
        section_label("Analysed market")
        st.markdown(
            f'<div class="pp-result-question">{escape(results["question"])}</div>',
            unsafe_allow_html=True,
        )
        if results.get("event_title"):
            st.caption(f"Event: {results['event_title']}")

    demand_value = demand["label"].upper() if demand else "Unavailable"
    volume_band = (
        demand.get("volume_band") if demand and demand.get("volume_band") else "Not published"
    )
    n_comp = comparable_count(history)
    topic = topic_from_history(history)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected demand", demand_value)
    col2.metric("Historical volume band", volume_band)
    col3.metric("Comparable markets", f"{n_comp:,}" if n_comp is not None else "—")
    col4.metric("Suggested topic", topic)


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

    st.markdown("#### Closest historical markets")
    data = pd.DataFrame(
        [
            {
                "Historical question": m["question"],
                "Similarity": 100 * float(m["similarity"]),
                "Actual volume (USD)": float(m["volume_usd"]),
            }
            for m in matches
        ]
    )
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        column_config={
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
            "Actual volume (USD)": st.column_config.NumberColumn(
                "Observed volume",
                format="$%.0f",
            ),
        },
    )

    elapsed = results.get("history_seconds")
    detail_text = (
        f"Showing {displayed_count} of {total_qualified:,} qualifying historical markets."
    )
    if elapsed is not None:
        detail_text += f" Search inference: {elapsed:.2f}s."
    st.caption(detail_text)

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

    chart_data = pd.DataFrame(
        {
            "Comparable": [f"#{i}" for i in range(1, displayed_count + 1)],
            "Observed volume (USD)": [
                float(m["volume_usd"]) for m in matches
            ],
        }
    ).set_index("Comparable")
    st.bar_chart(
        chart_data,
        x_label="Comparable market rank",
        y_label="Observed USD volume",
    )

    with st.expander("Full comparable-market details"):
        st.dataframe(
            pd.DataFrame(matches),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        f"Percentiles use the {stats_count} nearest qualified markets; the table and "
        f"chart display the top {displayed_count}. Historical volumes are observations, "
        "not forecasts."
    )


def show_summary(results, show_title=True):
    if show_title:
        st.subheader("Executive summary")

    demand = results.get("demand")
    history = results.get("history")

    with st.container(border=True):
        if demand and history and history.get("status") == "ok":
            label = demand["label"].upper()
            st.markdown(
                f"**PolyPulse indicates {label} expected demand** for this proposed market."
            )
            st.write(
                f"The {history['n_comparables']} nearest qualified historical markets "
                f"had a median observed trading volume of "
                f"**{money(history['median_volume_usd'])}**, with a 25th–75th percentile "
                f"range of **{money(history['p25_volume_usd'])}–{money(history['p75_volume_usd'])}**."
            )
        elif demand and history:
            st.markdown(
                f"**PolyPulse indicates {demand['label'].upper()} expected demand.**"
            )
            st.write(
                "Historical retrieval found too few sufficiently similar markets to "
                "report a reliable descriptive volume range."
            )
        elif demand:
            st.markdown(
                f"**PolyPulse indicates {demand['label'].upper()} expected demand.**"
            )
            st.write("Historical evidence is currently unavailable.")
        elif history and history.get("status") == "ok":
            st.write(
                "Demand classification is currently unavailable. "
                f"The {history['n_comparables']} comparable markets had a median observed "
                f"volume of **{money(history['median_volume_usd'])}**."
            )
        elif history:
            st.write(
                "Demand classification is unavailable and there are too few historical "
                "comparables to report a volume range."
            )
        else:
            st.warning("No analysis result is currently available.")

    st.caption(
        "Decision-support only. PolyPulse estimates market demand; it does not predict "
        "the event outcome or recommend launching a market."
    )


def show_analysis_results(result):
    show_result_header(result)

    overview_tab, demand_tab, history_tab = st.tabs(
        ["⚡ Overview", "📈 Demand", "🧭 Historical evidence"]
    )
    with overview_tab:
        show_summary(result, show_title=False)

    with demand_tab:
        show_demand(result, show_title=False)

    with history_tab:
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
    """Populate the Market Analysis form on the next Streamlit rerun."""
    st.session_state["market_question"] = str(question)
    st.session_state["market_event_title"] = str(event_title or "")
    st.session_state.pop("polypulse_analysis", None)
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

                st.button(
                    "Use in Market Analysis →",
                    key=f"use_example_{selected_category}_{idx}_{row['Market question'][:30]}",
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
    st.toast("Example loaded. Open Market Analysis to run it.", icon="✅")

market_tab, popular_tab, performance_tab = st.tabs(
    ["📊 Market Analysis", "✨ Explore Questions", "🧠 Model Stack"]
)

with market_tab:
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
                st.markdown("### Analysis")
                show_analysis_results(result)

with popular_tab:
    show_popular_questions()

with performance_tab:
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
