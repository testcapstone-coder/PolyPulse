"""PolyPulse: AI-powered prediction-market demand and historical intelligence.

Deploy `app.py` to Streamlit Community Cloud. Both Hugging Face pipelines
execute on CPU and are loaded only when a market is analysed. No model training.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import streamlit as st

from demand_prediction import LABELS, load_classifier, load_published_config, predict_demand
from historical_search import clean_text, find_similar_markets, get_extractor, load_reference_data

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPARISON_CSV = BASE_DIR / "results" / "model_comparison.csv"

# Topic labels come from the exported historical index, not hard-coded questions.
# This keeps the example tab compatible with each newly generated Colab index.

st.set_page_config(
    page_title="PolyPulse | Market Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def classifier_resources():
    """One fine-tuned DeBERTa pipeline and its published config per app process."""
    classifier = load_classifier()  # Loads model + its standard config.json from HF.
    config = load_published_config()
    return classifier, config


@st.cache_resource(show_spinner=False)
def historical_resources():
    """Cache exported reference records, embeddings, and the MiniLM pipeline."""
    markets, embeddings, metadata = load_reference_data(str(DATA_DIR))
    extractor = get_extractor(
        metadata["model_id"], metadata.get("model_revision"), device=-1
    )
    return markets, embeddings, metadata, extractor


def analyse(question, title):
    """Run both pipelines independently; preserve a successful result if one fails."""
    results = {
        "question": question, "event_title": title, "demand": None,
        "history": None, "demand_error": None, "history_error": None,
        "demand_seconds": None, "history_seconds": None,
        "history_index_count": None,
    }
    try:
        with st.spinner("Loading fine-tuned DeBERTa and predicting demand…"):
            classifier, config = classifier_resources()
            start = perf_counter()
            results["demand"] = predict_demand(
                question, title, classifier=classifier, config=config
            )
            results["demand_seconds"] = perf_counter() - start
    except Exception as exc:
        results["demand_error"] = f"{type(exc).__name__}: {exc}"

    try:
        with st.spinner("Searching the historical index with MiniLM…"):
            # The PolyPulse 04 notebook exports these metadata names.
            _, _, metadata, _ = historical_resources()
            results["history_index_count"] = int(metadata["record_count"])
            min_similarity = float(metadata.get(
                "min_similarity_default", metadata.get("default_min_similarity", 0.52)
            ))
            min_comparables = int(metadata.get(
                "min_comparables_default", metadata.get("default_min_comparables", 5)
            ))
            stats_top_n = max(10, min_comparables, int(metadata.get("statistics_top_n_default", 30)))
            start = perf_counter()
            results["history"] = find_similar_markets(
                question, event_title=title, data_dir=str(DATA_DIR),
                top_k=10, stats_top_n=stats_top_n,
                min_similarity=min_similarity, min_comparables=min_comparables,
                include_topic=True,
            )
            results["history_seconds"] = perf_counter() - start
    except Exception as exc:
        results["history_error"] = f"{type(exc).__name__}: {exc}"
    return results


def money(value):
    return f"${float(value):,.0f}"


def show_demand(results):
    st.subheader("Trading-demand prediction")
    pred = results["demand"]
    if pred is None:
        st.warning("Demand classification is currently unavailable. Historical results may still be shown.")
        with st.expander("Technical details"):
            st.code(results["demand_error"] or "Unknown error")
        return

    st.metric("Predicted market demand", pred["label"].upper())
    if pred["volume_band"]:
        st.write(f"**Historical category volume band:** {pred['volume_band']}")
        st.caption("Historical classification thresholds, not an exact USD-volume forecast.")
    else:
        st.caption("Historical category USD cutoffs are not published in the model config; "
                   "the actual volumes of similar markets are shown below instead.")
    if pred["adjustments_available"]:
        st.markdown("**Demand classification scores (with published decision adjustments)**")
    else:
        st.markdown("**Demand classification scores (raw model output)**")
    for label in LABELS:
        score = pred["model_scores"][label]
        st.progress(min(max(score, 0.0), 1.0), text=f"{label}: {score:.1%}")
    st.caption(
        "These are model scores, not necessarily calibrated probabilities. "
        + ("Published validation adjustments applied. " if pred["adjustments_available"]
           else "No validation adjustments are in config.json; scores are unadjusted "
                "and may differ from Colab-tuned results. ")
        + f"Measured classifier inference: {results['demand_seconds']:.2f} s "
          "(excludes initial download and model loading)."
    )


def show_history(results):
    """Display PolyPulse 04 results without conflating shown and statistical matches."""
    st.subheader("Historical comparable markets")
    history = results["history"]
    if history is None:
        st.warning("Historical similarity search is currently unavailable. Demand results may still be shown.")
        with st.expander("Technical details"):
            st.code(results.get("history_error") or "Unknown error")
        return

    matches = history.get("matches") or []
    displayed_count = len(matches)
    stats_count = int(history.get("n_comparables", displayed_count))
    total_qualified = int(history.get("n_above_threshold", stats_count))
    threshold = float(history.get("minimum_similarity", 0.52))
    index_count = results.get("history_index_count", history.get("n_reference_markets"))

    details = (
        f"Showing {displayed_count} of {total_qualified:,} historical markets above "
        f"the cosine similarity threshold ({threshold:.0%}). "
    )
    if index_count is not None:
        details += f"Indexed reference markets: {int(index_count):,}. "
    elapsed = results.get("history_seconds")
    if elapsed is not None:
        details += f"Search inference: {elapsed:.2f} s."
    st.caption(details)

    topic = history.get("topic_suggestion")
    if isinstance(topic, dict) and topic.get("topic"):
        st.write(f"**Suggested topic:** {topic['topic']}")
        st.caption("Topic is a MiniLM semantic similarity heuristic, not a trained classifier.")

    if not matches:
        st.info(history.get("message") or "No historical markets meet the similarity threshold.")
        return

    data = pd.DataFrame([
        {
            "Historical question": m["question"],
            "Similarity": 100 * float(m["similarity"]),
            "Actual volume (USD)": float(m["volume_usd"]),
        }
        for m in matches
    ])
    st.dataframe(
        data, use_container_width=True, hide_index=True,
        column_config={
            "Similarity": st.column_config.NumberColumn(
                format="%.1f%%", help="Cosine similarity displayed as a percentage", step=0.1
            ),
            "Actual volume (USD)": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    with st.expander("Expand full comparable-market details"):
        st.dataframe(pd.DataFrame(matches), hide_index=True, use_container_width=True)

    st.subheader("Historical trading-volume analysis")
    if history.get("status") != "ok":
        st.info(history.get("message") or "Insufficient comparable historical markets.")
        st.caption("Historical volume percentiles are not reported when comparable data is insufficient.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Median historical volume", money(history["median_volume_usd"]))
    col2.metric("25th percentile", money(history["p25_volume_usd"]))
    col3.metric("75th percentile", money(history["p75_volume_usd"]))
    st.caption(
        f"Statistics calculated from the {stats_count} nearest qualified historical markets; "
        f"the table and chart display the top {displayed_count}. "
        "These are recorded USD volumes, not predictions."
    )
    # Chart the DISPLAYED markets only. The up-to-30-market statistics sample
    # is not included in the returned matches list.
    chart_data = pd.DataFrame({
        "Comparable rank": [str(i) for i in range(1, displayed_count + 1)],
        "Historical volume (USD)": [float(m["volume_usd"]) for m in matches],
    }).set_index("Comparable rank")
    st.bar_chart(chart_data, x_label="Displayed comparable-market rank", y_label="Observed USD volume")


def show_summary(results):
    st.subheader("Combined market insight")
    demand, history = results["demand"], results["history"]
    if demand and history and history["status"] == "ok":
        st.write(
            f"DeBERTa classifies this proposed market as **{demand['label']} demand**. "
            f"MiniLM identified **{history['n_comparables']}** comparable historical markets "
            f"with a median observed trading volume of **{money(history['median_volume_usd'])}**."
        )
    elif demand and history:
        st.write(
            f"DeBERTa classifies this proposed market as **{demand['label']} demand**. "
            "MiniLM found too few sufficiently similar historical markets to report "
            "a reliable descriptive volume range."
        )
    elif demand:
        st.write(f"DeBERTa classifies this proposed market as **{demand['label']} demand**. Historical evidence is unavailable.")
    elif history and history["status"] == "ok":
        st.write(
            f"Demand classification is unavailable. The **{history['n_comparables']}** "
            f"comparable historical markets had median observed volume "
            f"**{money(history['median_volume_usd'])}**."
        )
    elif history:
        st.write("Demand classification is unavailable and there are too few historical comparables for a volume range.")
    else:
        st.warning("Neither model produced an analysis. See the component errors above.")
    st.caption("Market research support only. PolyPulse does not predict event outcomes or recommend market launches.")


def popular_questions_table(markets, embeddings, min_similarity=0.52, min_comparables=5):
    """Find a few real, high-volume reference questions per indexed topic.

    Suggested questions are taken from the export itself, never invented.
    Self-matches are EXCLUDED when counting neighbors or computing medians.
    The examples are a convenience feature, not evidence of search traffic.
    """
    columns = ["Category", "Market question", "Event title", "Similar markets",
               "Median historical volume (USD)"]
    if markets.empty:
        return pd.DataFrame(columns=columns)

    topics = (markets["topic"].fillna("Other").astype(str)
              if "topic" in markets.columns else pd.Series("Other", index=markets.index))
    # At most two examples per topic, chosen from the highest observed market
    # volumes. No fixed 20K-era question list is required.
    candidates = pd.DataFrame({
        "index": np.arange(len(markets)),
        "Category": topics.to_numpy(),
        "Volume": markets["volume_usd"].to_numpy(dtype=float),
    })
    candidates = (candidates.sort_values("Volume", ascending=False, kind="stable")
                  .groupby("Category", sort=True).head(2))
    rows = []
    for _, candidate in candidates.iterrows():
        idx = int(candidate["index"])
        query_vector = np.asarray(embeddings[idx], dtype=np.float32)
        similarities = np.asarray(embeddings @ query_vector, dtype=np.float32)
        qualified = np.flatnonzero(similarities >= min_similarity)
        qualified = qualified[qualified != idx]  # no self-match
        if len(qualified) < min_comparables:
            continue
        nearest = qualified[np.argsort(-similarities[qualified], kind="stable")[:10]]
        volumes = markets.iloc[nearest]["volume_usd"].to_numpy(dtype=float)
        rows.append({
            "Category": str(candidate["Category"]),
            "Market question": str(markets.iloc[idx]["question"]),
            "Event title": clean_text(markets.iloc[idx].get("event_title", "")),
            "Similar markets": int(len(qualified)),
            "Median historical volume (USD)": float(np.median(volumes)),
        })
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(["Similar markets", "Market question"], ascending=[False, True])
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def load_popular_questions():
    """Read only historical data and vectors; do not load either HF pipeline."""
    markets, embeddings, metadata = load_reference_data(str(DATA_DIR))
    table = popular_questions_table(
        markets, embeddings,
        min_similarity=float(metadata.get("min_similarity_default", metadata.get("default_min_similarity", 0.52))),
        min_comparables=int(metadata.get("min_comparables_default", metadata.get("default_min_comparables", 5))),
    )
    return table, int(metadata["record_count"]), float(metadata.get("min_similarity_default", metadata.get("default_min_similarity", 0.52)))


def show_popular_questions():
    st.subheader("🔥 Popular questions to explore")
    st.write(
        "Explore real historical Polymarket questions taken directly from the current index. "
        "Examples are selected among high-volume markets across available topics, "
        "then ranked by the number of similar markets. These are not trending searches."
    )
    try:
        with st.spinner("Finding the best-covered historical examples…"):
            table, total_markets, similarity_threshold = load_popular_questions()
    except Exception as exc:
        st.warning("Popular historical questions are currently unavailable.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    count_col, examples_col, threshold_col = st.columns(3)
    count_col.metric("Historical markets indexed", f"{total_markets:,}")
    examples_col.metric("Ready-to-try questions", len(table))
    threshold_col.metric("Minimum similarity", f"{similarity_threshold:.0%}")
    if table.empty:
        st.info("No matching sample questions were found in the historical dataset.")
        return

    categories = ["🌐 All categories"] + list(dict.fromkeys(table["Category"]))
    selection = st.selectbox("Explore a category", categories, key="popular_category_filter")
    if selection != "🌐 All categories":
        table = table.loc[table["Category"] == selection]

    st.dataframe(
        table.drop(columns=["Event title"]), hide_index=True, use_container_width=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Market question": st.column_config.TextColumn("Market question", width="large"),
            "Similar markets": st.column_config.NumberColumn(
                "Similar markets", format="%d", help="Other indexed markets with cosine similarity above the threshold, excluding the seed market."
            ),
            "Median historical volume (USD)": st.column_config.NumberColumn(
                "Median historical volume", format="$%.0f",
                help="Median observed volume among the 10 nearest qualifying markets, or fewer if fewer qualify."
            ),
        },
    )
    with st.expander("See event titles for these example questions"):
        st.caption("For a closer match to the values above, enter both the question "
                   "and its original event title in Market Analysis.")
        st.dataframe(table[["Market question", "Event title"]],
                     hide_index=True, use_container_width=True)
    st.caption(
        "‘Similar markets’ measures semantic coverage, not how often people searched. "
        "Median volume is based on up to 10 other matching reference markets (self excluded). "
        "Historical USD volumes are observations, not forecasts. "
        "Use new questions for independent evaluation."
    )


def show_performance():
    st.subheader("Model 1 · Demand prediction")
    st.write("**Fine-tuned model:** [DeBERTa-v3-base demand classifier](https://huggingface.co/2610gnam05/polypulse-deberta-v3-base-demand)")
    st.write("**Hugging Face pipeline:** `text-classification` · CPU · three demand labels")
    st.write(
        "Fine-tuning dataset, training sample count, accuracy, macro-F1, training runtime "
        "and confusion matrix must be supplied from the final Colab experiments. "
        "They are not present in the provided reference bundle."
    )
    if COMPARISON_CSV.exists():
        comparison = pd.read_csv(COMPARISON_CSV, dtype=str).fillna("")
        if comparison.empty or not comparison["test_accuracy"].str.strip().any():
            st.info("Final model-comparison metrics have not been supplied. The table is a template, not evaluation evidence.")
        else:
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        with st.expander("Model experiment table / editable template"):
            st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("Model 2 · Historical similarity")
    st.write(
        "**Pretrained model:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)"
    )
    st.write("**Hugging Face pipeline:** `feature-extraction` · CPU · no additional fine-tuning")
    st.write(
        "Historical questions and optional event titles are converted to 384-dimensional "
        "embeddings, attention-mask mean-pooled and L2-normalised. NumPy cosine similarity "
        "retrieves observed Polymarket USD volumes from the supplied parquet reference."
    )
    try:
        # Lightweight metadata read does NOT cause model download on this tab.
        import json
        with open(DATA_DIR / "metadata.json", encoding="utf-8") as handle:
            meta = json.load(handle)
        st.metric("Indexed historical markets", f"{meta['record_count']:,}")
        st.caption(
            f"Historical source snapshot cutoff: {meta.get('reference_cutoff_utc', 'unavailable')}. "
            "Historical cosine-similarity retrieval is not a demand-accuracy metric."
        )
    except Exception as exc:
        st.warning(f"Historical metadata unavailable: {exc}")


st.title("PolyPulse")
st.caption("AI-Powered Prediction Market Intelligence")
market_tab, popular_tab, performance_tab = st.tabs(
    ["Market Analysis", "🔥 Popular Questions", "Model Performance"]
)
with market_tab:
    st.write("Assess potential trading demand and compare a proposed market with historical Polymarket observations.")
    # Keep the displayed analysis while the user edits the inputs.
    # Only a click on Analyse Market (or the explicit Clear button) resets it.
    def clear_previous_analysis():
        st.session_state.pop("polypulse_analysis", None)

    def clear_everything():
        clear_previous_analysis()
        st.session_state["market_question"] = ""
        st.session_state["market_event_title"] = ""

    question = st.text_area(
        "Market Question *", placeholder="Will Bitcoin exceed $150,000 by December 2027?",
        max_chars=1200, height=90, key="market_question",
    )
    event_title = st.text_input(
        "Event Title (optional)", placeholder="Bitcoin Price Predictions", max_chars=300,
        key="market_event_title",
    )
    analyse_col, clear_col = st.columns([1, 5])
    with analyse_col:
        submitted = st.button("Analyse Market", type="primary", use_container_width=True)
    with clear_col:
        st.button("Clear", on_click=clear_everything)

    if submitted:
        # Never show an older market's results during a new analysis.
        clear_previous_analysis()
        if not question.strip():
            st.error("Please enter a Market Question before analysing.")
        else:
            st.session_state["polypulse_analysis"] = analyse(question.strip(), event_title.strip())
    if "polypulse_analysis" in st.session_state:
        result = st.session_state["polypulse_analysis"]
        st.divider()
        st.write(f"**Market:** {result['question']}")
        if result["event_title"]:
            st.caption(f"Event: {result['event_title']}")
        show_demand(result)
        st.divider()
        show_history(result)
        st.divider()
        show_summary(result)
with popular_tab:
    show_popular_questions()
with performance_tab:
    show_performance()
