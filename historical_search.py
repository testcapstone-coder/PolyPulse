"""PolyPulse historical market comparison, MiniLM only (CPU, no re-embedding index).

Streamlit usage:
    from historical_search import find_similar_markets
    result = find_similar_markets("Will Bitcoin hit $150,000 by 2027?", data_dir="data")

Expected data/ directory contains metadata.json plus numbered parquet/npy parts.
No raw Hugging Face datasets or transaction records are used at runtime.
"""
from functools import lru_cache
from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256
DEFAULT_MIN_SIMILARITY = 0.52
DEFAULT_MIN_COMPARABLES = 5
TOPIC_DESCRIPTIONS = {
    "Politics": "Political elections, presidents, political parties, government decisions, wars and geopolitics.",
    "Cryptocurrency": "Bitcoin, Ethereum, cryptocurrency token price targets, blockchain and crypto exchanges.",
    "Sports": "Sports teams, football, soccer, NBA, NFL, tennis, championships, athletes and match results.",
    "Economics": "Stock market, interest rates, economy, GDP, inflation, Federal Reserve, business and finance.",
    "Technology": "Technology, artificial intelligence, software, computing, digital products and tech companies.",
    "Entertainment": "Entertainment, popular culture, films, music, awards, celebrities, streaming and television.",
    "Science": "Science, weather, climate, astronomy, space missions, health, medicine and scientific discoveries.",
}


def clean_text(value):
    """Apply identical, deterministic text normalization to every question."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def market_text(question, event_title=""):
    q = clean_text(question)
    if not q:
        raise ValueError("A nonempty question is required")
    title = clean_text(event_title)
    return f"Question: {q}" + (f" | Event: {title}" if title else "")


@lru_cache(maxsize=3)
def get_extractor(model_id=MODEL_ID, revision=None, device=-1):
    """Hugging Face feature-extraction pipeline; GPU during build, CPU for Streamlit."""
    from transformers import pipeline
    return pipeline(
        task="feature-extraction", model=model_id, tokenizer=model_id,
        revision=revision, framework="pt", device=device,
    )


def encode_texts(texts, extractor, batch_size=32):
    """Pipeline token embeddings -> attention-mask mean pooling -> L2 normalized vectors.

    Pipeline outputs can include pad tokens when batching. Valid token positions come
    from the SAME tokenizer and truncation settings. Padding direction is respected.
    This function is used unchanged for history, new queries, and topic descriptions.
    """
    texts = list(texts)
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        token_info = extractor.tokenizer(batch, truncation=True, max_length=MAX_TOKENS)
        lengths = [len(ids) for ids in token_info["input_ids"]]
        # Pipeline is the only model execution; no direct model.forward or other model.
        token_states = extractor(batch, batch_size=len(batch), truncation=True,
                                 max_length=MAX_TOKENS)
        if len(token_states) != len(batch):
            raise RuntimeError("Feature-extraction returned an unexpected batch")
        for output, length in zip(token_states, lengths):
            if hasattr(output, "detach"):
                states = output.detach().float().cpu().numpy()
            else:
                states = np.asarray(output, dtype=np.float32)
            if states.ndim == 3 and states.shape[0] == 1:
                states = states[0]
            if states.ndim != 2 or states.shape[0] < length:
                raise RuntimeError(f"Unexpected MiniLM output shape: {states.shape}, valid tokens={length}")
            # Equivalent to (token_embeddings * attention_mask).sum(0) / mask.sum().
            mask = np.zeros(states.shape[0], dtype=np.float32)
            if extractor.tokenizer.padding_side == "left":
                mask[-length:] = 1.0
            else:
                mask[:length] = 1.0
            pooled = (states * mask[:, None]).sum(axis=0) / mask.sum()
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            vectors.append(pooled.astype(np.float32))
    return np.stack(vectors).astype(np.float32)


@lru_cache(maxsize=2)
def load_reference_data(data_dir="data"):
    """Load ONLY exported historical metadata and precomputed embeddings."""
    folder = Path(data_dir).expanduser().resolve()
    with (folder / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("export_version") != 2:
        raise ValueError("Expected PolyPulse v2 index. Run the updated Colab notebook.")
    parquet_names = meta["market_files"]
    embedding_names = meta["embedding_files"]
    if not parquet_names or not embedding_names:
        raise ValueError("Empty manifest")
    markets = pd.concat([pd.read_parquet(folder / name) for name in parquet_names],
                        ignore_index=True)
    embeddings = np.concatenate([np.load(folder / name, allow_pickle=False)
                                 for name in embedding_names], axis=0)
    if not (len(markets) == len(embeddings) == meta["record_count"]):
        raise ValueError("Mismatch between markets, vectors and metadata")
    if embeddings.ndim != 2 or embeddings.shape[1] != 384 or not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix must be finite Nx384")
    if not {"id", "question", "event_title", "volume_usd"}.issubset(markets.columns):
        raise ValueError("Historical index is missing required columns")
    if (not np.isfinite(markets["volume_usd"].to_numpy(dtype=float)).all()
            or (markets["volume_usd"].to_numpy(dtype=float) <= 0).any()):
        raise ValueError("Historical volumes must be recorded positive USD figures")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    if not np.allclose(norms, 1, atol=2e-3):
        raise ValueError("Index embeddings are not normalized")
    return markets, embeddings, meta


@lru_cache(maxsize=4)
def _topic_vectors(model_id, revision, device):
    extractor = get_extractor(model_id, revision, device)
    return encode_texts(list(TOPIC_DESCRIPTIONS.values()), extractor, batch_size=7)


def _suggest_topic(query_text, extractor, model_id, revision, device, query_vector=None):
    """Untrained MiniLM similarity heuristic, NOT a topic classifier."""
    vec = query_vector if query_vector is not None else encode_texts([query_text], extractor)[0]
    scores = _topic_vectors(model_id, revision, device) @ vec
    ix = int(np.argmax(scores))
    return {"topic": list(TOPIC_DESCRIPTIONS)[ix], "similarity": float(scores[ix]),
            "method": "MiniLM semantic similarity heuristic (not a trained classifier)"}


def search_index(question, event_title, markets, embeddings, extractor, *,
                 top_k=10, stats_top_n=30, min_similarity=DEFAULT_MIN_SIMILARITY,
                 min_comparables=DEFAULT_MIN_COMPARABLES, include_topic=True,
                 model_id=MODEL_ID, revision=None, device=-1):
    """Top matches + historical USD statistics over up to stats_top_n qualified markets."""
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if stats_top_n < top_k or stats_top_n < min_comparables:
        raise ValueError("stats_top_n must be >= top_k and min_comparables")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [-1,1]")
    text = market_text(question, event_title)
    vector = encode_texts([text], extractor, batch_size=1)[0]
    # The vectors are L2-normalized; dot product = cosine similarity.
    scores = embeddings.astype(np.float32) @ vector
    qualified = np.flatnonzero(scores >= float(min_similarity))
    ranked = qualified[np.argsort(-scores[qualified], kind="stable")[:max(top_k, stats_top_n)]]
    rows = markets.iloc[ranked]
    matches = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if i >= top_k:
            break
        matches.append({
            "id": str(row["id"]),
            "condition_id": clean_text(row.get("condition_id")),
            "question": str(row["question"]),
            "event_title": clean_text(row.get("event_title")),
            "event_id": clean_text(row.get("event_id")),
            "topic": clean_text(row.get("topic")),
            "market_structure": clean_text(row.get("market_structure")),
            "created_at": str(row.get("created_at", "")),
            "end_date": str(row.get("end_date", "")),
            "volume_usd": float(row["volume_usd"]),
            "similarity": float(scores[ranked[i]]),
            "source": clean_text(row.get("source")),
        })
    relevant_n = min(len(ranked), stats_top_n)
    enough = relevant_n >= min_comparables
    volumes = rows["volume_usd"].iloc[:relevant_n].to_numpy(dtype=np.float64)
    result = {
        "query": clean_text(question), "event_title": clean_text(event_title),
        "status": "ok" if enough else "insufficient_comparable_data",
        "message": "" if enough else (
            f"Only {relevant_n} qualified historical markets; at least {min_comparables} required. "
            "No historical volume range is reported."
        ),
        "minimum_similarity": float(min_similarity),
        "n_above_threshold": int(len(qualified)),
        "n_comparables": int(relevant_n),
        "n_displayed": len(matches),
        "median_volume_usd": float(np.median(volumes)) if enough else None,
        "p25_volume_usd": float(np.percentile(volumes, 25)) if enough else None,
        "p75_volume_usd": float(np.percentile(volumes, 75)) if enough else None,
        "matches": matches,
        "topic_suggestion": (_suggest_topic(text, extractor, model_id, revision, device, vector)
                             if include_topic else None),
        "statistics_note": "Descriptive statistics from actual historical market volumes; not model predictions.",
    }
    return result


def find_similar_markets(question, event_title="", top_k=10, *, data_dir="data",
                         stats_top_n=30, min_similarity=DEFAULT_MIN_SIMILARITY,
                         min_comparables=DEFAULT_MIN_COMPARABLES, include_topic=True):
    """Convenience function for Streamlit, with cached model and cached reference index."""
    markets, embeddings, metadata = load_reference_data(str(Path(data_dir).resolve()))
    model_id = metadata["model_id"]
    revision = metadata.get("model_revision")
    extractor = get_extractor(model_id, revision, device=-1)
    return search_index(question, event_title, markets, embeddings, extractor,
                        top_k=top_k, stats_top_n=stats_top_n,
                        min_similarity=min_similarity, min_comparables=min_comparables,
                        include_topic=include_topic, model_id=model_id,
                        revision=revision, device=-1)
