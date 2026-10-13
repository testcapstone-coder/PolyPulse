"""PolyPulse MiniLM historical market search (CPU-friendly, no model training).

Only the three precomputed files in polypulse_historical/ are needed at runtime.
Historical market USD volumes always come from historical_markets.parquet.
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
DEFAULT_MIN_SIMILARITY = 0.58
DEFAULT_MIN_COMPARABLES = 5

# Category suggestions are semantic comparisons, NOT trained classification.
TOPIC_DESCRIPTIONS = {
    "Politics": "Elections, politicians, presidents, governments, parliament, policy and geopolitical events.",
    "Cryptocurrency": "Bitcoin, Ethereum, crypto token prices, cryptocurrency exchanges and blockchain markets.",
    "Sports": "Football, basketball, tennis, baseball, esports, teams, athletes and match results.",
    "Economics": "Inflation, GDP, interest rates, central banks, employment, recession and financial markets.",
    "Technology": "Artificial intelligence, software, computers, startups, digital products and technology companies.",
    "Entertainment": "Music, movies, celebrities, television, awards, streaming and popular culture.",
    "Science": "Scientific discoveries, space exploration, medicine, climate research and scientific breakthroughs.",
}


def clean_text(value):
    """Use the same Unicode and whitespace normalization for index and query."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def market_text(question, event_title=""):
    """Identical question+optional-title template for history and new questions."""
    question = clean_text(question)
    if not question:
        raise ValueError("Market question must not be empty")
    title = clean_text(event_title)
    return f"Question: {question}" + (f" | Event: {title}" if title else "")


@lru_cache(maxsize=3)
def get_extractor(model_id=MODEL_ID, revision=None, device=-1):
    """Load exactly one Hugging Face feature-extraction pipeline per setting."""
    from transformers import pipeline

    return pipeline(
        task="feature-extraction",
        model=model_id,
        tokenizer=model_id,
        revision=revision,
        framework="pt",
        device=device,
    )


def encode_texts(texts, extractor, batch_size=32):
    """Mean-pool token states with attention masks, then L2-normalize.

    The pipeline computes token embeddings. We reconstruct its tokenizer attention
    masks from exact token lengths and its padding direction. There is no
    sentence-transformers dependency and no additional model.
    """
    texts = list(texts)
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = extractor.tokenizer(batch, truncation=True, max_length=MAX_TOKENS)
        valid_lengths = [len(ids) for ids in encoded["input_ids"]]
        # transformers>=4.57 supports tokenize_kwargs in feature-extraction.
        token_outputs = extractor(
            batch,
            truncation=True,
            tokenize_kwargs={"max_length": MAX_TOKENS},
            batch_size=len(batch),
            return_tensors=True,
        )
        if len(token_outputs) != len(batch):
            raise RuntimeError("Feature-extraction batch size mismatch")
        for output, valid_length in zip(token_outputs, valid_lengths):
            states = output.detach().float().cpu().numpy() if hasattr(output, "detach") else np.asarray(output, dtype=np.float32)
            if states.ndim == 3 and states.shape[0] == 1:
                states = states[0]
            if states.ndim != 2 or states.shape[0] < valid_length:
                raise RuntimeError("Unexpected MiniLM token embedding shape")
            # Equivalent to attention-mask mean pooling; ignore padding tokens.
            attention_mask = np.zeros(states.shape[0], dtype=np.float32)
            if extractor.tokenizer.padding_side == "left":
                attention_mask[-valid_length:] = 1.0
            else:
                attention_mask[:valid_length] = 1.0
            pooled = (states * attention_mask[:, None]).sum(axis=0) / max(attention_mask.sum(), 1.0)
            pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-12)
            all_vectors.append(pooled.astype(np.float32))
    return np.vstack(all_vectors).astype(np.float32)


@lru_cache(maxsize=3)
def load_reference_data(data_dir="polypulse_historical"):
    """Read a precomputed index only; never read or download source markets."""
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    markets = pd.read_parquet(data_dir / "historical_markets.parquet")
    embeddings = np.load(data_dir / "market_embeddings.npy", mmap_mode="r")
    if len(markets) != len(embeddings):
        raise ValueError("Market records and embedding rows differ")
    if embeddings.ndim != 2 or embeddings.shape[1] != metadata["embedding_dimension"]:
        raise ValueError("Inconsistent embedding dimensions")
    if metadata["record_count"] != len(markets):
        raise ValueError("Record count does not match metadata")
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    if not {"id", "question", "event_title", "volume_usd"}.issubset(markets.columns):
        raise ValueError("Required historical fields missing")
    return markets, embeddings, metadata


@lru_cache(maxsize=4)
def _topic_vectors(model_id, revision, device):
    extractor = get_extractor(model_id, revision, device)
    return encode_texts(TOPIC_DESCRIPTIONS.values(), extractor, batch_size=7)


def search_index(question, event_title, markets, embeddings, extractor, *,
                 top_k=10, min_similarity=DEFAULT_MIN_SIMILARITY,
                 min_comparables=DEFAULT_MIN_COMPARABLES,
                 include_topic=True, model_id=MODEL_ID, model_revision=None,
                 device=-1):
    """Search a loaded index; statistics use ONLY the displayed qualified matches."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not -1.0 <= float(min_similarity) <= 1.0:
        raise ValueError("min_similarity must be in [-1, 1]")
    if not isinstance(min_comparables, int) or min_comparables < 2:
        raise ValueError("min_comparables must be at least 2")
    if top_k < min_comparables:
        raise ValueError("top_k must be >= min_comparables")
    query_text = market_text(question, event_title)
    query_vector = encode_texts([query_text], extractor, batch_size=1)[0]
    # Dot product of normalized embeddings is cosine similarity (pure NumPy).
    similarities = np.asarray(embeddings @ query_vector, dtype=np.float32)
    relevant = np.flatnonzero(similarities >= float(min_similarity))
    # Stable descending rank: similarity first, input record order on ties.
    ranked = relevant[np.argsort(-similarities[relevant], kind="stable")[:top_k]]
    matches = []
    for row_idx in ranked:
        row = markets.iloc[int(row_idx)]
        matches.append({
            "id": str(row["id"]),
            "question": str(row["question"]),
            "event_title": str(row["event_title"]),
            "event_id": str(row.get("event_id", "")),
            "event_slug": str(row.get("event_slug", "")),
            "created_at": str(row.get("created_at", "")),
            "end_date": str(row.get("end_date", "")),
            "volume_usd": float(row["volume_usd"]),
            "similarity": float(similarities[row_idx]),
        })
    enough = len(matches) >= min_comparables
    volumes = np.array([item["volume_usd"] for item in matches], dtype=np.float64)
    response = {
        "query": clean_text(question),
        "event_title": clean_text(event_title),
        "status": "ok" if enough else "insufficient_comparable_data",
        "message": "" if enough else f"Only {len(matches)} relevant markets; at least {min_comparables} required. No volume range is reported.",
        "minimum_similarity": float(min_similarity),
        "minimum_comparables": min_comparables,
        "n_reference_markets": len(markets),
        "n_above_threshold": len(relevant),
        "n_comparables": len(matches),
        "statistics_population": "returned matches meeting the similarity threshold",
        "matches": matches,
        "median_volume_usd": float(np.median(volumes)) if enough else None,
        "p25_volume_usd": float(np.percentile(volumes, 25)) if enough else None,
        "p75_volume_usd": float(np.percentile(volumes, 75)) if enough else None,
    }
    if include_topic:
        categories = list(TOPIC_DESCRIPTIONS)
        vectors = _topic_vectors(model_id, model_revision, device)
        scores = vectors @ query_vector
        best = int(np.argmax(scores))
        response["topic_suggestion"] = {
            "topic": categories[best],
            "similarity": float(scores[best]),
            "method": "MiniLM semantic similarity heuristic (not a trained classifier)",
        }
    return response


def find_similar_markets(question, event_title="", top_k=10, *,
                         data_dir="polypulse_historical",
                         min_similarity=DEFAULT_MIN_SIMILARITY,
                         min_comparables=DEFAULT_MIN_COMPARABLES,
                         include_topic=True):
    """Streamlit-ready entry point returning a plain Python dictionary."""
    markets, embeddings, metadata = load_reference_data(str(Path(data_dir).resolve()))
    model_id = metadata["model_id"]
    model_revision = metadata.get("model_revision")
    extractor = get_extractor(model_id, model_revision, device=-1)  # CPU
    return search_index(
        question, event_title, markets, embeddings, extractor,
        top_k=top_k, min_similarity=min_similarity,
        min_comparables=min_comparables, include_topic=include_topic,
        model_id=model_id, model_revision=model_revision, device=-1,
    )
