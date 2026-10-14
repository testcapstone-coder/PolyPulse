"""PolyPulse prediction-market question validity gate.

This module uses a dedicated Hugging Face zero-shot NLI model to decide whether a user's
input is sufficiently clear and resolvable to continue to PolyPulse demand
prediction and historical analysis.

The validator is intentionally independent of MiniLM similarity and the
fine-tuned Low/Medium/High demand classifier.
"""

from functools import lru_cache
import re

import numpy as np

MODEL_ID = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
MAX_LENGTH = 256

# The two hypotheses are evaluated independently with the NLI model.
VALID_HYPOTHESIS = (
    "This text describes a meaningful prediction-market proposition "
    "with an outcome that can be objectively resolved."
)
INVALID_HYPOTHESIS = (
    "This text is meaningless, incomplete, or unsuitable as a resolvable "
    "prediction-market proposition."
)

# Conservative fail-closed gate. This is an operating threshold, not a calibrated
# probability threshold. It should be benchmarked on a small held-out validity set.
VALIDITY_THRESHOLD = 0.45
MIN_VALID_ENTAILMENT = 0.20


class ValidityModelError(RuntimeError):
    """Raised when the NLI model metadata/output cannot be interpreted safely."""


def _normalise_label(value):
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def _entailment_index(model):
    """Find the entailment output index from published model metadata."""
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if _normalise_label(label) == "ENTAILMENT":
            return int(idx)

    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        if _normalise_label(label) == "ENTAILMENT":
            return int(idx)

    raise ValidityModelError(
        "The validity model does not publish a reliable ENTAILMENT label mapping."
    )


@lru_cache(maxsize=1)
def load_validator():
    """Load the dedicated ModernBERT NLI validator once per Python process, on CPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def _entailment_score(text, hypothesis, validator=None):
    """Return the NLI entailment softmax score for one premise/hypothesis pair."""
    import torch

    tokenizer, model = validator or load_validator()
    inputs = tokenizer(
        text,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    if probs.ndim != 1 or not np.isfinite(probs).all():
        raise ValidityModelError("Validity model returned invalid scores.")

    entailment_idx = _entailment_index(model)
    if entailment_idx >= len(probs):
        raise ValidityModelError("Entailment index is outside the model output.")

    return float(probs[entailment_idx])


def validate_market_question(question, event_title="", validator=None):
    """Check whether input is a clear, resolvable prediction-market proposition.

    The model evaluates a positive and negative NLI hypothesis independently.
    The final validity score is their relative evidence:

        valid_entailment / (valid_entailment + invalid_entailment)

    This is a routing score, not a calibrated probability.
    """
    question = str(question or "").strip()
    event_title = str(event_title or "").strip()
    if not question:
        raise ValueError("Market question must not be empty.")

    premise = question
    if event_title:
        premise = f"Question: {question}\nEvent context: {event_title}"

    valid_entailment = _entailment_score(
        premise, VALID_HYPOTHESIS, validator=validator
    )
    invalid_entailment = _entailment_score(
        premise, INVALID_HYPOTHESIS, validator=validator
    )

    total = valid_entailment + invalid_entailment
    if total <= 0 or not np.isfinite(total):
        raise ValidityModelError("Could not compute a finite validity evidence score.")

    validity_score = valid_entailment / total
    is_valid = bool(
        validity_score >= VALIDITY_THRESHOLD
        and valid_entailment >= MIN_VALID_ENTAILMENT
    )

    return {
        "is_valid": is_valid,
        "label": "Valid" if is_valid else "Invalid",
        "validity_score": float(validity_score),
        "valid_entailment": float(valid_entailment),
        "invalid_entailment": float(invalid_entailment),
        "validity_threshold": float(VALIDITY_THRESHOLD),
        "minimum_valid_entailment": float(MIN_VALID_ENTAILMENT),
        "model_id": MODEL_ID,
        "method": "ModernBERT dual-hypothesis zero-shot NLI validity gate",
    }
