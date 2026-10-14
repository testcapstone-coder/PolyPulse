"""PolyPulse fine-tuned prediction-market question validity gate.

This module loads the PolyPulse binary VALID/INVALID classifier fine-tuned from
cross-encoder/nli-deberta-v3-small. It runs before demand prediction and
historical comparison.
"""

from functools import lru_cache
import json
import re

import numpy as np

MODEL_ID = "2610gnam05/nli-deberta-v3-small-demand"
CONFIG_FILENAME = "polypulse_validity_config.json"
DEFAULT_MAX_LENGTH = 128
DEFAULT_VALID_THRESHOLD = 0.50


class ValidityModelError(RuntimeError):
    """Raised when the published validity model/config cannot be interpreted safely."""


def _clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_label(value):
    value = re.sub(r"[^A-Z0-9]+", "", str(value).upper())
    if value == "VALID":
        return "VALID"
    if value == "INVALID":
        return "INVALID"
    raise ValidityModelError(
        f"Unexpected validity label {value!r}; expected VALID/INVALID."
    )


@lru_cache(maxsize=1)
def load_validity_config():
    """Load the threshold/max-length exported by the training notebook."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_ID, filename=CONFIG_FILENAME)
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValidityModelError("Validity configuration must be a JSON object.")

    threshold = float(config.get("valid_threshold", DEFAULT_VALID_THRESHOLD))
    max_length = int(config.get("max_length", DEFAULT_MAX_LENGTH))

    if not 0.0 <= threshold <= 1.0:
        raise ValidityModelError("valid_threshold must be between 0 and 1.")
    if not 8 <= max_length <= 1024:
        raise ValidityModelError("max_length is outside the accepted range.")

    config["valid_threshold"] = threshold
    config["max_length"] = max_length
    return config


@lru_cache(maxsize=1)
def load_validator():
    """Load the fine-tuned binary classifier once per Python process, on CPU."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def _label_index(model, wanted):
    id2label = getattr(model.config, "id2label", None) or {}
    matches = []
    for idx, label in id2label.items():
        try:
            parsed = _canonical_label(label)
        except ValidityModelError:
            continue
        if parsed == wanted:
            matches.append(int(idx))

    if len(matches) != 1:
        raise ValidityModelError(
            f"Could not identify exactly one {wanted} output in model metadata."
        )
    return matches[0]


def validate_market_question(question, event_title="", validator=None, config=None):
    """Return direct binary classifier scores for a proposed market question.

    `event_title` is accepted for compatibility with the app API but is not fed to
    this model because the fine-tuning objective is question validity itself.
    """
    import torch

    text = _clean_text(question)
    if not text:
        raise ValueError("Market question must not be empty.")

    tokenizer, model = validator or load_validator()
    config = config or load_validity_config()
    max_length = int(config["max_length"])
    threshold = float(config["valid_threshold"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0].float()

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    if probs.ndim != 1 or not np.isfinite(probs).all():
        raise ValidityModelError("Validity model returned invalid scores.")

    valid_idx = _label_index(model, "VALID")
    invalid_idx = _label_index(model, "INVALID")

    if valid_idx >= len(probs) or invalid_idx >= len(probs):
        raise ValidityModelError("Published label indices exceed model output size.")

    valid_score = float(probs[valid_idx])
    invalid_score = float(probs[invalid_idx])
    is_valid = bool(valid_score >= threshold)

    return {
        "is_valid": is_valid,
        "label": "Valid" if is_valid else "Invalid",
        "validity_score": valid_score,
        "valid_score": valid_score,
        "invalid_score": invalid_score,
        "validity_threshold": threshold,
        "model_id": MODEL_ID,
        "base_model": config.get("base_model", "cross-encoder/nli-deberta-v3-small"),
        "method": "fine-tuned binary prediction-market validity classifier",
        "max_sequence_length": max_length,
    }
