"""PolyPulse demand classifier: published fine-tuned DeBERTa, not the base model.

No model training, hand-entered thresholds or inferred validation biases occur here.
The notebook's inference contract must be documented in the published
`polypulse_config.json`; unfamiliar config formats fail visibly instead of
silently producing untuned predictions.
"""

from functools import lru_cache
import json
import math
import re

import numpy as np

MODEL_ID = "2610gnam05/polypulse-deberta-v3-base-demand"
LABELS = ("Low", "Medium", "High")


class ModelConfigurationError(ValueError):
    """The published model configuration cannot be safely interpreted."""


def canonical_label(label):
    value = re.sub(r"[^A-Z0-9]", "", str(label).upper())
    for expected in LABELS:
        if value == expected.upper():
            return expected
    raise ModelConfigurationError(f"Unrecognised demand label {label!r}; expected Low/Medium/High.")


def _find(config, *names):
    """Search a few standard config sections, never invent values."""
    scopes = [config]
    for key in ("inference", "prediction", "training", "classification", "demand", "decision", "validation", "thresholds"):
        section = config.get(key)
        if isinstance(section, dict):
            scopes.append(section)
    for scope in scopes:
        for name in names:
            if name in scope and scope[name] is not None:
                return scope[name]
    return None


def model_text(question, event_title=""):
    """Exact string structure used in the user's Colab training notebook."""
    if not question or not question.strip():
        raise ValueError("Market question must not be empty")
    return f"Question: {question.strip()} [SEP] Event: {(event_title or '').strip()}"


def label_order(config, model_id2label):
    """Return the *model output index* to demand-label mapping."""
    explicit_order = _find(config, "label_order", "labels", "class_names", "classes")
    explicit_id2label = _find(config, "id2label")
    explicit_label2id = _find(config, "label2id")
    if isinstance(explicit_order, (list, tuple)):
        ordered = [canonical_label(v) for v in explicit_order]
    elif isinstance(explicit_id2label, dict):
        ordered = [canonical_label(explicit_id2label[str(i)] if str(i) in explicit_id2label else explicit_id2label[i]) for i in range(3)]
    elif isinstance(explicit_label2id, dict):
        mapping = {int(index): canonical_label(label) for label, index in explicit_label2id.items()}
        ordered = [mapping[i] for i in range(3)]
    elif isinstance(model_id2label, dict):
        ordered = [canonical_label(model_id2label.get(i, model_id2label.get(str(i)))) for i in range(3)]
    else:
        raise ModelConfigurationError("No reliable label index ordering in model/config metadata.")
    if len(ordered) != 3 or set(ordered) != set(LABELS):
        raise ModelConfigurationError(f"Invalid label ordering: {ordered!r}")
    # If both publish canonical mappings, guard against an accidental mismatch.
    if isinstance(model_id2label, dict):
        try:
            published = [canonical_label(model_id2label.get(i, model_id2label.get(str(i)))) for i in range(3)]
        except ModelConfigurationError:
            published = None  # Generic LABEL_0 names need explicit config mapping.
        if published is not None and published != ordered:
            raise ModelConfigurationError("Config and model disagree about class label order.")
    return ordered


def max_sequence_length(config):
    length = _find(config, "max_length", "max_seq_length", "max_sequence_length", "max_input_length", "max_input_tokens")
    if length is None or isinstance(length, bool):
        raise ModelConfigurationError("Published config is missing max sequence length.")
    try:
        length_int = int(length)
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError("Invalid max sequence length in published config.") from exc
    if length_int < 8 or length_int > 8192 or float(length) != length_int:
        raise ModelConfigurationError("Invalid max sequence length in published config.")
    return length_int


def volume_cutoffs(config):
    """Two historical USD thresholds separating Low/Medium/High.

    Accepted representations are an ordered pair or an explicitly USD-denominated
    dict. Do not transform numbers whose units cannot be established.
    """
    source = _find(config, "volume_thresholds_usd", "usd_volume_thresholds", "volume_thresholds", "volume_cutoffs_usd", "volume_bins_usd")
    if source is None:
        raise ModelConfigurationError("Published config is missing historical USD volume cutoffs.")
    if isinstance(source, (tuple, list)) and len(source) == 2:
        raw = source
    elif isinstance(source, dict):
        pairs = [
            ("low_medium", "medium_high"),
            ("low_to_medium", "medium_to_high"),
            ("low_max", "medium_max"),
            ("low_max_usd", "medium_max_usd"),
            ("p33", "p67"),
            ("q33", "q67"),
            ("q1", "q2"),
        ]
        raw = next(((source[a], source[b]) for a, b in pairs if a in source and b in source), None)
        if raw is None:
            raise ModelConfigurationError("Unknown USD volume threshold keys; inspect polypulse_config.json.")
    else:
        raise ModelConfigurationError("Unknown USD volume cutoff representation.")
    try:
        low, high = map(float, raw)
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError("USD cutoffs must be numeric.") from exc
    if not all(map(math.isfinite, (low, high))) or not 0 <= low < high:
        raise ModelConfigurationError("USD cutoffs must satisfy 0 <= lower < upper.")
    return low, high


def _bias_vector(raw, order):
    if isinstance(raw, dict):
        mapped = {canonical_label(k): float(v) for k, v in raw.items()}
        if set(mapped) != set(LABELS):
            raise ModelConfigurationError("Adjustment dict must contain Low, Medium and High.")
        arr = np.array([mapped[label] for label in order], dtype=float)
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        arr = np.asarray(raw, dtype=float)
    else:
        raise ModelConfigurationError("Adjustments must be a 3-element vector or per-label dict.")
    if not np.isfinite(arr).all():
        raise ModelConfigurationError("Nonfinite decision adjustments.")
    return arr


def decision_rule(config, order):
    """Read *explicit* validation adjustments and their meaning; no defaults."""
    specs = [
        ("logit_bias", "logit_bias"),
        ("logit_biases", "logit_bias"),
        ("class_logit_biases", "logit_bias"),
        ("best_logit_biases", "logit_bias"),
        ("validation_logit_biases", "logit_bias"),
        ("score_bias", "additive_score"),
        ("score_biases", "additive_score"),
        ("class_score_biases", "additive_score"),
        ("probability_adjustments", "additive_score"),
        ("probability_multipliers", "multiplicative_score"),
        ("class_multipliers", "multiplicative_score"),
        ("decision_adjustments", None),
        ("validation_adjustments", None),
        ("class_adjustments", None),
    ]
    found = [(key, value, implied) for key, implied in specs if (value := _find(config, key)) is not None]
    explicit_method = _find(config, "adjustment_method", "decision_adjustment_method", "score_adjustment_method", "decision_rule")
    if not found:
        if str(explicit_method).lower() in ("none", "no_adjustment", "argmax"):
            return "none", np.zeros(3)
        raise ModelConfigurationError("Validation decision adjustments/method are missing from published config.")
    key, raw, implied = found[0]
    if len(found) > 1 and found[1][1] != raw:
        raise ModelConfigurationError("Multiple conflicting adjustment fields; inspect published config.")
    method = str(explicit_method).lower().strip().replace("-", "_") if explicit_method else implied
    synonyms = {
        "additive_logits": "logit_bias", "logit_add": "logit_bias", "logit_bias": "logit_bias",
        "additive_score": "additive_score", "score_bias": "additive_score", "additive_probabilities": "additive_score",
        "additive_probability": "additive_score", "probability_bias": "additive_score",
        "multiplicative_score": "multiplicative_score", "probability_multiplier": "multiplicative_score",
        "none": "none", "argmax": "none",
    }
    if method not in synonyms:
        raise ModelConfigurationError(
            f"Cannot establish how '{key}' modifies classifier scores. "
            "Add the training notebook's explicit adjustment_method to the published config."
        )
    method = synonyms[method]
    biases = _bias_vector(raw, order)
    if method == "multiplicative_score" and np.any(biases < 0):
        raise ModelConfigurationError("Probability multipliers must be nonnegative.")
    if method == "none" and np.any(biases != 0):
        raise ModelConfigurationError("Nonzero adjustments cannot have 'none' decision method.")
    return method, biases


def tuned_scores(raw_probabilities, adjustment_method, adjustments):
    """Convert published pipeline softmax scores to adjusted scores.

    Log-softmax reconstruction is exactly equivalent to adding biases to the
    original logits (the shared softmax constant cancels).
    """
    probabilities = np.asarray(raw_probabilities, dtype=float)
    adj = np.asarray(adjustments, dtype=float)
    if probabilities.shape != (3,) or adj.shape != (3,):
        raise ValueError("Three scores and three adjustments required.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        raise ValueError("Invalid classification scores from pipeline.")
    if adjustment_method == "logit_bias":
        z = np.log(np.maximum(probabilities, np.finfo(float).tiny)) + adj
        z = np.exp(z - z.max())
    elif adjustment_method == "additive_score":
        z = np.maximum(probabilities + adj, 0.0)
    elif adjustment_method == "multiplicative_score":
        z = probabilities * adj
    elif adjustment_method == "none":
        z = probabilities
    else:
        raise ModelConfigurationError(f"Unrecognized adjustment method {adjustment_method!r}.")
    if not np.isfinite(z).all() or z.sum() <= 0:
        raise ModelConfigurationError("Adjusted classification scores cannot be normalised.")
    return z / z.sum()


def volume_band(label, cutoffs):
    low, high = cutoffs
    if label == "Low":
        return f"Below ${low:,.0f}"
    if label == "Medium":
        return f"${low:,.0f} to ${high:,.0f}"
    if label == "High":
        return f"${high:,.0f} or above"
    raise ValueError(f"Unknown demand label {label}")


@lru_cache(maxsize=1)
def load_published_config():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_ID, filename="polypulse_config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ModelConfigurationError("Published config must be a JSON object.")
    return config


@lru_cache(maxsize=1)
def load_classifier():
    from transformers import pipeline

    # Model AND tokenizer are loaded from the fine-tuned model repository.
    return pipeline(task="text-classification", model=MODEL_ID, tokenizer=MODEL_ID,
                    device=-1, framework="pt")


def predict_demand(question, event_title="", classifier=None, config=None):
    """Run one CPU prediction and apply published validation decision adjustment.

    Optional injected classifier/config enable pure unit tests; the Streamlit app
    does not override the published configuration.
    """
    if config is None:
        config = load_published_config()
    if classifier is None:
        classifier = load_classifier()
    order = label_order(config, classifier.model.config.id2label)
    length = max_sequence_length(config)
    bounds = volume_cutoffs(config)
    adjustment_method, adjustment_values = decision_rule(config, order)
    output = classifier(model_text(question, event_title), truncation=True,
                        max_length=length, top_k=None)
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, list):
        raise ValueError("Classification pipeline did not return all class scores.")
    raw_by_name = {}
    for item in output:
        label = str(item["label"])
        if re.fullmatch(r"LABEL_\d+", label.upper()):
            idx = int(label.split("_")[-1])
            if idx >= len(order):
                raise ModelConfigurationError("Classifier produced an unknown label index.")
            parsed = order[idx]
        else:
            parsed = canonical_label(label)
        if parsed in raw_by_name:
            raise ModelConfigurationError("Classifier returned duplicate class scores.")
        raw_by_name[parsed] = float(item["score"])
    if set(raw_by_name) != set(LABELS):
        raise ValueError("Classifier must return exactly Low/Medium/High scores.")
    ordered_raw = np.array([raw_by_name[label] for label in order], dtype=float)
    adjusted = tuned_scores(ordered_raw, adjustment_method, adjustment_values)
    predicted = order[int(np.argmax(adjusted))]
    return {
        "label": predicted,
        "model_scores": {label: float(adjusted[i]) for i, label in enumerate(order)},
        "raw_model_scores": {label: float(ordered_raw[i]) for i, label in enumerate(order)},
        "adjustment_method": adjustment_method,
        "volume_band": volume_band(predicted, bounds),
        "volume_thresholds_usd": list(bounds),
        "max_sequence_length": length,
    }
