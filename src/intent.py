"""Load a numeric BANKING77 classifier and score one frozen sentence vector."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.embedding import PROJECT_ROOT, load_config

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts"


def load_classifier(
    directory: Path = MODEL_DIRECTORY,
    encoder_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load trained numeric arrays with pickle disabled and check compatibility."""
    metadata = json.loads((directory / "intent.json").read_text(encoding="utf-8"))
    config = load_config() if encoder_config is None else encoder_config
    if (
        metadata["encoder_model"] != config["model_id"]
        or metadata["encoder_revision"] != config["revision"]
        or metadata["embedding_dimension"] != config["embedding_dimension"]
    ):
        raise ValueError("Classifier and encoder configuration do not match.")
    artifact = directory / "intent.npz"
    labels = metadata["classes"]
    if (
        len(labels) != 77
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("Classifier must contain 77 unique intent labels.")
    with np.load(artifact, allow_pickle=False) as arrays:
        weights = arrays["weights"].copy()
        bias = arrays["bias"].copy()
        classes = arrays["classes"].copy()
    if (
        weights.shape != (77, config["embedding_dimension"])
        or bias.shape != (77,)
        or not np.array_equal(classes, np.arange(77))
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Classifier arrays are invalid.")
    thresholds = metadata["uncertainty"]
    for key in ("confidence_threshold", "margin_threshold"):
        value = thresholds[key]
        if (
            not isinstance(value, (float, int))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Classifier uncertainty thresholds are invalid.")
    return {"weights": weights, "bias": bias, "classes": labels, "metadata": metadata}


def classify_vector(vector: np.ndarray, classifier: dict[str, Any]) -> dict[str, Any]:
    """Return top topics and an empirical uncertainty flag, not an OOD guarantee."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if (
        vector.shape != (classifier["weights"].shape[1],)
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3)
    ):
        raise ValueError("Intent classification needs one normalized encoder vector.")
    logits = classifier["weights"] @ vector + classifier["bias"]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    ranking = np.argsort(-probabilities, kind="stable")
    confidence = float(probabilities[ranking[0]])
    margin = float(confidence - probabilities[ranking[1]])
    thresholds = classifier["metadata"]["uncertainty"]
    return {
        "intent": classifier["classes"][ranking[0]],
        "confidence": confidence,
        "margin": margin,
        "top3": [
            {
                "intent": classifier["classes"][index],
                "score": float(probabilities[index]),
            }
            for index in ranking[:3]
        ],
        "review_required": confidence < thresholds["confidence_threshold"]
        or margin < thresholds["margin_threshold"],
    }
