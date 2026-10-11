"""Reproduce the fixed MPNet classifier without replacing the shipped artifact."""

import argparse
import csv
import hashlib
import json
import sys
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.embedding import (
    MODEL_DIRECTORY,
    PROJECT_ROOT,
    encode_texts,
    load_config,
    open_encoder,
)
from src.intent import load_classifier

DATA_DIRECTORY = PROJECT_ROOT / "data/banking77"
OUTPUT_DIRECTORY = PROJECT_ROOT / "models/retrained-intent"
REPORT_PATH = PROJECT_ROOT / "reports/retrained_intent.json"
DATASET_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
DATASET_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
FILE_HASHES = {
    "categories.json": (
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"
    ),
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
C_VALUES = (3.0, 10.0, 30.0, 100.0)
SEED = 42
# The original corpus used batches of 16. Quantized vectors depend on batching;
# runtime question/policy encoding keeps its separate app configuration.
TRAINING_BATCH_SIZE = 16


def digest(data: bytes) -> str:
    """Hash a dataset, configuration, or numeric artifact for traceability."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Save a readable result; repeated runs replace the previous reproduction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_dataset() -> dict[str, Any]:
    """Download missing public files and check the fixed revision's checksums."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected_hash in FILE_HASHES.items():
        path = DATA_DIRECTORY / name
        url = f"{DATASET_URL}/{DATASET_REVISION}/banking_data/{name}"
        if not path.exists():
            request = Request(
                url, headers={"User-Agent": "financial-support-assistant"}
            )
            with urlopen(request, timeout=60) as response:
                data = response.read(4_000_001)
            if len(data) > 4_000_000 or digest(data) != expected_hash:
                raise ValueError(f"Unexpected dataset download: {name}")
            path.write_bytes(data)
        data = path.read_bytes()
        if digest(data) != expected_hash:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        files[name] = {"url": url, "bytes": len(data), "sha256": expected_hash}
    return {
        "dataset": "BANKING77",
        "revision": DATASET_REVISION,
        "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "license": "CC-BY-4.0",
        "files": files,
    }


def read_split(
    name: str, labels: list[str], expected_count: int
) -> tuple[list[str], np.ndarray]:
    """Validate official rows and preserve the published category order."""
    reader = csv.DictReader(
        StringIO((DATA_DIRECTORY / name).read_text(encoding="utf-8"))
    )
    if reader.fieldnames != ["text", "category"]:
        raise ValueError(f"Unexpected dataset columns: {name}")
    texts = []
    targets = []
    for row in reader:
        if not row["text"].strip() or row["category"] not in labels:
            raise ValueError(f"Invalid dataset row: {name}")
        texts.append(row["text"])
        targets.append(labels.index(row["category"]))
    if len(texts) != expected_count or len(set(targets)) != 77:
        raise ValueError(f"Unexpected dataset size or missing intent: {name}")
    return texts, np.array(targets, dtype=np.int64)


def encoder_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Include actual ONNX/tokenizer bytes in the training cache identity."""
    return {
        "config": config,
        "encoding_batch_size": TRAINING_BATCH_SIZE,
        "model_sha256": digest((MODEL_DIRECTORY / config["model_file"]).read_bytes()),
        "tokenizer_sha256": digest((MODEL_DIRECTORY / "tokenizer.json").read_bytes()),
    }


def cache_fingerprint(texts: list[str], identity: dict[str, Any]) -> str:
    """Bind cached vectors to the exact encoder and ordered input texts."""
    value = {"encoder": identity, "texts": texts}
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def encode_split(
    texts: list[str], config: dict[str, Any], identity: dict[str, Any], name: str
) -> tuple[np.ndarray, int]:
    """Reuse matching vectors or reproduce the original 16-row corpus batches."""
    cache = DATA_DIRECTORY / f"{name}_mpnet_vectors.npz"
    fingerprint = cache_fingerprint(texts, identity)
    shape = (len(texts), config["embedding_dimension"])
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            if str(arrays["fingerprint"]) == fingerprint:
                vectors = arrays["vectors"].copy()
                valid = (
                    vectors.shape == shape
                    and vectors.dtype == np.float32
                    and np.isfinite(vectors).all()
                    and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
                )
                if valid:
                    print(f"{name}: reusing checked vectors", flush=True)
                    return vectors, int(arrays["truncated_count"])
    session, tokenizer = open_encoder(config)
    chunks = []
    truncated = 0
    for start in range(0, len(texts), 512):
        vectors, diagnostics = encode_texts(
            texts[start : start + 512], session, tokenizer, TRAINING_BATCH_SIZE
        )
        chunks.append(vectors)
        truncated += sum(item["truncated"] for item in diagnostics)
        print(
            f"{name}: encoded {min(start + 512, len(texts))}/{len(texts)}", flush=True
        )
    vectors = np.concatenate(chunks)
    np.savez_compressed(
        cache, vectors=vectors, fingerprint=fingerprint, truncated_count=truncated
    )
    return vectors, truncated


def classifier(c_value: float) -> LogisticRegression:
    """Keep the original learner fixed; vary only regularization strength."""
    return LogisticRegression(
        C=c_value, solver="lbfgs", max_iter=1500, random_state=SEED
    )


def scores(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Measure intent prediction, not downstream answer correctness."""
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    return {
        "accuracy": float(accuracy_score(targets, ranking[:, 0])),
        "macro_f1": float(f1_score(targets, ranking[:, 0], average="macro")),
        "top3_accuracy": float(
            np.any(ranking[:, :3] == targets[:, None], axis=1).mean()
        ),
    }


def cross_validate(vectors: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    """Select C from three fixed stratified folds of official training rows."""
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            vectors, targets
        )
    )
    results = []
    for c_value in C_VALUES:
        fold_scores = []
        for training, validation in folds:
            model = classifier(c_value)
            model.fit(vectors[training], targets[training])
            fold_scores.append(
                scores(model.predict_proba(vectors[validation]), targets[validation])
            )
        mean = float(np.mean([item["macro_f1"] for item in fold_scores]))
        results.append({"C": c_value, "folds": fold_scores, "mean_macro_f1": mean})
        print(f"CV C={c_value:g}: macro F1={mean:.4f}", flush=True)
    return results


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Keep maximum validation coverage with >=95% accuracy and >=20% retained."""
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    confidence = np.take_along_axis(probabilities, ranking[:, :1], axis=1).ravel()
    runner_up = np.take_along_axis(probabilities, ranking[:, 1:2], axis=1).ravel()
    margin = confidence - runner_up
    correct = ranking[:, 0] == targets
    candidates = []
    for confidence_threshold in np.arange(0.2, 1.0, 0.05):
        for margin_threshold in np.arange(0, 0.85, 0.05):
            kept = (confidence >= confidence_threshold) & (margin >= margin_threshold)
            count = int(kept.sum())
            if count >= len(targets) * 0.2 and float(correct[kept].mean()) >= 0.95:
                candidates.append(
                    (
                        count,
                        float(correct[kept].mean()),
                        float(confidence_threshold),
                        float(margin_threshold),
                    )
                )
    if candidates:
        count, accuracy, confidence_threshold, margin_threshold = max(
            candidates, key=lambda row: (row[0], row[1], -row[2], -row[3])
        )
    else:
        count, accuracy, confidence_threshold, margin_threshold = 0, None, 1.0, 1.0
    return {
        "confidence_threshold": round(confidence_threshold, 4),
        "margin_threshold": round(margin_threshold, 4),
        "validation_retained_count": count,
        "validation_total": len(targets),
        "validation_coverage": count / len(targets),
        "validation_selective_accuracy": accuracy,
        "selection_split": "train-only stratified 80/20 validation",
        "limitation": "Uncalibrated scores; not an out-of-domain guarantee.",
    }


def runtime_scores(
    vectors: np.ndarray, targets: np.ndarray, exported: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the exported weights with the app's float64 stable softmax."""
    logits = np.asarray(vectors, dtype=np.float64) @ exported["weights"].T
    logits += exported["bias"]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    confidence = np.take_along_axis(probabilities, ranking[:, :1], axis=1).ravel()
    runner_up = np.take_along_axis(probabilities, ranking[:, 1:2], axis=1).ravel()
    thresholds = exported["metadata"]["uncertainty"]
    kept = (confidence >= thresholds["confidence_threshold"]) & (
        confidence - runner_up >= thresholds["margin_threshold"]
    )
    return {
        "count": len(targets),
        **scores(probabilities, targets),
        "selective": {
            "retained_count": int(kept.sum()),
            "coverage": float(kept.mean()),
            "accuracy": float((ranking[kept, 0] == targets[kept]).mean())
            if kept.any()
            else None,
        },
    }


def fit_and_export(
    vectors: np.ndarray, targets: np.ndarray, labels: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    """Select C and uncertainty from training rows, then export a separate refit."""
    candidates = cross_validate(vectors, targets)
    selected = max(candidates, key=lambda row: (row["mean_macro_f1"], -row["C"]))
    fit_indices, validation_indices = train_test_split(
        np.arange(len(targets)), test_size=0.2, random_state=SEED, stratify=targets
    )
    threshold_model = classifier(selected["C"])
    threshold_model.fit(vectors[fit_indices], targets[fit_indices])
    thresholds = select_threshold(
        threshold_model.predict_proba(vectors[validation_indices]),
        targets[validation_indices],
    )
    final_model = classifier(selected["C"])
    final_model.fit(vectors, targets)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIRECTORY / "intent.npz"
    np.savez_compressed(
        artifact,
        weights=final_model.coef_,
        bias=final_model.intercept_,
        classes=final_model.classes_,
    )
    metadata = {
        "encoder_model": config["model_id"],
        "encoder_revision": config["revision"],
        "embedding_dimension": config["embedding_dimension"],
        "classes": labels,
        "uncertainty": thresholds,
        "selected_C": selected["C"],
        "seed": SEED,
        "training_count": len(targets),
        "artifact_sha256": digest(artifact.read_bytes()),
    }
    write_json(OUTPUT_DIRECTORY / "intent.json", metadata)
    return {
        "exported": load_classifier(OUTPUT_DIRECTORY, config),
        "cv_candidates": candidates,
        "threshold_fit_count": len(fit_indices),
        "threshold_validation_count": len(validation_indices),
    }


def write_report(
    provenance: dict[str, Any],
    identity: dict[str, Any],
    training: dict[str, Any],
    metrics: dict[str, Any],
    resources: dict[str, Any],
    overlapping_texts: int,
) -> None:
    """Save the experiment, source identities, and exposed-test limitation."""
    exported = training["exported"]
    metadata = exported["metadata"]
    write_json(
        REPORT_PATH,
        {
            "task": "77-class banking intent classification, not answer correctness",
            "dataset": provenance,
            "encoder": identity,
            "artifact": {
                "path": "models/retrained-intent/intent.npz",
                "sha256": metadata["artifact_sha256"],
                "weight_dtype": str(exported["weights"].dtype),
            },
            "protocol": {
                "seed": SEED,
                "C_selection": (
                    "3-fold stratified training CV; mean macro F1; ties lower C"
                ),
                "cv_candidates": training["cv_candidates"],
                "selected_C": metadata["selected_C"],
                "refit_count": metadata["training_count"],
                "threshold_fit_count": training["threshold_fit_count"],
                "threshold_validation_count": training["threshold_validation_count"],
                "test_role": (
                    "Previously inspected official test: development benchmark, "
                    "not fresh blind evaluation"
                ),
                "overlapping_exact_texts_train_test": overlapping_texts,
            },
            "uncertainty": metadata["uncertainty"],
            "test_development_benchmark": metrics,
            "resources": resources,
            "software": {
                name: version(name)
                for name in ("numpy", "scikit-learn", "onnxruntime", "threadpoolctl")
            },
            "limitations": [
                "Neither MPNet nor Qwen is fine-tuned; "
                "only logistic regression is trained.",
                "Closed-set intent classification does not establish policy coverage, "
                "account access, or reply grounding.",
                "Runtime artifacts are written separately; "
                "the app keeps using artifacts/intent.npz.",
            ],
        },
    )


def train() -> None:
    """Run train-only selection, refit all training rows, then measure exposed test."""
    started = perf_counter()
    provenance = prepare_dataset()
    labels = json.loads(
        (DATA_DIRECTORY / "categories.json").read_text(encoding="utf-8")
    )
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("BANKING77 needs exactly 77 distinct categories.")
    train_texts, train_targets = read_split("train.csv", labels, 10003)
    test_texts, test_targets = read_split("test.csv", labels, 3080)
    config = load_config()
    identity = encoder_identity(config)
    train_vectors, train_truncated = encode_split(
        train_texts, config, identity, "train"
    )
    with threadpool_limits(limits=2):
        training = fit_and_export(train_vectors, train_targets, labels, config)
        test_vectors, test_truncated = encode_split(
            test_texts, config, identity, "test"
        )
        metrics = runtime_scores(test_vectors, test_targets, training["exported"])
    resources = {
        "encoder_threads": 2,
        "blas_threads_max": 2,
        "encoding_batch_size": TRAINING_BATCH_SIZE,
        "train_truncated_count": train_truncated,
        "test_truncated_count": test_truncated,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    write_report(
        provenance,
        identity,
        training,
        metrics,
        resources,
        len(set(train_texts) & set(test_texts)),
    )
    print(
        f"Selected C={training['exported']['metadata']['selected_C']:g}; "
        f"development benchmark={metrics}"
    )
    print(
        "Saved models/retrained-intent and reports/retrained_intent.json; "
        "app unchanged."
    )


def main() -> int:
    """Run one reproducible classifier experiment without calling an LLM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        train()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Intent training failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
