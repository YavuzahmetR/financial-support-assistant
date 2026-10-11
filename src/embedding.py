"""Frozen MPNet sentence encoder using two CPU threads."""

import json
import os
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/embedding.json"
MODEL_DIRECTORY = PROJECT_ROOT / "models/all-mpnet-base-v2"


def load_config() -> dict[str, Any]:
    """Read the pinned CPU encoder configuration."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def open_encoder(config: dict[str, Any]) -> tuple[ort.InferenceSession, Tokenizer]:
    """Load the local quantized model and its tokenizer."""
    directory = MODEL_DIRECTORY
    model_path = directory / config["model_file"]
    tokenizer_path = directory / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("MPNet files are missing. Run the embedding setup.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:
        raise ValueError(
            "MPNet files could not be loaded. Check the installation."
        ) from error
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("An unexpected execution provider was selected.")
    tokenizer.enable_truncation(max_length=config["max_sequence_length"])
    pad_token = "<pad>"
    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError("The encoder tokenizer has no configured padding token.")
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return session, tokenizer


def mean_pool_and_normalize(
    token_vectors: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Average real tokens, excluding padding, then normalize each sentence."""
    mask = attention_mask[..., None].astype(np.float32)
    pooled = (token_vectors * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(pooled).all() or np.any(norms <= 0):
        raise ValueError("The model produced invalid sentence vectors.")
    return pooled / norms


def encode_texts(
    texts: list[str],
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return sentence vectors and disclose any input truncation."""
    if not texts or any(not text.strip() for text in texts) or batch_size < 1:
        raise ValueError("Encoding needs nonempty texts and a positive batch size.")
    vectors = []
    diagnostics = []
    for start in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(texts[start : start + batch_size])
        inputs = {
            "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [item.attention_mask for item in encodings], dtype=np.int64
            ),
            "token_type_ids": np.array(
                [item.type_ids for item in encodings], dtype=np.int64
            ),
        }
        names = {item.name for item in session.get_inputs()}
        if not names <= inputs.keys():
            raise ValueError("The encoder expects an unsupported input tensor.")
        token_vectors = session.run(
            ["last_hidden_state"], {key: inputs[key] for key in names}
        )[0]
        vectors.append(mean_pool_and_normalize(token_vectors, inputs["attention_mask"]))
        diagnostics.extend(
            {
                "token_count_with_special_tokens": sum(item.attention_mask),
                "truncated": bool(item.overflowing),
            }
            for item in encodings
        )
    return np.concatenate(vectors), diagnostics
