"""Utilities for keeping tokenizer and checkpoint vocabulary sizes compatible."""

from typing import Any


def _embedding_rows(embedding: Any) -> int:
    """Return the number of rows in an embedding-like module."""
    if embedding is None or not hasattr(embedding, "weight"):
        raise ValueError("The model must expose embedding weights to align its vocabulary")
    return int(embedding.weight.shape[0])


def resize_token_embeddings_if_needed(model: Any, tokenizer: Any) -> int:
    """Resize vocabulary embeddings without truncating checkpoint-only rows.

    Causal-language-model checkpoints commonly pad their vocabulary to a hardware
    friendly size.  The tokenizer can therefore contain fewer entries than the
    model's input and output embeddings.  Resizing to ``len(tokenizer)`` in that
    case discards padded rows (which may also contain action-token weights).  Only
    grow the embeddings when the tokenizer or an existing output head needs more
    rows; never shrink a loaded checkpoint.

    Returns:
        The vocabulary row count used after any required resize.
    """
    tokenizer_rows = int(len(tokenizer))
    input_rows = _embedding_rows(model.get_input_embeddings())

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    output_rows = _embedding_rows(output_embeddings) if output_embeddings is not None else input_rows
    target_rows = max(tokenizer_rows, input_rows, output_rows)

    if target_rows != input_rows or target_rows != output_rows:
        model.resize_token_embeddings(target_rows)

    return target_rows
