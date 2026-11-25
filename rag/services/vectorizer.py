"""
Local CPU Dense Vector Embedding Service using FastEmbed.

Model: BAAI/bge-small-en-v1.5 (Qdrant ONNX quantized)
Dimensions: Exactly 384 dimensions
Hardware: Pure CPU inference via onnxruntime
Cost: ₦0 (100% local, no external APIs, no token costs)
"""
from typing import List, Optional
import numpy as np

# Global cached model instance (lazy loaded)
_EMBEDDING_MODEL = None
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIMENSIONS = 384


def get_embedding_model():
    """
    Returns the singleton TextEmbedding model, lazily loading it on first call.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from fastembed import TextEmbedding
        _EMBEDDING_MODEL = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generates 384-dimensional dense vector embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for inference (default: 32).

    Returns:
        List of 384-float embedding vectors.
    """
    if not texts:
        return []

    model = get_embedding_model()
    # FastEmbed returns an iterable generator of numpy float32 ndarrays
    raw_embeddings = model.embed(texts, batch_size=batch_size)

    results: List[List[float]] = []
    for vec in raw_embeddings:
        vec_list = [float(x) for x in vec]
        if len(vec_list) != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Embedding vector dimension mismatch: expected {EXPECTED_DIMENSIONS}, got {len(vec_list)}"
            )
        results.append(vec_list)

    return results


def embed_query(query: str) -> List[float]:
    """
    Generates a 384-dimensional dense vector embedding for a single search query.

    Args:
        query: Query string.

    Returns:
        List of 384 float values.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    vectors = embed_texts([query.strip()], batch_size=1)
    return vectors[0]
