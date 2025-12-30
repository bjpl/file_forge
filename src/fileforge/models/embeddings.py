"""Text embedding generation and semantic similarity.

Provides vector embeddings for semantic search and duplicate detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore


class EmbeddingModel:
    """Text embedding model using Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        normalize: bool = True,
        cache: bool = True,
        base_url: str = "http://localhost:11434",
    ):
        """Initialize embedding model.

        Args:
            model: Embedding model name
            normalize: Normalize embeddings to unit length
            cache: Enable embedding caching
            base_url: Ollama API base URL
        """
        self.model = model
        self.normalize = normalize
        self.base_url = base_url
        self._cache_enabled = cache
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            768-dimensional embedding vector
        """
        # Check cache
        if self._cache_enabled and text in self._cache:
            return self._cache[text]

        if ollama is None:
            raise ImportError("ollama package not installed")

        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = np.array(response["embedding"], dtype=np.float32)

            # Normalize if requested
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            # Cache result
            if self._cache_enabled:
                self._cache[text] = embedding

            return embedding

        except Exception as e:
            raise Exception(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    # Normalize vectors
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

    # Compute dot product
    similarity = np.dot(emb1_norm, emb2_norm)

    return float(similarity)


def find_similar(
    query_embedding: np.ndarray,
    stored_embeddings: List[Tuple[int, np.ndarray]],
    threshold: float = 0.8,
    top_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Find similar embeddings above threshold.

    Args:
        query_embedding: Query embedding vector
        stored_embeddings: List of (id, embedding) tuples
        threshold: Minimum similarity threshold
        top_k: Return only top K results

    Returns:
        List of (id, similarity) tuples sorted by similarity
    """
    similarities = []

    for file_id, embedding in stored_embeddings:
        similarity = cosine_similarity(query_embedding, embedding)

        if similarity >= threshold:
            similarities.append((file_id, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Limit to top K
    if top_k is not None:
        similarities = similarities[:top_k]

    return similarities


def detect_duplicates(
    embeddings: List[Tuple[int, np.ndarray]], threshold: float = 0.95
) -> List[Tuple[int, int, float]]:
    """Detect semantic duplicates in embedding set.

    Args:
        embeddings: List of (id, embedding) tuples
        threshold: Similarity threshold for duplicates

    Returns:
        List of (id1, id2, similarity) tuples for duplicates
    """
    duplicates = []

    for i, (id1, emb1) in enumerate(embeddings):
        for id2, emb2 in embeddings[i + 1 :]:
            similarity = cosine_similarity(emb1, emb2)

            if similarity >= threshold:
                duplicates.append((id1, id2, similarity))

    return duplicates
