"""
EnterpriseAI Handoff Kit — Embedder
Two real backends, selected by EMBEDDER_BACKEND env var:

  tfidf_svd             TF-IDF vectorization + Truncated SVD (LSA).
                        Real vectorization, deterministic, no model download.
                        Must be fit on the corpus before use.

  sentence_transformers all-MiniLM-L6-v2 dense embeddings.
                        Higher semantic quality, requires torch (~2GB).

Both implement the same interface, so swapping is a one-line env change.
The vector store and all downstream monitoring are backend-agnostic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import hashlib

import numpy as np

from src.config import settings


def content_hash(text: str) -> str:
    """Stable SHA256 of document content — detects source changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BaseEmbedder(ABC):
    """Interface every embedding backend must implement."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def fit(self, corpus: list[str]) -> None: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()

    def coverage(self, text: str) -> dict:
        """
        Vocabulary coverage signal. Backends without a fitted vocabulary
        (transformers) report full coverage, since they handle unseen tokens
        natively through subword embeddings.
        """
        return {
            "tfidf_mass":     1.0,
            "matched_terms":  [],
            "matched_count":  -1,
            "token_count":    -1,
            "token_coverage": 1.0,
        }


class TfidfSvdEmbedder(BaseEmbedder):
    """
    TF-IDF + Truncated SVD (Latent Semantic Analysis).

    This is a real, widely-used vectorization method in information retrieval.
    Pipeline: raw text -> TF-IDF sparse matrix -> SVD -> dense L2-normalized vector.
    Cosine similarity on these vectors is a genuine semantic-ish similarity measure.

    Advantages here: deterministic, no network download, fast, fully inspectable.
    Limitation vs transformers: no contextual understanding of synonyms
    beyond co-occurrence statistics in the fitted corpus.
    """

    name = "tfidf_svd"

    def __init__(self, dim: int | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        self.dim = dim or settings.EMBEDDING_DIM
        self._TfidfVectorizer = TfidfVectorizer
        self._TruncatedSVD = TruncatedSVD
        self._Pipeline = Pipeline
        self._Normalizer = Normalizer
        self._pipeline = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Cannot fit embedder on empty corpus")

        # SVD components cannot exceed min(n_docs, n_features) - 1
        n_docs = len(corpus)
        target_dim = min(self.dim, max(2, n_docs - 1))

        self._pipeline = self._Pipeline([
            ("tfidf", self._TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )),
            ("svd", self._TruncatedSVD(n_components=target_dim, random_state=42)),
            ("norm", self._Normalizer(copy=False)),
        ])
        self._pipeline.fit(corpus)
        self.dim = target_dim
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before encode()")
        return np.asarray(self._pipeline.transform(texts), dtype=np.float32)

    def coverage(self, text: str) -> dict:
        """
        How much of this text is actually represented in the fitted vocabulary.

        L2 normalization makes a query matching one incidental term look as
        confident as a query matching ten specific ones. Coverage restores that
        signal by measuring, before normalization, how much IDF-weighted mass
        the query carries and how many of its tokens are in-vocabulary.

        Returns:
            tfidf_mass      L2 norm of the raw TF-IDF vector (unnormalized)
            matched_terms   in-vocabulary terms found
            token_coverage  matched tokens / total content tokens
        """
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before coverage()")

        tfidf = self._pipeline.named_steps["tfidf"]
        X = tfidf.transform([text])
        mass = float(np.sqrt(X.multiply(X).sum()))

        vocab = tfidf.vocabulary_
        names = tfidf.get_feature_names_out()
        matched = [str(names[i]) for i in X.nonzero()[1]]

        analyzer = tfidf.build_analyzer()
        tokens = analyzer(text)
        unigrams = [t for t in tokens if " " not in t]
        in_vocab = [t for t in unigrams if t in vocab]
        token_coverage = (len(set(in_vocab)) / len(set(unigrams))) if unigrams else 0.0

        return {
            "tfidf_mass":      round(mass, 6),
            "matched_terms":   matched,
            "matched_count":   len(matched),
            "token_count":     len(set(unigrams)),
            "token_coverage":  round(token_coverage, 4),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self._pipeline, "dim": self.dim}, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self._pipeline = blob["pipeline"]
        self.dim = blob["dim"]
        self._fitted = True
        return True


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Dense transformer embeddings via sentence-transformers.
    Requires torch. Set EMBEDDER_BACKEND=sentence_transformers to use.
    No fit() needed — the model is pretrained.
    """

    name = "sentence_transformers"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name or settings.ST_MODEL_NAME
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, corpus: list[str]) -> None:
        return  # pretrained

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )


def build_embedder(backend: str | None = None) -> BaseEmbedder:
    """Factory. Falls back to tfidf_svd if transformers are unavailable."""
    backend = backend or settings.EMBEDDER_BACKEND
    if backend == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # torch missing, download blocked, etc.
            print(f"[embedder] sentence_transformers unavailable ({exc}); "
                  f"falling back to tfidf_svd")
            return TfidfSvdEmbedder()
    return TfidfSvdEmbedder()
