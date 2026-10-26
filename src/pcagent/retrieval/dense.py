"""Dense retrieval over the taxonomy, on the CPU, with no API key.

Embeddings come from a small ONNX model through fastembed, so a reviewer can
clone this repo and reproduce every retrieval number in the eval table without
an account anywhere.

bge-base-en-v1.5 is the default because it measured best, not because it is the
biggest thing that fits. Against bge-small it is worth about a point of accuracy
at 1 and 0.013 MRR on the dev split, for 210MB instead of 67MB. Both numbers are
in evals/report.md, and bge-small remains a one line change for anyone who wants
the smaller download.

Two things to know about this module.

First, embedding is CPU bound and synchronous. Nothing here is async, and that is
not an oversight. Wrapping blocking work in ``async def`` does not make it yield.
The async layer calls into this module through ``asyncio.to_thread`` instead, so
the event loop stays free.

Second, the corpus vectors are cached on disk keyed by a hash of the model name
and the corpus itself. Change either and the cache misses and rebuilds. That
keeps the evals reproducible without re-embedding on every run.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

# fastembed and numpy hand back loosely typed arrays, so the float32 matrix
# type is named once here and used everywhere rather than spelled out.
Vectors = npt.NDArray[np.float32]

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"


class DenseUnavailable(RuntimeError):
    """The embedding model could not be loaded, usually because it has not been
    downloaded yet and there is no network. Callers fall back to lexical only."""


def _corpus_key(model_name: str, documents: list[str]) -> str:
    digest = hashlib.sha256(model_name.encode())
    for document in documents:
        digest.update(b"\x00")
        digest.update(document.encode())
    return digest.hexdigest()[:16]


class DenseIndex:
    """Normalised corpus embeddings plus a cosine search over them."""

    def __init__(
        self,
        documents: list[str],
        *,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._documents = documents
        self._cache_dir = cache_dir or CACHE_DIR
        # Left unset by default so ONNX uses its own heuristic. Pinned in the
        # eval runner so timings are comparable between runs.
        self._threads = threads or int(os.environ.get("PCAGENT_ONNX_THREADS", "0")) or None
        self._model: object | None = None
        self._matrix: Vectors | None = None

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - import guard
                raise DenseUnavailable("fastembed is not installed") from exc
            try:
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    threads=self._threads,
                )
            except Exception as exc:
                raise DenseUnavailable(f"could not load {self.model_name}: {exc}") from exc
        return self._model

    @property
    def matrix(self) -> Vectors:
        """The corpus embedding matrix, built or loaded from cache on first use."""
        if self._matrix is not None:
            return self._matrix

        key = _corpus_key(self.model_name, self._documents)
        cache_path = self._cache_dir / f"dense-{key}.npy"
        if cache_path.exists():
            cached: Vectors = np.load(cache_path)
            self._matrix = cached
            return cached

        model = self._load_model()
        vectors = np.asarray(list(model.embed(self._documents)), dtype=np.float32)  # type: ignore[attr-defined]
        self._matrix = _l2_normalise(vectors)

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a crash midway cannot leave a half written cache
        # that a later run would happily load.
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        # Written through a handle because np.save appends ".npy" to a path
        # argument that does not already end in it, which would defeat the rename.
        with temporary.open("wb") as handle:
            np.save(handle, self._matrix)
        temporary.replace(cache_path)
        return self._matrix

    def embed_query(self, query: str) -> Vectors:
        model = self._load_model()
        # query_embed applies whatever instruction prefix the model was trained
        # with. bge wants one, and skipping it quietly costs several points of
        # recall, which is the kind of bug that never raises.
        vector = np.asarray(next(iter(model.query_embed([query]))), dtype=np.float32)  # type: ignore[attr-defined]
        normalised: Vectors = _l2_normalise(vector.reshape(1, -1))[0]
        return normalised

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return ``(document index, cosine similarity)`` for the best matches."""
        matrix = self.matrix
        similarities = matrix @ self.embed_query(query)
        limit = min(limit, similarities.shape[0])
        # argpartition first so a 2000 row corpus is not fully sorted per query.
        top = np.argpartition(-similarities, limit - 1)[:limit]
        top = top[np.argsort(-similarities[top], kind="stable")]
        return [(int(index), float(similarities[index])) for index in top]

    def warm(self) -> None:
        """Build the cache and load the model up front.

        Called at service start so the first real request does not pay for a
        model load that can take a couple of seconds.
        """
        _ = self.matrix
        self._load_model()


def _l2_normalise(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against a zero vector, which a degenerate input can produce.
    np.maximum(norms, 1e-12, out=norms)
    unit: Vectors = (vectors / norms).astype(np.float32)
    return unit
