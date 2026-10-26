"""Cross encoder reranking of the fused candidate list.

Retrieval and reranking answer different questions. BM25 and the dense index
both score a document without ever looking at the query and the document
together, which is what makes them cheap enough to run over the whole corpus. A
cross encoder reads the pair jointly, so it can tell that "monitor" next to
"desktop" means computer equipment rather than patient monitoring, at a cost
that only makes sense over a few dozen candidates.

That is the entire trade. Recall is bought cheaply and in bulk, precision is
bought expensively and only where it can still change the answer.

Which cross encoder matters more than the fact of having one. Three were
measured on the dev split, and two of them did not earn their place:

    ms-marco-MiniLM-L-6     MRR 0.119 -> 0.120   (no effect)
    ms-marco-MiniLM-L-12    MRR 0.119 -> 0.126
    jina-reranker-v1-turbo  MRR 0.119 -> 0.140   (recall@5 up 28% relative)

The ms-marco models are trained on web passages and barely separate short
taxonomy titles: on a query with one obviously correct answer they spread four
candidates across a single logit. That is why the default is jina.

Rerank depth was measured too, and deeper is worse. Going from 30 candidates to
80 dropped MRR from 0.140 to 0.126, because a reranker that is only moderately
better than fusion promotes noise when you hand it more of it.

Like the dense index, this is synchronous CPU work. The async layer calls it
through a thread.
"""

from __future__ import annotations

DEFAULT_MODEL = "jinaai/jina-reranker-v1-turbo-en"


class RerankUnavailable(RuntimeError):
    """The cross encoder could not be loaded. Callers fall back to fused order."""


class CrossEncoderReranker:
    """Scores (query, document) pairs with a small ONNX cross encoder."""

    def __init__(self, *, model_name: str = DEFAULT_MODEL, threads: int | None = None) -> None:
        self.model_name = model_name
        self._threads = threads
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:  # pragma: no cover - import guard
                raise RerankUnavailable("fastembed is not installed") from exc
            try:
                self._model = TextCrossEncoder(model_name=self.model_name, threads=self._threads)
            except Exception as exc:
                raise RerankUnavailable(f"could not load {self.model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, in the order given.

        The scores are raw logits. They are comparable within a single call and
        meaningless across calls, which is why nothing downstream treats them as
        a probability or compares them to a fixed threshold.
        """
        if not documents:
            return []
        model = self._load()
        return [float(score) for score in model.rerank(query, documents)]  # type: ignore[attr-defined]

    def warm(self) -> None:
        self._load()
