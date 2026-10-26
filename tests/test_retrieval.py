"""Retrieval: BM25 scoring, fusion, and what happens when the models are gone."""

from __future__ import annotations

from pcagent.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from pcagent.retrieval.lexical import BM25Index, normalise, tokenize


def test_plural_folding_is_deliberate_and_narrow() -> None:
    assert normalise("services") == normalise("service")
    assert normalise("devices") == normalise("device")
    # Short words and double-s words are left alone, because folding them
    # collides codes the taxonomy keeps distinct.
    assert normalise("gas") == "gas"
    assert normalise("class") == "class"


def test_tokenize_drops_stopwords_but_keeps_meaningful_short_words() -> None:
    tokens = tokenize("LEASE OF CONTAINERS FOR THE 30MM GUNS")
    assert "of" not in tokens and "for" not in tokens and "the" not in tokens
    assert "30mm" in tokens
    assert "lease" in tokens


def test_bm25_ranks_the_obvious_document_first() -> None:
    documents = [
        "OFFICE FURNITURE | FURNITURE",
        "GUNS THROUGH 30MM | WEAPONS",
        "COMPUTERS, DIGITAL | IT EQUIPMENT",
    ]
    index = BM25Index(documents)
    best = index.search("office furniture for the new building", 3)
    assert best[0][0] == 0


def test_bm25_returns_nothing_for_a_query_with_no_known_terms() -> None:
    index = BM25Index(["OFFICE FURNITURE", "GUNS THROUGH 30MM"])
    assert index.search("zzzz qqqq", 5) == []


def test_rare_terms_outweigh_common_ones() -> None:
    """The whole point of IDF. "equipment" is everywhere, "theodolite" is not."""
    documents = [
        "SURVEYING EQUIPMENT THEODOLITE",
        "OFFICE EQUIPMENT",
        "MEDICAL EQUIPMENT",
        "KITCHEN EQUIPMENT",
        "LABORATORY EQUIPMENT",
    ]
    index = BM25Index(documents)
    assert index.search("equipment theodolite", 5)[0][0] == 0


def test_fusion_rewards_agreement_between_rankers() -> None:
    """Second in both beats first in one and twentieth in the other.

    This is the property worth having: two rankers that independently like a
    document are better evidence than one ranker that loves it. Note that the
    scores are never compared, only the ranks, which is the whole reason this
    works across BM25 sums and cosine similarities.
    """
    agreed, lexical_favourite, dense_favourite = 0, 1, 2
    fillers = list(range(10, 37))

    # Ranks 1, 2, then 27 fillers, then the other ranker's favourite at 30.
    lexical = (
        [(lexical_favourite, 9.0), (agreed, 8.0)]
        + [(doc, 1.0) for doc in fillers]
        + [(dense_favourite, 0.5)]
    )
    dense = (
        [(dense_favourite, 0.9), (agreed, 0.8)]
        + [(doc, 0.1) for doc in fillers]
        + [(lexical_favourite, 0.05)]
    )

    fused = reciprocal_rank_fusion([lexical, dense], rrf_k=60)
    assert fused[0][0] == agreed
    ranking = [doc for doc, _ in fused]
    assert ranking.index(lexical_favourite) > ranking.index(agreed)
    assert ranking.index(dense_favourite) > ranking.index(agreed)


def test_fusion_of_one_ranking_preserves_its_order() -> None:
    single = [(5, 3.0), (2, 2.0), (9, 1.0)]
    fused = reciprocal_rank_fusion([single], rrf_k=60)
    assert [index for index, _ in fused] == [5, 2, 9]


def test_fusion_is_stable_for_tied_scores() -> None:
    """Deterministic output, so an eval diff means something actually changed."""
    a = [(3, 1.0), (1, 1.0)]
    b = [(1, 1.0), (3, 1.0)]
    assert reciprocal_rank_fusion([a, b], rrf_k=60) == reciprocal_rank_fusion(
        [b, a], rrf_k=60
    )


async def test_retrieval_uses_both_rankers_and_the_reranker(retriever) -> None:
    await retriever.warm()
    outcome = await retriever.retrieve("janitorial custodial", limit=3)
    assert outcome.used_dense is True
    assert outcome.used_rerank is True
    assert outcome.candidates
    assert all(c.reranked_rank is not None for c in outcome.candidates)


async def test_degrades_to_lexical_when_the_models_are_unavailable(
    lexical_only_retriever: HybridRetriever,
) -> None:
    """No network, no ONNX weights, still a working system.

    A reviewer who clones this repo on a plane gets answers rather than a
    traceback, and the result says plainly which stages did not run.
    """
    await lexical_only_retriever.warm()
    outcome = await lexical_only_retriever.retrieve("office furniture", limit=3)
    assert outcome.candidates
    assert outcome.used_dense is False
    assert outcome.used_rerank is False
    assert outcome.candidates[0].code == "7110"
    assert all(c.reranked_rank is None for c in outcome.candidates)


async def test_empty_query_returns_nothing_rather_than_everything(retriever) -> None:
    outcome = await retriever.retrieve("   ")
    assert outcome.candidates == []


async def test_reranker_only_sees_the_configured_depth(retriever) -> None:
    await retriever.warm()
    await retriever.retrieve("office", limit=5)
    _query, count = retriever._reranker.calls[-1]  # type: ignore[attr-defined]
    assert count <= retriever.settings.rerank_depth
