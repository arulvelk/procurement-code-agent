"""Tool behaviour beyond authorization: validation, identifiers, persistence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pcagent.errors import NotFound, ValidationFailed
from pcagent.tools import (
    ClassifyParams,
    GetCodeParams,
    ListReviewsParams,
    SearchCodesParams,
    SubmitReviewParams,
    classification_id_for,
    classify_purchase,
    get_code,
    list_reviews,
    search_codes,
    submit_review,
)


async def test_search_returns_ranked_candidates(reader) -> None:
    result = await search_codes(reader, SearchCodesParams(query="office furniture"))
    assert result.candidates
    assert result.taxonomy == "psc"
    # reranked_rank is the position actually returned. retrieved_rank is where
    # fusion had it, and the two differ precisely when reranking did something,
    # which is the only reason both are on the record.
    assert [c.reranked_rank for c in result.candidates] == list(
        range(1, len(result.candidates) + 1)
    )
    assert all(c.retrieved_rank >= 1 for c in result.candidates)


async def test_get_code_round_trips(reader) -> None:
    code = await get_code(reader, GetCodeParams(code="7021"))
    assert code.title == "COMPUTERS, DIGITAL"


async def test_get_code_is_case_insensitive(reader) -> None:
    assert (await get_code(reader, GetCodeParams(code="s201"))).code == "S201"


async def test_unknown_code_is_not_found_rather_than_empty(reader) -> None:
    """An agent told "no results" may conclude the code has no information.
    Told "that code does not exist" it can correct itself."""
    with pytest.raises(NotFound):
        await get_code(reader, GetCodeParams(code="ZZZZ"))


async def test_classification_id_is_derived_from_the_input(reader) -> None:
    first = await classify_purchase(reader, ClassifyParams(description="office chairs"))
    second = await classify_purchase(
        reader, ClassifyParams(description="  OFFICE   chairs  ")
    )
    assert first.classification_id == second.classification_id
    assert first.classification_id == classification_id_for("psc", "office chairs")


async def test_different_taxonomies_get_different_ids() -> None:
    assert classification_id_for("psc", "chairs") != classification_id_for(
        "naics", "chairs"
    )


async def test_classification_is_persisted_and_readable(reader) -> None:
    created = await classify_purchase(
        reader, ClassifyParams(description="janitorial custodial services")
    )
    stored = await reader.services.store.get_classification(created.classification_id)
    assert stored.code == created.code
    assert stored.confidence == pytest.approx(created.confidence)


async def test_persist_false_leaves_no_record(reader) -> None:
    created = await classify_purchase(
        reader, ClassifyParams(description="office chairs", persist=False)
    )
    with pytest.raises(NotFound):
        await reader.services.store.get_classification(created.classification_id)


async def test_classification_carries_the_runners_up(reader) -> None:
    """A reviewer rejecting the top answer usually wants the second one."""
    created = await classify_purchase(
        reader, ClassifyParams(description="office furniture cabinets shelving")
    )
    assert created.alternatives
    assert created.code not in {c.code for c in created.alternatives}


async def test_blank_description_is_rejected(reader) -> None:
    with pytest.raises(ValidationError):
        ClassifyParams(description="  ")


async def test_review_requires_an_existing_classification(reviewer) -> None:
    with pytest.raises(NotFound):
        await submit_review(
            reviewer,
            SubmitReviewParams(
                classification_id="cls_nope",
                decision="accept",
                idempotency_key="key-00000001",
            ),
        )


async def test_reassign_requires_a_code() -> None:
    with pytest.raises(ValidationError):
        SubmitReviewParams(
            classification_id="cls_1", decision="reassign", idempotency_key="key-0001xyz"
        )


async def test_assigned_code_is_rejected_when_not_reassigning() -> None:
    with pytest.raises(ValidationError):
        SubmitReviewParams(
            classification_id="cls_1",
            decision="accept",
            assigned_code="7110",
            idempotency_key="key-0001xyz",
        )


async def test_reassign_to_a_code_outside_the_taxonomy_is_rejected(reviewer) -> None:
    created = await classify_purchase(reviewer, ClassifyParams(description="office desks"))
    with pytest.raises(ValidationFailed):
        await submit_review(
            reviewer,
            SubmitReviewParams(
                classification_id=created.classification_id,
                decision="reassign",
                assigned_code="9999",
                idempotency_key="key-00000002",
            ),
        )


async def test_a_failed_validation_does_not_burn_the_key(reviewer) -> None:
    """Validation runs before the key is reserved.

    Otherwise a caller who sent a bad body once could never retry with a
    corrected one under the same key, which is exactly what a client library
    with automatic retries would do.
    """
    created = await classify_purchase(reviewer, ClassifyParams(description="office"))
    key = "key-00000003"

    with pytest.raises(ValidationFailed):
        await submit_review(
            reviewer,
            SubmitReviewParams(
                classification_id=created.classification_id,
                decision="reassign",
                assigned_code="9999",
                idempotency_key=key,
            ),
        )

    recorded = await submit_review(
        reviewer,
        SubmitReviewParams(
            classification_id=created.classification_id,
            decision="reassign",
            assigned_code="7110",
            idempotency_key=key,
        ),
    )
    assert recorded.assigned_code == "7110"


async def test_the_reviewer_is_the_principal_not_an_argument(reviewer) -> None:
    created = await classify_purchase(reviewer, ClassifyParams(description="office"))
    recorded = await submit_review(
        reviewer,
        SubmitReviewParams(
            classification_id=created.classification_id,
            decision="accept",
            idempotency_key="key-00000004",
        ),
    )
    assert recorded.reviewer == "ana.reviewer"


async def test_reviews_can_be_filtered_by_classification(reviewer) -> None:
    first = await classify_purchase(reviewer, ClassifyParams(description="office desks"))
    second = await classify_purchase(reviewer, ClassifyParams(description="janitorial"))
    for index, classification in enumerate((first, second)):
        await submit_review(
            reviewer,
            SubmitReviewParams(
                classification_id=classification.classification_id,
                decision="accept",
                idempotency_key=f"key-0000001{index}",
            ),
        )

    filtered = await list_reviews(
        reviewer, ListReviewsParams(classification_id=first.classification_id)
    )
    assert len(filtered.reviews) == 1
    assert filtered.reviews[0].classification_id == first.classification_id
    assert len((await list_reviews(reviewer, ListReviewsParams())).reviews) == 2
