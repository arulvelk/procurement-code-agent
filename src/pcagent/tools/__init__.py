"""The typed tool core.

Importing this package registers every tool. Both transports import from here,
so there is exactly one definition of what a tool is and what it enforces.
"""

from .base import REGISTRY, Services, ToolContext, ToolSpec, tool
from .catalog import (
    GetCodeParams,
    SearchCodesParams,
    SearchCodesResult,
    get_code,
    search_codes,
)
from .classify import ClassifyParams, classification_id_for, classify_purchase
from .review import (
    ListReviewsParams,
    ListReviewsResult,
    SubmitReviewParams,
    list_reviews,
    submit_review,
)

__all__ = [
    "REGISTRY",
    "ClassifyParams",
    "GetCodeParams",
    "ListReviewsParams",
    "ListReviewsResult",
    "SearchCodesParams",
    "SearchCodesResult",
    "Services",
    "SubmitReviewParams",
    "ToolContext",
    "ToolSpec",
    "classification_id_for",
    "classify_purchase",
    "get_code",
    "list_reviews",
    "search_codes",
    "submit_review",
    "tool",
]
