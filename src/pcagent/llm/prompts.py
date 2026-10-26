"""The prompt, kept in one file so it can be read and argued with.

Three things in here are load bearing.

The model picks from a list, it does not generate a code. Federal code sets are
large, old and full of near misses, and a model asked to produce "the PSC for
this" will happily return a well formed code that does not exist. Constraining
the answer to the shortlist turns a hallucination into a wrong but checkable
answer. The tool layer then verifies the choice against the list anyway, because
a prompt is a request and not a guarantee.

Evidence is quoted, not summarised. Asking for verbatim spans from the input
gives a reviewer something to check in two seconds, and ``ground_evidence``
drops any span that is not actually in the text.

Low confidence is an allowed answer. The point of the system is to route
uncertain cases to a person, so a prompt that pushes the model towards
confidence would defeat it.
"""

from __future__ import annotations

from ..domain import Candidate, Taxonomy

TAXONOMY_BRIEF: dict[Taxonomy, str] = {
    "psc": (
        "PSC (Product Service Code) describes WHAT was bought. A four character "
        "code. Codes beginning with a digit are products, those beginning with a "
        "letter are services, and those beginning with A are research and "
        "development."
    ),
    "naics": (
        "NAICS describes WHAT THE SELLER DOES, the industry of the vendor, as a "
        "six digit code. It is about the supplier's line of business, not about "
        "the item purchased, and the two can legitimately differ."
    ),
}

SYSTEM_PROMPT = """\
You assign United States federal procurement codes to purchase descriptions.

{taxonomy_brief}

Rules:
1. Choose exactly one code from the numbered candidate list. Never return a code
   that is not on the list, even if you believe a better one exists. If nothing
   fits well, choose the closest candidate and set confidence below 0.3.
2. Quote evidence verbatim from the purchase description. Copy the exact words,
   do not paraphrase or correct them. Two or three short spans is plenty. If the
   description gives you nothing to quote, return an empty list.
3. Keep the rationale to one or two sentences, and make it about why this code
   beats the other candidates, not a restatement of the description.
4. Set confidence honestly. Federal descriptions are abbreviated, full of jargon
   and sometimes genuinely ambiguous. A low confidence answer sends the case to a
   human reviewer, which is a correct and useful outcome. Treat 0.9 and above as
   a claim that another expert would agree without hesitation.
"""


def build_system_prompt(taxonomy: Taxonomy) -> str:
    return SYSTEM_PROMPT.format(taxonomy_brief=TAXONOMY_BRIEF[taxonomy])


def build_user_prompt(description: str, candidates: tuple[Candidate, ...]) -> str:
    lines = [
        "Purchase description:",
        description.strip(),
        "",
        "Candidate codes:",
    ]
    for position, candidate in enumerate(candidates, start=1):
        lines.append(f"{position}. {candidate.code} - {candidate.title}")
    lines.append("")
    lines.append("Return the single best code from the list above.")
    return "\n".join(lines)
