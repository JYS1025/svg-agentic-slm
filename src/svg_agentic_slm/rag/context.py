"""Token-budgeted assembly of structurally complete RAG context items."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable, Literal

from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.svg.validator import SVG_NAMESPACE, SVGValidator

ContextUsageStatus = Literal["fully_used", "partially_used", "dropped"]


@dataclass(frozen=True)
class ContextItemUsage:
    """Observed use of one retrieved item in the final Generator context."""

    item_id: str
    status: ContextUsageStatus
    token_count: int
    included_element_count: int
    total_element_count: int


@dataclass(frozen=True)
class ContextSelection:
    """Selected context and exact tokenizer-based accounting."""

    items: list[RetrievedExample]
    usage: list[ContextItemUsage]
    token_count: int


def select_context_by_tokens(
    items: list[RetrievedExample],
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    format_context: Callable[[list[RetrievedExample]], str],
) -> ContextSelection:
    """Select whole items or whole top-level SVG elements within a token budget."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative.")
    selected: list[RetrievedExample] = []
    usage: list[ContextItemUsage] = []
    current_tokens = 0

    for item in items:
        full_tokens = _context_token_count(
            selected + [item], count_tokens, format_context
        )
        element_variants = (
            _svg_element_variants(item) if item.kind == "reference_svg" else []
        )
        total_elements = len(element_variants)
        if full_tokens <= max_tokens:
            usage.append(
                ContextItemUsage(
                    item_id=item.item_id,
                    status="fully_used",
                    token_count=max(0, full_tokens - current_tokens),
                    included_element_count=total_elements,
                    total_element_count=total_elements,
                )
            )
            selected.append(item)
            current_tokens = full_tokens
            continue

        partial_item: RetrievedExample | None = None
        partial_tokens = current_tokens
        included_elements = 0
        for element_count, content in enumerate(element_variants, start=1):
            candidate = replace(item, content=content)
            candidate_tokens = _context_token_count(
                selected + [candidate],
                count_tokens,
                format_context,
            )
            if candidate_tokens > max_tokens:
                break
            partial_item = candidate
            partial_tokens = candidate_tokens
            included_elements = element_count

        if partial_item is not None and included_elements < total_elements:
            selected.append(partial_item)
            usage.append(
                ContextItemUsage(
                    item_id=item.item_id,
                    status="partially_used",
                    token_count=max(0, partial_tokens - current_tokens),
                    included_element_count=included_elements,
                    total_element_count=total_elements,
                )
            )
            current_tokens = partial_tokens
        else:
            usage.append(
                ContextItemUsage(
                    item_id=item.item_id,
                    status="dropped",
                    token_count=0,
                    included_element_count=0,
                    total_element_count=total_elements,
                )
            )

    return ContextSelection(items=selected, usage=usage, token_count=current_tokens)


def _context_token_count(
    items: list[RetrievedExample],
    count_tokens: Callable[[str], int],
    format_context: Callable[[list[RetrievedExample]], str],
) -> int:
    if not items:
        return 0
    count = count_tokens(format_context(items))
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise RuntimeError(
            "Model tokenizer must return a non-negative integer token count."
        )
    return count


def _svg_element_variants(item: RetrievedExample) -> list[str]:
    """Return valid SVG documents containing whole top-level elements only."""
    from lxml import etree  # type: ignore[import-untyped]

    content = item.content.strip()
    wrapped = not content.lower().startswith("<svg")
    candidate = f'<svg xmlns="{SVG_NAMESPACE}">{content}</svg>' if wrapped else content
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )
    try:
        root = etree.fromstring(candidate.encode("utf-8"), parser=parser)
    except (etree.XMLSyntaxError, ValueError):
        return []
    children = list(root)
    if len(children) < 2:
        return []

    base = deepcopy(root)
    for child in list(base):
        base.remove(child)
    variants: list[str] = []
    for child in children:
        base.append(deepcopy(child))
        serialized = etree.tostring(base, encoding="unicode")
        if SVGValidator().validate(serialized).is_valid:
            variants.append(serialized)
    return variants
