"""Attempt-local SVG node labeling for grounded critic feedback."""

from __future__ import annotations

import copy
import re
from collections import defaultdict, deque

class _LazyEtree:
    """Load lxml only when an XML operation is requested."""

    _module = None

    def __getattr__(self, name: str):
        module = self._module
        if module is None:
            from lxml import etree as module

            self._module = module
        return getattr(module, name)


etree = _LazyEtree()


from svg_agentic_slm.svg.schemas import SVGElementRef, SVGLabelingResult

RESOURCE_TAGS = {"symbol", "linearGradient", "radialGradient", "pattern", "clipPath", "mask", "filter", "marker"}
GRAPHICS_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "image", "use"}
NONVISUAL_TEXT_TAGS = {"desc", "metadata", "title"}
REFERENCE_RE = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)")
URL_FRAGMENT_RE = re.compile(
    r"(url\(\s*['\"]?#)([^)'\"\s]+)(['\"]?\s*\))",
    re.IGNORECASE,
)


class CriticLabeler:
    """Create a labeled deep copy without changing canonical SVG."""

    def label(self, svg: str, attempt_id: str) -> SVGLabelingResult:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
        canonical = etree.fromstring(svg.encode("utf-8"), parser)
        root = copy.deepcopy(canonical)
        _remove_nonvisual_text(root, remove_descriptive_elements=True)
        original_ids = _neutralize_critic_attributes(root)
        for node in root.iter():
            if isinstance(node.tag, str):
                node.attrib.pop("data-agent-id", None)

        by_id = {node.get("id"): node for node in root.iter() if isinstance(node.tag, str) and node.get("id")}
        reachable_resources = self._reachable_resources(root, by_id)
        counters: dict[str, int] = defaultdict(int)
        assigned: dict[etree._Element, str] = {}
        elements: dict[str, SVGElementRef] = {}
        tree = root.getroottree()

        for node in root.iter():
            if not isinstance(node.tag, str):
                continue
            tag = etree.QName(node).localname
            role: str | None = None
            prefix = ""
            if tag == "svg": role, prefix = "svg", "s"
            elif tag == "g" and any(isinstance(child.tag, str) for child in node): role, prefix = "group", "g"
            elif tag in GRAPHICS_TAGS: role, prefix = "graphics", "e"
            elif tag in RESOURCE_TAGS and node in reachable_resources: role, prefix = "resource", "d"
            if role is None:
                continue
            counters[prefix] += 1
            agent_id = f"{prefix}{counters[prefix]:04d}"
            node.set("data-agent-id", agent_id)
            assigned[node] = agent_id
            parent = node.getparent()
            while parent is not None and parent not in assigned:
                parent = parent.getparent()
            elements[agent_id] = SVGElementRef(
                agent_id=agent_id,
                xpath=tree.getpath(node),
                tag=tag,
                original_id=original_ids.get(node),
                parent_agent_id=assigned.get(parent) if parent is not None else None,
                role=role,  # type: ignore[arg-type]
            )
        return SVGLabelingResult(attempt_id, etree.tostring(root, encoding="unicode"), elements)

    def _reachable_resources(self, root: etree._Element, by_id: dict[str, etree._Element]) -> set[etree._Element]:
        found: set[etree._Element] = set()
        queue: deque[str] = deque()
        for node in root.iter():
            if not isinstance(node.tag, str) or etree.QName(node).localname in RESOURCE_TAGS:
                continue
            queue.extend(_references(node))
        while queue:
            target = by_id.get(queue.popleft())
            if target is None or target in found:
                continue
            found.add(target)
            queue.extend(_references(target))
            for child in target.iterdescendants():
                queue.extend(_references(child))
        return found


def _references(node: etree._Element) -> list[str]:
    result: list[str] = []
    for name, value in node.attrib.items():
        local = etree.QName(name).localname
        if local == "href" and value.startswith("#"):
            result.append(value[1:])
        result.extend(REFERENCE_RE.findall(value))
    return result


def strip_reserved_labels(svg: str) -> str:
    """Remove Critic-only labels and non-rendered XML annotations."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
    root = etree.fromstring(svg.encode("utf-8"), parser)
    _remove_nonvisual_text(root, remove_descriptive_elements=False)
    for node in root.iter():
        if isinstance(node.tag, str):
            node.attrib.pop("data-agent-id", None)
    return etree.tostring(root, encoding="unicode")


def _remove_nonvisual_text(
    root: etree._Element,
    *,
    remove_descriptive_elements: bool,
) -> None:
    """Remove invisible prose while preserving surrounding XML whitespace.

    Comments and processing instructions are never part of the rendered SVG, so
    they are removed from every generated candidate.  The Critic-facing copy
    additionally omits title, description, and metadata elements: these remain
    available in the canonical SVG for accessibility/provenance, but must not
    act as textual evidence for a visual judgment.
    """
    nodes = list(root.xpath("//comment() | //processing-instruction()"))
    if remove_descriptive_elements:
        nodes.extend(
            node
            for node in root.iter()
            if isinstance(node.tag, str)
            and etree.QName(node).localname in NONVISUAL_TEXT_TAGS
        )
    for node in nodes:
        _remove_node_preserving_tail(node)


def _remove_node_preserving_tail(node: etree._Element) -> None:
    parent = node.getparent()
    if parent is None:
        return
    tail = node.tail
    previous = node.getprevious()
    parent.remove(node)
    if not tail:
        return
    if previous is None:
        parent.text = (parent.text or "") + tail
    else:
        previous.tail = (previous.tail or "") + tail


def _neutralize_critic_attributes(
    root: etree._Element,
) -> dict[etree._Element, str]:
    """Remove hidden semantic labels while retaining valid local references."""
    id_mapping: dict[str, str] = {}
    original_ids: dict[etree._Element, str] = {}
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        original_id = node.get("id")
        if original_id:
            neutral_id = f"ref{len(id_mapping) + 1:04d}"
            id_mapping[original_id] = neutral_id
            original_ids[node] = original_id
            node.set("id", neutral_id)

    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        for raw_name, value in list(node.attrib.items()):
            local_name = etree.QName(raw_name).localname.lower()
            if (
                local_name == "class"
                or local_name == "role"
                or local_name.startswith("aria-")
                or local_name.startswith("data-")
            ):
                del node.attrib[raw_name]
                continue
            if local_name == "href" and value.startswith("#"):
                target = id_mapping.get(value[1:])
                if target is not None:
                    node.set(raw_name, f"#{target}")
                    continue
            node.set(raw_name, _rewrite_url_fragments(value, id_mapping))
    return original_ids


def _rewrite_url_fragments(value: str, id_mapping: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = id_mapping.get(match.group(2), match.group(2))
        return f"{match.group(1)}{target}{match.group(3)}"

    return URL_FRAGMENT_RE.sub(replace, value)
