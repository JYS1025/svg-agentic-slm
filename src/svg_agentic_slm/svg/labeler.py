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
REFERENCE_RE = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)")


class CriticLabeler:
    """Create a labeled deep copy without changing canonical SVG."""

    def label(self, svg: str, attempt_id: str) -> SVGLabelingResult:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
        canonical = etree.fromstring(svg.encode("utf-8"), parser)
        root = copy.deepcopy(canonical)
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
                original_id=node.get("id"),
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
    """Remove Critic-only labels from a generated candidate."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
    root = etree.fromstring(svg.encode("utf-8"), parser)
    for node in root.iter():
        if isinstance(node.tag, str):
            node.attrib.pop("data-agent-id", None)
    return etree.tostring(root, encoding="unicode")
