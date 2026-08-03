"""Strict structural and security validation for generated SVG."""

from __future__ import annotations

import logging
import re

from lxml import etree

from svg_agentic_slm.svg.base import BaseValidator
from svg_agentic_slm.svg.schemas import SVGDiagnostic, SVGValidationResult

logger = logging.getLogger(__name__)

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
SAFE_ELEMENTS = {
    "circle",
    "clipPath",
    "defs",
    "desc",
    "ellipse",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feDistantLight",
    "feDropShadow",
    "feFlood",
    "feFuncA",
    "feFuncB",
    "feFuncG",
    "feFuncR",
    "feGaussianBlur",
    "feImage",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "fePointLight",
    "feSpecularLighting",
    "feSpotLight",
    "feTile",
    "feTurbulence",
    "filter",
    "g",
    "image",
    "line",
    "linearGradient",
    "marker",
    "mask",
    "metadata",
    "path",
    "pattern",
    "polygon",
    "polyline",
    "radialGradient",
    "rect",
    "stop",
    "svg",
    "switch",
    "symbol",
    "text",
    "textPath",
    "title",
    "tspan",
    "use",
}
URL_PATTERN = re.compile(
    r"url\(\s*(['\"]?)(.*?)\1\s*\)",
    re.IGNORECASE | re.DOTALL,
)
URI_SCHEME_PATTERN = re.compile(
    r"(?:data|file|ftp|https?|javascript):",
    re.IGNORECASE,
)
URL_IGNORED_ASCII_WHITESPACE = str.maketrans("", "", "\t\n\r")
CSS_FUNCTION_PATTERN = re.compile(r"([A-Za-z-]+)\s*\(")
SAFE_STYLE_PROPERTIES = {
    "color",
    "display",
    "dominant-baseline",
    "fill",
    "fill-opacity",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "opacity",
    "paint-order",
    "shape-rendering",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "text-anchor",
    "text-rendering",
    "transform",
    "vector-effect",
    "visibility",
}
SAFE_STYLE_FUNCTIONS = {
    "calc",
    "hsl",
    "hsla",
    "matrix",
    "rgb",
    "rgba",
    "rotate",
    "scale",
    "skewx",
    "skewy",
    "translate",
    "url",
}


class SVGValidator(BaseValidator):
    """Validate XML structure and reject active or externally loaded content."""

    def validate(self, svg_content: str) -> SVGValidationResult:
        """Validate an SVG without resolving DTDs, entities, or networks."""
        result = SVGValidationResult()
        errors: list[str] = []
        warnings: list[str] = []

        if not svg_content or not svg_content.strip():
            result.errors = ["SVG content is empty."]
            result.diagnostics = [SVGDiagnostic("empty_svg", result.errors[0])]
            return result

        result.has_svg_tag = bool(re.search(r"<(?:[\w.-]+:)?svg\b", svg_content))
        result.has_closing_tag = bool(
            re.search(r"</(?:[\w.-]+:)?svg\s*>", svg_content)
        )
        if not result.has_svg_tag:
            errors.append("Missing <svg> opening tag.")
        if "<!DOCTYPE" in svg_content.upper():
            errors.append("DOCTYPE declarations are not allowed.")

        parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            load_dtd=False,
            huge_tree=False,
            recover=False,
        )
        try:
            root = etree.fromstring(svg_content.encode("utf-8"), parser=parser)
            result.is_well_formed_xml = True
        except (etree.XMLSyntaxError, ValueError) as exc:
            errors.append(f"XML parsing error: {exc}")
            result.errors = errors
            result.warnings = warnings
            result.diagnostics = _build_diagnostics(errors, warnings)
            return result

        root_name = etree.QName(root).localname
        root_namespace = etree.QName(root).namespace
        if root_name != "svg":
            errors.append("Root element must be <svg>.")
        else:
            result.has_svg_tag = True
        if root_namespace not in (None, SVG_NAMESPACE):
            errors.append(f"Unsupported SVG namespace: {root_namespace}.")
        if root_namespace is None:
            warnings.append("Missing xmlns attribute in <svg> element.")
        if root.getroottree().xpath("//processing-instruction()"):
            errors.append("XML processing instructions are not allowed.")

        for element in root.iter():
            if not isinstance(element.tag, str):
                continue
            element_qname = etree.QName(element)
            element_name = element_qname.localname
            if element_qname.namespace not in (None, SVG_NAMESPACE):
                errors.append(
                    f"Foreign element namespace is not allowed: {element_qname.namespace}."
                )
            if element_name not in SAFE_ELEMENTS:
                errors.append(f"Element is not allowed in static SVG: <{element_name}>.")

            for raw_name, raw_value in element.attrib.items():
                attribute_qname = etree.QName(raw_name)
                attribute_name = attribute_qname.localname.lower()
                value = raw_value.strip()
                _check_attribute_namespace(
                    attribute_qname.namespace,
                    attribute_name,
                    errors,
                )
                _check_obfuscated_or_absolute_reference(value, errors)
                if attribute_name.startswith("on"):
                    errors.append(
                        f"Event handler attribute is not allowed: {attribute_name}."
                    )
                if attribute_name == "base":
                    errors.append("Base URI attributes are not allowed.")
                if attribute_name in {"href", "src"} and value and not value.startswith("#"):
                    errors.append(
                        f"External reference is not allowed in {attribute_name}: {value}."
                    )
                if attribute_name == "style":
                    _check_style_attribute(value, errors)
                else:
                    _check_url_references(value, errors)

        result.errors = errors
        result.warnings = warnings
        result.is_valid = not errors
        result.diagnostics = _build_diagnostics(errors, warnings)
        return result


def _build_diagnostics(errors: list[str], warnings: list[str]) -> list[SVGDiagnostic]:
    def code(message: str) -> str:
        lowered = message.lower()
        if "xml parsing" in lowered:
            return "xml_parse_error"
        if "missing <svg>" in lowered:
            return "missing_svg_root"
        if "root element" in lowered:
            return "invalid_root"
        if "namespace" in lowered:
            return "invalid_namespace"
        if "doctype" in lowered:
            return "unsafe_doctype"
        if "element is not allowed" in lowered:
            return "unsafe_element"
        if "reference" in lowered or "uri" in lowered or "url" in lowered:
            return "external_reference"
        if "css" in lowered or "style" in lowered:
            return "unsafe_css"
        return "unsafe_attribute"

    return [
        SVGDiagnostic(code(message), message, "error") for message in errors
    ] + [
        SVGDiagnostic(code(message), message, "warning") for message in warnings
    ]


def _check_url_references(value: str, errors: list[str]) -> None:
    for match in URL_PATTERN.finditer(value):
        target = match.group(2).strip()
        if target and not target.startswith("#"):
            errors.append(f"External URL reference is not allowed: {target}.")


def _check_attribute_namespace(
    namespace: str | None,
    attribute_name: str,
    errors: list[str],
) -> None:
    if namespace is None:
        return
    if namespace == XML_NAMESPACE and attribute_name in {"lang", "space"}:
        return
    if namespace == XLINK_NAMESPACE and attribute_name == "href":
        return
    errors.append(
        "Foreign attribute namespace is not allowed: "
        f"{namespace} ({attribute_name})."
    )


def _check_obfuscated_or_absolute_reference(value: str, errors: list[str]) -> None:
    if "\\" in value or "/*" in value or "*/" in value:
        errors.append("Escaped or commented attribute values are not allowed.")
    normalized_url_value = value.translate(URL_IGNORED_ASCII_WHITESPACE)
    if URI_SCHEME_PATTERN.search(normalized_url_value):
        errors.append("Absolute URI schemes are not allowed in SVG attributes.")


def _check_style_attribute(value: str, errors: list[str]) -> None:
    if "@" in value:
        errors.append("CSS at-rules are not allowed.")
    for declaration in value.split(";"):
        declaration = declaration.strip()
        if not declaration:
            continue
        if ":" not in declaration:
            errors.append("Malformed inline CSS declaration.")
            continue
        property_name, property_value = declaration.split(":", 1)
        normalized_property = property_name.strip().lower()
        if normalized_property not in SAFE_STYLE_PROPERTIES:
            errors.append(f"Unsafe inline CSS property: {normalized_property}.")
        for function_name in CSS_FUNCTION_PATTERN.findall(property_value):
            if function_name.lower() not in SAFE_STYLE_FUNCTIONS:
                errors.append(f"Unsafe inline CSS function: {function_name}.")
        _check_url_references(property_value, errors)
