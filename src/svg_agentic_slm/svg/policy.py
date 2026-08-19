"""Single static-SVG policy shared by generation and validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticSVGPolicy:
    safe_elements: frozenset[str]
    safe_style_properties: frozenset[str]
    safe_style_functions: frozenset[str]
    forbidden_elements: frozenset[str]
    external_reference_attributes: frozenset[str]
    forbidden_uri_schemes: tuple[str, ...]

    def generator_rule(self) -> str:
        """Return the authoritative policy wording for Generator prompts."""
        active = ", ".join(sorted(self.forbidden_elements))
        schemes = ", ".join(self.forbidden_uri_schemes)
        return (
            "Generate static SVG only. Do not use active elements "
            f"({active}), any event-handler attribute beginning with on, "
            "foreign content, data URLs, or external references. href/src and "
            "CSS url() references may use same-document #fragment targets only; "
            f"absolute URI schemes ({schemes}) are forbidden."
        )


STATIC_SVG_POLICY = StaticSVGPolicy(
    safe_elements=frozenset(
        {
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
    ),
    safe_style_properties=frozenset(
        {
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
    ),
    safe_style_functions=frozenset(
        {
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
    ),
    forbidden_elements=frozenset(
        {"animate", "animatemotion", "animatetransform", "discard", "script", "set"}
    ),
    external_reference_attributes=frozenset({"href", "src"}),
    forbidden_uri_schemes=("data", "file", "ftp", "http", "https", "javascript"),
)
