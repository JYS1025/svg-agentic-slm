"""Tests for SVG validator."""

from __future__ import annotations

import pytest

from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.svg.policy import STATIC_SVG_POLICY
from svg_agentic_slm.svg.validator import SVGValidator, safe_svg_element_names


def test_valid_svg() -> None:
    """Test that a well-formed SVG passes validation."""
    validator = SVGValidator()
    svg = '<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
    result = validator.validate(svg)

    assert result.is_valid
    assert result.has_svg_tag
    assert result.has_closing_tag
    assert not result.errors


def test_missing_svg_tag() -> None:
    """Test that missing <svg> tag is detected."""
    validator = SVGValidator()
    result = validator.validate("<div>Not an SVG</div>")

    assert not result.is_valid
    assert not result.has_svg_tag
    assert any("<svg>" in e for e in result.errors)


def test_missing_closing_tag() -> None:
    """Test that missing </svg> tag is detected."""
    validator = SVGValidator()
    result = validator.validate('<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/>')

    assert not result.is_valid
    assert result.has_svg_tag
    assert not result.has_closing_tag


def test_empty_content() -> None:
    """Test that empty content is rejected."""
    validator = SVGValidator()
    result = validator.validate("")

    assert not result.is_valid
    assert result.errors


def test_whitespace_only() -> None:
    """Test that whitespace-only content is rejected."""
    validator = SVGValidator()
    result = validator.validate("   \n\t  ")

    assert not result.is_valid


def test_missing_xmlns_warning() -> None:
    """Test that missing xmlns produces a warning."""
    validator = SVGValidator()
    svg = '<svg width="256" height="256"><circle r="10"/></svg>'
    result = validator.validate(svg)

    assert result.is_valid  # Warnings don't cause failure
    assert result.warnings
    assert any("xmlns" in w for w in result.warnings)


def test_rejects_malformed_xml() -> None:
    result = SVGValidator().validate("<svg><g></svg>")

    assert not result.is_valid
    assert not result.is_well_formed_xml
    assert any("XML parsing error" in error for error in result.errors)


def test_rejects_active_content_and_external_references() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" onclick="run()">
      <script>alert(1)</script>
      <image href="https://example.com/image.png"/>
      <rect fill="url(https://example.com/paint.svg#gradient)"/>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any("not allowed in static SVG" in error for error in result.errors)
    assert any("Event handler" in error for error in result.errors)
    assert any("External reference" in error for error in result.errors)
    assert any("External URL" in error for error in result.errors)


def test_allows_local_fragment_references() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="paint"/></defs>
      <rect fill="url(#paint)"/>
      <use href="#shape"/>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert result.is_valid


def test_rejects_svg_elements_with_incorrect_case() -> None:
    invalid_elements = [
        "Rect",
        "RECT",
        "clippath",
        "lineargradient",
        "textpath",
        "fegaussianblur",
    ]

    for element_name in invalid_elements:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            f"<{element_name}/>"
            "</svg>"
        )
        result = SVGValidator().validate(svg)

        assert not result.is_valid
        assert any(
            f"<{element_name}>" in error
            for error in result.errors
        )


def test_allows_canonical_case_sensitive_svg_elements() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <defs>
        <clipPath id="clip"><rect width="10" height="10"/></clipPath>
        <linearGradient id="paint"/>
        <radialGradient id="radial"/>
        <filter id="blur"><feGaussianBlur stdDeviation="1"/></filter>
      </defs>
      <text><textPath href="#path">Label</textPath></text>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert result.is_valid


def test_rejects_doctype() -> None:
    svg = '<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg"></svg>'

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any("DOCTYPE" in error for error in result.errors)


def test_rejects_processing_instructions_and_foreign_namespaces() -> None:
    svg = """
    <?xml-stylesheet href="https://example.com/theme.css"?>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:x="https://example.com/custom">
      <x:widget/>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any("processing instructions" in error for error in result.errors)
    assert any("Foreign element namespace" in error for error in result.errors)


def test_rejects_css_escape_and_smil_reference_mutation() -> None:
    css_escape = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <style>.shape { fill: u\\72l(https://example.com/paint.svg#x); }</style>
    </svg>
    """
    smil_mutation = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <animate href="#shape" attributeName="href"
               values="https://example.com/image.svg#shape"/>
    </svg>
    """

    css_result = SVGValidator().validate(css_escape)
    smil_result = SVGValidator().validate(smil_mutation)

    assert not css_result.is_valid
    assert any("not allowed in static SVG" in error for error in css_result.errors)
    assert not smil_result.is_valid
    assert any("not allowed in static SVG" in error for error in smil_result.errors)


def test_rejects_svg_handler_element() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <handler>run()</handler>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any("not allowed in static SVG" in error for error in result.errors)


def test_rejects_unlisted_network_capable_elements() -> None:
    video = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <video poster="relative.png"/>
    </svg>
    """
    anchor = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <a href="#shape" ping="collect"><rect id="shape"/></a>
    </svg>
    """

    video_result = SVGValidator().validate(video)
    anchor_result = SVGValidator().validate(anchor)

    assert not video_result.is_valid
    assert any("not allowed in static SVG" in error for error in video_result.errors)
    assert not anchor_result.is_valid
    assert any("not allowed in static SVG" in error for error in anchor_result.errors)


def test_rejects_foreign_event_attribute_namespace() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:ev="http://www.w3.org/2001/xml-events">
      <rect ev:event="click" ev:handler="relative.svg#handler"/>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any("Foreign attribute namespace" in error for error in result.errors)


def test_allows_local_xlink_href_but_rejects_external_xlink_href() -> None:
    local = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:xlink="http://www.w3.org/1999/xlink">
      <defs><symbol id="shape"><rect/></symbol></defs>
      <use xlink:href="#shape"/>
    </svg>
    """
    external = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:xlink="http://www.w3.org/1999/xlink">
      <use xlink:href="relative.svg#shape"/>
    </svg>
    """

    assert SVGValidator().validate(local).is_valid
    external_result = SVGValidator().validate(external)
    assert not external_result.is_valid
    assert any("External reference" in error for error in external_result.errors)


def test_self_closing_root_is_valid_without_explicit_closing_tag() -> None:
    result = SVGValidator().validate('<svg xmlns="http://www.w3.org/2000/svg"/>')

    assert result.is_valid
    assert result.has_svg_tag
    assert not result.has_closing_tag


def test_allows_safe_inline_style() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="paint"/></defs>
      <rect style="fill: url(#paint); stroke: rgb(0, 0, 0); opacity: 0.5"/>
    </svg>
    """

    result = SVGValidator().validate(svg)

    assert result.is_valid


@pytest.mark.parametrize(
    "obfuscated_url",
    [
        "ht&#10;tps://example.com/paint.svg",
        "java&#10;script:alert(1)",
    ],
)
def test_rejects_control_character_obfuscated_urls(obfuscated_url: str) -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<rect fill="url(&quot;{obfuscated_url}&quot;)"/>'
        "</svg>"
    )

    result = SVGValidator().validate(svg)

    assert not result.is_valid
    assert any(
        "Absolute URI schemes" in error or "External URL" in error
        for error in result.errors
    )


def test_allows_multiline_geometry_attributes() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <polyline points="0,0&#10;10,10&#10;20,0"/>
    </svg>
    """

    assert SVGValidator().validate(svg).is_valid


def test_safe_element_names_supports_documents_and_fragments() -> None:
    document = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        "<defs><linearGradient/></defs><rect/>"
        "</svg>"
    )

    assert safe_svg_element_names(document) == ["defs", "lineargradient", "rect"]
    assert safe_svg_element_names("<circle/><path/>", allow_fragment=True) == [
        "circle",
        "path",
    ]
    assert safe_svg_element_names("plain text", allow_fragment=True) is None


@pytest.mark.parametrize(
    "unsafe_svg",
    [
        '<svg xmlns="http://www.w3.org/2000/svg"><animate attributeName="x"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:x="urn:x"><x:item/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><Rect/></svg>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect fill="url(ht&#10;tps://example.com/x)"/>'
            "</svg>"
        ),
    ],
)
def test_safe_element_names_uses_the_shared_strict_policy(unsafe_svg: str) -> None:
    assert safe_svg_element_names(unsafe_svg) is None


def test_generator_prompt_and_validator_share_static_svg_policy() -> None:
    prompt = get_svg_generator_system_prompt().lower()

    for element in STATIC_SVG_POLICY.forbidden_elements:
        assert element in prompt
        result = SVGValidator().validate(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            f"<{element}/>"
            "</svg>"
        )
        assert not result.is_valid
        assert any("static SVG policy" in error for error in result.errors)
    assert "event-handler" in prompt
    assert "#fragment" in prompt
