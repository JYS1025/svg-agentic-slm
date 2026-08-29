"""Restricted discrete-SVG codec for the OmniSVG-inspired ablation.

This module is intentionally independent from the production raw-XML path.  It
borrows OmniSVG's high-level command/coordinate/color factorisation, but it is
not token-ID or checkpoint compatible with the official OmniSVG codec.  The
restricted grammar makes a representation experiment reproducible without
silently approximating unsupported SVG features.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Final

from lxml import etree

CODEC_NAME: Final = "omnisvg-inspired-discrete-svg"
CODEC_VERSION: Final = "1.0.0"
TOKEN_NAMESPACE: Final = "svgd1"
SVG_NAMESPACE: Final = "http://www.w3.org/2000/svg"

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(rf"^{_NUMBER_PATTERN}$")
_PATH_LEXEME_RE = re.compile(rf"[MLCAZ]|{_NUMBER_PATTERN}")
_SEPARATOR_RE = re.compile(r"^[\s,]*$")
_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

_COMMAND_ARITY: Final[dict[str, int]] = {
    "M": 2,
    "L": 2,
    "C": 6,
    "A": 7,
    "Z": 0,
}


class DiscreteSVGCodecError(ValueError):
    """Raised when SVG or tokens fall outside the restricted codec contract."""


@dataclass(frozen=True)
class _PathSegment:
    command: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class _RestrictedPath:
    segments: tuple[_PathSegment, ...]
    fill: str


class OmniSVGDiscreteCodec:
    """Encode a normalized path-only SVG subset as namespaced string tokens.

    The default 200 by 200 grid follows the coordinate resolution used by the
    OmniSVG representation.  One coordinate token represents an ``(x, y)``
    pair, giving 40,000 coordinate tokens at the default resolution.  RGB is
    quantized to four bits per channel.

    Supported input is deliberately narrow:

    * one SVG root with a positive ``viewBox`` and no other root attributes;
    * one or more direct ``path`` children;
    * path attributes ``d`` and optional ``fill`` only;
    * explicit absolute ``M``, ``L``, ``C``, ``A`` and ``Z`` commands;
    * ``none``, ``#RGB`` and ``#RRGGBB`` fills.

    Anything else fails closed instead of being dropped or approximated.
    """

    def __init__(self, *, grid_size: int = 200, color_levels: int = 16) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if color_levels != 16:
            raise ValueError("codec version 1 requires exactly 16 RGB levels")

        self.grid_size = grid_size
        self.color_levels = color_levels
        self.sop_token = self._token("sop")
        self.eop_token = self._token("eop")
        self.eos_token = self._token("eos")
        self.none_color_token = self._token("color:none")
        self.command_tokens = {
            command: self._token(f"cmd:{command}") for command in ("M", "L", "C", "A", "Z", "F")
        }
        self._tokens_to_commands = {
            token: command for command, token in self.command_tokens.items()
        }
        self.flag_tokens = {flag: self._token(f"flag:{flag}") for flag in (0, 1)}

        coordinate_count = grid_size * grid_size
        self._coordinate_width = len(str(coordinate_count - 1))
        self._rotation_width = 3
        self._vocabulary = tuple(self._build_vocabulary())
        self._vocabulary_sha256 = hashlib.sha256(
            "\n".join(self._vocabulary).encode("utf-8")
        ).hexdigest()

    def vocabulary_tokens(self) -> list[str]:
        """Return tokens in their deterministic registration order.

        Callers may register this list as ``additional_special_tokens`` on a
        trainable Hugging Face tokenizer.  Token IDs are intentionally assigned
        by that tokenizer and are never mixed with raw XML targets.
        """

        return list(self._vocabulary)

    def encode_svg(self, svg: str) -> list[str]:
        """Encode supported SVG XML into a deterministic discrete token list."""

        paths, view_box = self._parse_svg(svg)
        min_x, min_y, width, height = view_box
        encoded: list[str] = []

        for path in paths:
            encoded.append(self.sop_token)
            for segment in path.segments:
                command = segment.command
                values = segment.values
                encoded.append(self.command_tokens[command])

                if command in {"M", "L"}:
                    encoded.append(
                        self._point_token(values[0], values[1], min_x, min_y, width, height)
                    )
                elif command == "C":
                    for offset in (0, 2, 4):
                        encoded.append(
                            self._point_token(
                                values[offset],
                                values[offset + 1],
                                min_x,
                                min_y,
                                width,
                                height,
                            )
                        )
                elif command == "A":
                    radius_token = self._radius_token(values[0], values[1], width, height)
                    encoded.extend(
                        (
                            radius_token,
                            self._rotation_token(values[2]),
                            self.flag_tokens[int(values[3])],
                            self.flag_tokens[int(values[4])],
                            self._point_token(
                                values[5], values[6], min_x, min_y, width, height
                            ),
                        )
                    )

            encoded.extend(
                (
                    self.command_tokens["F"],
                    self._color_token(path.fill),
                    self.eop_token,
                )
            )

        encoded.append(self.eos_token)
        return encoded

    def decode_tokens(self, tokens: list[str]) -> str:
        """Decode a complete token list into canonical 200-grid SVG XML."""

        if not isinstance(tokens, list) or not tokens:
            raise DiscreteSVGCodecError("token sequence must be a non-empty list")
        if any(not isinstance(token, str) for token in tokens):
            raise DiscreteSVGCodecError("every discrete SVG token must be a string")
        if tokens[-1] != self.eos_token or self.eos_token in tokens[:-1]:
            raise DiscreteSVGCodecError("sequence must contain one final EOS token")

        cursor = 0
        decoded_paths: list[tuple[str, str]] = []
        while cursor < len(tokens) - 1:
            if tokens[cursor] != self.sop_token:
                raise DiscreteSVGCodecError(f"expected SOP token at index {cursor}")
            cursor += 1
            segments: list[str] = []
            has_move = False
            closed_subpath = False

            while cursor < len(tokens) and tokens[cursor] != self.command_tokens["F"]:
                command = self._tokens_to_commands.get(tokens[cursor])
                if command not in _COMMAND_ARITY:
                    raise DiscreteSVGCodecError(
                        f"expected M/L/C/A/Z/F command token at index {cursor}"
                    )
                cursor += 1

                if command == "M":
                    point, cursor = self._take_point(tokens, cursor)
                    segments.append(f"M {point[0]} {point[1]}")
                    has_move = True
                    closed_subpath = False
                elif not has_move:
                    raise DiscreteSVGCodecError("each path must begin with an M command")
                elif command == "L":
                    if closed_subpath:
                        raise DiscreteSVGCodecError("a closed subpath must restart with M")
                    point, cursor = self._take_point(tokens, cursor)
                    segments.append(f"L {point[0]} {point[1]}")
                elif command == "C":
                    if closed_subpath:
                        raise DiscreteSVGCodecError("a closed subpath must restart with M")
                    first, cursor = self._take_point(tokens, cursor)
                    second, cursor = self._take_point(tokens, cursor)
                    end, cursor = self._take_point(tokens, cursor)
                    segments.append(
                        f"C {first[0]} {first[1]} {second[0]} {second[1]} "
                        f"{end[0]} {end[1]}"
                    )
                elif command == "A":
                    if closed_subpath:
                        raise DiscreteSVGCodecError("a closed subpath must restart with M")
                    radius, cursor = self._take_point(tokens, cursor)
                    if radius[0] == 0 or radius[1] == 0:
                        raise DiscreteSVGCodecError("arc radius tokens must be non-zero")
                    rotation, cursor = self._take_rotation(tokens, cursor)
                    large_arc, cursor = self._take_flag(tokens, cursor)
                    sweep, cursor = self._take_flag(tokens, cursor)
                    end, cursor = self._take_point(tokens, cursor)
                    segments.append(
                        f"A {radius[0]} {radius[1]} {rotation} {large_arc} {sweep} "
                        f"{end[0]} {end[1]}"
                    )
                elif command == "Z":
                    if closed_subpath:
                        raise DiscreteSVGCodecError("duplicate Z command in one subpath")
                    segments.append("Z")
                    closed_subpath = True

            if not has_move or not segments:
                raise DiscreteSVGCodecError("a discrete path must contain geometry")
            if cursor >= len(tokens) or tokens[cursor] != self.command_tokens["F"]:
                raise DiscreteSVGCodecError("path is missing its F command")
            cursor += 1
            if cursor >= len(tokens):
                raise DiscreteSVGCodecError("F command is missing a color token")
            fill = self._decode_color_token(tokens[cursor])
            cursor += 1
            if cursor >= len(tokens) or tokens[cursor] != self.eop_token:
                raise DiscreteSVGCodecError("path must terminate with EOP")
            cursor += 1
            decoded_paths.append((" ".join(segments), fill))

        if not decoded_paths:
            raise DiscreteSVGCodecError("sequence must contain at least one path")

        path_xml = "".join(
            f'<path d="{path_data}" fill="{fill}"/>'
            for path_data, fill in decoded_paths
        )
        return (
            f'<svg xmlns="{SVG_NAMESPACE}" '
            f'viewBox="0 0 {self.grid_size} {self.grid_size}">{path_xml}</svg>'
        )

    def manifest(self) -> dict[str, object]:
        """Return a JSON-serializable description of the codec contract."""

        manifest: dict[str, object] = {
            "name": CODEC_NAME,
            "version": CODEC_VERSION,
            "token_namespace": TOKEN_NAMESPACE,
            "representation": "one-token-per-2d-coordinate",
            "canvas": {
                "view_box": [0, 0, self.grid_size, self.grid_size],
                "coordinate_grid_size": self.grid_size,
                "coordinate_token_count": self.grid_size * self.grid_size,
            },
            "color": {
                "space": "rgb",
                "levels_per_channel": self.color_levels,
                "supports_none": True,
            },
            "commands": ["M", "L", "C", "A", "Z", "F"],
            "sequence": "(SOP geometry F color EOP)+ EOS",
            "special_tokens": {
                "sop": self.sop_token,
                "eop": self.eop_token,
                "eos": self.eos_token,
            },
            "supported_svg": {
                "elements": ["svg", "path"],
                "root_attributes": ["viewBox"],
                "path_attributes": ["d", "fill"],
                "path_commands": ["M", "L", "C", "A", "Z"],
                "absolute_commands_only": True,
            },
            "vocabulary_size": len(self._vocabulary),
            "vocabulary_sha256": self._vocabulary_sha256,
            "official_omnisvg_checkpoint_compatible": False,
            "raw_xml_interoperability": "none; this codec is an isolated ablation target",
        }
        # Make accidental non-JSON additions fail at the contract boundary.
        json.dumps(manifest, sort_keys=True)
        return manifest

    def _build_vocabulary(self) -> list[str]:
        tokens = [self.sop_token, self.eop_token, self.eos_token]
        tokens.extend(self.command_tokens[command] for command in ("M", "L", "C", "A", "Z", "F"))
        tokens.extend(self.flag_tokens[flag] for flag in (0, 1))
        tokens.extend(self._rotation_token(rotation) for rotation in range(360))
        tokens.extend(
            self._coordinate_token(index)
            for index in range(self.grid_size * self.grid_size)
        )
        tokens.append(self.none_color_token)
        tokens.extend(self._rgb_token(index) for index in range(self.color_levels**3))
        if len(tokens) != len(set(tokens)):
            raise RuntimeError("discrete SVG vocabulary contains duplicate tokens")
        return tokens

    def _parse_svg(
        self, svg: str
    ) -> tuple[list[_RestrictedPath], tuple[float, float, float, float]]:
        if not isinstance(svg, str) or not svg.strip():
            raise DiscreteSVGCodecError("SVG input must be a non-empty string")
        if len(svg.encode("utf-8")) > 8 * 1024 * 1024:
            raise DiscreteSVGCodecError("SVG input exceeds the 8 MiB codec limit")

        parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            load_dtd=False,
            dtd_validation=False,
            recover=False,
            remove_comments=False,
            remove_pis=False,
            huge_tree=False,
        )
        try:
            root = etree.fromstring(svg.encode("utf-8"), parser=parser)
        except (etree.XMLSyntaxError, ValueError) as exc:
            raise DiscreteSVGCodecError(f"invalid SVG XML: {exc}") from exc

        if root.getroottree().docinfo.doctype:
            raise DiscreteSVGCodecError("DOCTYPE declarations are not supported")
        if not isinstance(root.tag, str):
            raise DiscreteSVGCodecError("SVG root must be an element")
        root_name = etree.QName(root)
        if root_name.localname != "svg" or root_name.namespace != SVG_NAMESPACE:
            raise DiscreteSVGCodecError(
                "root must be an SVG element in the standard SVG namespace"
            )
        namespace_values = {value for value in root.nsmap.values() if value is not None}
        if namespace_values != {SVG_NAMESPACE}:
            raise DiscreteSVGCodecError("additional XML namespaces are not supported")
        if set(root.attrib) != {"viewBox"}:
            raise DiscreteSVGCodecError("root must contain only the viewBox attribute")
        if root.text and root.text.strip():
            raise DiscreteSVGCodecError("text nodes are not supported")

        view_box_values = self._parse_number_sequence(root.attrib["viewBox"], expected=4)
        min_x, min_y, width, height = view_box_values
        if width <= 0 or height <= 0:
            raise DiscreteSVGCodecError("viewBox width and height must be positive")

        paths: list[_RestrictedPath] = []
        for child in root.iterchildren():
            if not isinstance(child.tag, str):
                raise DiscreteSVGCodecError("comments and processing instructions are unsupported")
            child_name = etree.QName(child)
            if child_name.localname != "path" or child_name.namespace != SVG_NAMESPACE:
                raise DiscreteSVGCodecError("only direct SVG path children are supported")
            if len(child):
                raise DiscreteSVGCodecError("path elements cannot contain child nodes")
            if (child.text and child.text.strip()) or (child.tail and child.tail.strip()):
                raise DiscreteSVGCodecError("text nodes are not supported")
            if not set(child.attrib).issubset({"d", "fill"}) or "d" not in child.attrib:
                raise DiscreteSVGCodecError("path supports only required d and optional fill")
            fill = child.attrib.get("fill", "#000000")
            self._validate_color(fill)
            paths.append(
                _RestrictedPath(
                    segments=tuple(self._parse_path_data(child.attrib["d"])),
                    fill=fill,
                )
            )
            if len(paths) > 2048:
                raise DiscreteSVGCodecError("SVG exceeds the 2048 path limit")

        if not paths:
            raise DiscreteSVGCodecError("SVG must contain at least one path")
        return paths, (min_x, min_y, width, height)

    def _parse_path_data(self, path_data: str) -> list[_PathSegment]:
        lexemes: list[str] = []
        position = 0
        for match in _PATH_LEXEME_RE.finditer(path_data):
            if not _SEPARATOR_RE.fullmatch(path_data[position : match.start()]):
                raise DiscreteSVGCodecError(
                    "path contains a relative or unsupported SVG command"
                )
            lexemes.append(match.group(0))
            position = match.end()
        if not _SEPARATOR_RE.fullmatch(path_data[position:]):
            raise DiscreteSVGCodecError("path contains an unsupported SVG command")
        if not lexemes:
            raise DiscreteSVGCodecError("path data cannot be empty")

        segments: list[_PathSegment] = []
        cursor = 0
        has_move = False
        closed_subpath = False
        while cursor < len(lexemes):
            command = lexemes[cursor]
            if command not in _COMMAND_ARITY:
                raise DiscreteSVGCodecError(
                    "every segment must use an explicit absolute M/L/C/A/Z command"
                )
            cursor += 1
            arity = _COMMAND_ARITY[command]
            raw_values = lexemes[cursor : cursor + arity]
            if len(raw_values) != arity or any(not _NUMBER_RE.fullmatch(v) for v in raw_values):
                raise DiscreteSVGCodecError(f"command {command} has invalid arity")
            values = tuple(self._finite_float(value) for value in raw_values)
            cursor += arity

            if command == "M":
                has_move = True
                closed_subpath = False
            elif not has_move:
                raise DiscreteSVGCodecError("each path must begin with an M command")
            elif closed_subpath:
                raise DiscreteSVGCodecError("a closed subpath must restart with M")

            if command == "A":
                if values[0] <= 0 or values[1] <= 0:
                    raise DiscreteSVGCodecError("arc radii must be positive")
                if values[3] not in (0.0, 1.0) or values[4] not in (0.0, 1.0):
                    raise DiscreteSVGCodecError("arc flags must be exactly 0 or 1")
            if command == "Z":
                closed_subpath = True
            segments.append(_PathSegment(command=command, values=values))

        return segments

    def _point_token(
        self,
        x: float,
        y: float,
        min_x: float,
        min_y: float,
        width: float,
        height: float,
    ) -> str:
        x_index = self._quantize_axis(x, min_x, width, "x coordinate")
        y_index = self._quantize_axis(y, min_y, height, "y coordinate")
        return self._coordinate_token(y_index * self.grid_size + x_index)

    def _radius_token(self, x_radius: float, y_radius: float, width: float, height: float) -> str:
        x_index = self._quantize_axis(x_radius, 0.0, width, "arc x radius")
        y_index = self._quantize_axis(y_radius, 0.0, height, "arc y radius")
        if x_index == 0 or y_index == 0:
            raise DiscreteSVGCodecError("arc radius is too small for the coordinate grid")
        return self._coordinate_token(y_index * self.grid_size + x_index)

    def _quantize_axis(self, value: float, minimum: float, extent: float, label: str) -> int:
        normalized = (value - minimum) / extent
        tolerance = 1e-12
        if normalized < -tolerance or normalized > 1.0 + tolerance:
            raise DiscreteSVGCodecError(f"{label} falls outside the viewBox")
        normalized = min(1.0, max(0.0, normalized))
        return int(math.floor(normalized * (self.grid_size - 1) + 0.5))

    def _color_token(self, color: str) -> str:
        normalized = self._validate_color(color)
        if normalized == "none":
            return self.none_color_token
        red = int(normalized[1:3], 16)
        green = int(normalized[3:5], 16)
        blue = int(normalized[5:7], 16)
        red_q = int(math.floor(red * 15 / 255 + 0.5))
        green_q = int(math.floor(green * 15 / 255 + 0.5))
        blue_q = int(math.floor(blue * 15 / 255 + 0.5))
        return self._rgb_token((red_q << 8) | (green_q << 4) | blue_q)

    def _decode_color_token(self, token: str) -> str:
        if token == self.none_color_token:
            return "none"
        prefix = self._token_prefix("rgb:")
        if not token.startswith(prefix) or not token.endswith("|>"):
            raise DiscreteSVGCodecError("F command requires a discrete RGB or none token")
        value_text = token[len(prefix) : -2]
        if len(value_text) != 3 or not re.fullmatch(r"[0-9a-f]{3}", value_text):
            raise DiscreteSVGCodecError("invalid discrete RGB token")
        value = int(value_text, 16)
        red = (value >> 8) & 0xF
        green = (value >> 4) & 0xF
        blue = value & 0xF
        return f"#{red * 17:02x}{green * 17:02x}{blue * 17:02x}"

    def _validate_color(self, color: str) -> str:
        normalized = color.strip().lower()
        if normalized == "none":
            return normalized
        if not _HEX_COLOR_RE.fullmatch(normalized):
            raise DiscreteSVGCodecError("fill must be none, #RGB, or #RRGGBB")
        if len(normalized) == 4:
            normalized = "#" + "".join(component * 2 for component in normalized[1:])
        return normalized

    def _take_point(self, tokens: list[str], cursor: int) -> tuple[tuple[int, int], int]:
        if cursor >= len(tokens):
            raise DiscreteSVGCodecError("command is missing a 2D coordinate token")
        return self._decode_coordinate_token(tokens[cursor]), cursor + 1

    def _take_rotation(self, tokens: list[str], cursor: int) -> tuple[int, int]:
        if cursor >= len(tokens):
            raise DiscreteSVGCodecError("A command is missing a rotation token")
        prefix = self._token_prefix("rot:")
        token = tokens[cursor]
        if not token.startswith(prefix) or not token.endswith("|>"):
            raise DiscreteSVGCodecError("A command requires a rotation token")
        value_text = token[len(prefix) : -2]
        if len(value_text) != self._rotation_width or not value_text.isdigit():
            raise DiscreteSVGCodecError("invalid rotation token")
        rotation = int(value_text)
        if rotation >= 360:
            raise DiscreteSVGCodecError("rotation token is outside [0, 359]")
        return rotation, cursor + 1

    def _take_flag(self, tokens: list[str], cursor: int) -> tuple[int, int]:
        if cursor >= len(tokens):
            raise DiscreteSVGCodecError("A command is missing an arc flag token")
        for value, token in self.flag_tokens.items():
            if tokens[cursor] == token:
                return value, cursor + 1
        raise DiscreteSVGCodecError("A command requires a 0 or 1 flag token")

    def _decode_coordinate_token(self, token: str) -> tuple[int, int]:
        prefix = self._token_prefix("xy:")
        if not token.startswith(prefix) or not token.endswith("|>"):
            raise DiscreteSVGCodecError("expected a 2D coordinate token")
        index_text = token[len(prefix) : -2]
        if len(index_text) != self._coordinate_width or not index_text.isdigit():
            raise DiscreteSVGCodecError("invalid 2D coordinate token")
        index = int(index_text)
        if index >= self.grid_size * self.grid_size:
            raise DiscreteSVGCodecError("2D coordinate token is outside the grid")
        y, x = divmod(index, self.grid_size)
        return x, y

    def _coordinate_token(self, index: int) -> str:
        return self._token(f"xy:{index:0{self._coordinate_width}d}")

    def _rotation_token(self, rotation: float) -> str:
        normalized = int(math.floor(rotation % 360 + 0.5)) % 360
        return self._token(f"rot:{normalized:0{self._rotation_width}d}")

    def _rgb_token(self, index: int) -> str:
        return self._token(f"rgb:{index:03x}")

    def _token_prefix(self, value: str) -> str:
        return f"<|{TOKEN_NAMESPACE}:{value}"

    def _token(self, value: str) -> str:
        return f"<|{TOKEN_NAMESPACE}:{value}|>"

    @staticmethod
    def _parse_number_sequence(value: str, *, expected: int) -> tuple[float, ...]:
        pieces = [piece for piece in re.split(r"[\s,]+", value.strip()) if piece]
        if len(pieces) != expected or any(not _NUMBER_RE.fullmatch(piece) for piece in pieces):
            raise DiscreteSVGCodecError(f"expected exactly {expected} finite numbers")
        return tuple(OmniSVGDiscreteCodec._finite_float(piece) for piece in pieces)

    @staticmethod
    def _finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise DiscreteSVGCodecError("coordinates must be finite")
        return number
