"""Network-free tests for the SVGenius-specific benchmark adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from svg_agentic_slm.benchmarks.svgenius import (
    SVGENIUS_DIFFICULTIES,
    SVGENIUS_HF_REVISION,
    SVGENIUS_KNOWN_EXCLUSIONS,
    SVGENIUS_TASK_REVISION,
    SVGeniusAdapter,
    SVGeniusPreparationConfig,
    main,
)
from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.data.text_to_svg_dataset import TextToSVGDataset


def _dataset_loader(**kwargs):
    difficulty = kwargs["split"]
    return [
        {
            "id": 1,
            "filename": f"{difficulty}-icon.svg",
            "difficulty": difficulty,
            "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
        }
    ]


def _caption_loader(difficulty: str, task_revision: str):
    return [
        {
            "question": [f"Draw the {difficulty} icon."],
            "answer": "",
            "image": f"../../../data/{difficulty}/{difficulty}-icon.png",
        }
    ]


def test_svgenius_adapter_prepares_pinned_held_out_snapshot(tmp_path: Path) -> None:
    result = SVGeniusAdapter(
        dataset_loader=_dataset_loader,
        caption_loader=_caption_loader,
    ).prepare(SVGeniusPreparationConfig(output_dir=tmp_path))

    records = read_jsonl(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    dataset = TextToSVGDataset(result.output_path)
    dataset.load()

    assert result.num_records == len(SVGENIUS_DIFFICULTIES)
    assert records[0]["task"] == "text_to_svg"
    assert records[0]["metadata"]["memory_eligible"] is False
    assert records[0]["metadata"]["data_partition"] == "held_out_test"
    assert manifest["benchmark_status"] == "adopted"
    assert manifest["adapter"] == "svgenius-text-to-svg-v3"
    assert manifest["manifest_schema_version"] == 3
    assert manifest["source_splits"] == ["easy", "medium", "hard"]
    assert manifest["excluded_source_splits"] == ["train"]
    assert manifest["data_partition"] == "held_out_test"
    assert manifest["license"] == "Apache-2.0"
    assert manifest["license_review_required"] is False
    assert manifest["strict"] is True
    assert manifest["limit_per_difficulty"] is None
    assert manifest["reference_validation"] == "strict_xml_structure_and_no_external_resources"
    assert manifest["memory_policy"]["memory_ingestion_allowed"] is False
    assert len(dataset) == len(SVGENIUS_DIFFICULTIES)
    assert dataset[0].output_svg.startswith("<svg")
    assert manifest["output_sha256"] == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    assert manifest["hf_revision"] == SVGENIUS_HF_REVISION
    assert manifest["task_revision"] == SVGENIUS_TASK_REVISION
    assert manifest["configured_known_exclusions"] == [
        exclusion.to_manifest() for exclusion in SVGENIUS_KNOWN_EXCLUSIONS
    ]
    assert manifest["applied_known_exclusions"] == []
    assert manifest["join_stats"]["easy"] == {
        "total_rows": 1,
        "selected_rows": 1,
        "total_captions": 1,
        "joined": 1,
        "missing_caption_rows": 0,
        "known_excluded_rows": 0,
        "unexpected_missing_caption_rows": 0,
        "unused_captions": 0,
    }


def test_svgenius_adapter_fails_on_unmatched_caption(tmp_path: Path) -> None:
    adapter = SVGeniusAdapter(
        dataset_loader=_dataset_loader,
        caption_loader=lambda difficulty, revision: [],
    )

    with pytest.raises(ValueError, match="unexpected.*no matching caption"):
        adapter.prepare(SVGeniusPreparationConfig(output_dir=tmp_path))


def test_svgenius_adapter_rejects_mutable_upstream_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable"):
        SVGeniusPreparationConfig(output_dir=tmp_path, hf_revision="main")

    with pytest.raises(ValueError, match="immutable"):
        SVGeniusPreparationConfig(output_dir=tmp_path, task_revision="main")


def test_svgenius_cli_rejects_broad_unmatched_bypass() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--allow-unmatched"])

    assert exc_info.value.code == 2


def test_svgenius_full_snapshot_requires_two_sided_join(tmp_path: Path) -> None:
    def caption_loader(difficulty: str, task_revision: str):
        return [
            *_caption_loader(difficulty, task_revision),
            {
                "question": ["Draw an orphan caption."],
                "answer": "",
                "image": f"../../../data/{difficulty}/orphan-{difficulty}.png",
            },
        ]

    adapter = SVGeniusAdapter(
        dataset_loader=_dataset_loader,
        caption_loader=caption_loader,
    )

    with pytest.raises(ValueError, match="no matching SVG row"):
        adapter.prepare(SVGeniusPreparationConfig(output_dir=tmp_path))


def test_svgenius_limited_snapshot_allows_unselected_captions(tmp_path: Path) -> None:
    def caption_loader(difficulty: str, task_revision: str):
        return [
            *_caption_loader(difficulty, task_revision),
            {
                "question": ["Draw a second icon."],
                "answer": "",
                "image": f"../../../data/{difficulty}/second-{difficulty}.png",
            },
        ]

    result = SVGeniusAdapter(
        dataset_loader=_dataset_loader,
        caption_loader=caption_loader,
    ).prepare(
        SVGeniusPreparationConfig(
            output_dir=tmp_path,
            limit_per_difficulty=1,
        )
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["join_stats"]["easy"]["unused_captions"] == 1


def test_svgenius_adapter_applies_only_pinned_known_exclusion(tmp_path: Path) -> None:
    known_asset_key = "page_38_ant_design_48353_icon_95"

    def dataset_loader(**kwargs):
        rows = list(_dataset_loader(**kwargs))
        if kwargs["split"] == "medium":
            rows.append(
                {
                    "id": 254,
                    "filename": f"{known_asset_key}.svg",
                    "difficulty": "medium",
                    "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
                }
            )
        return rows

    result = SVGeniusAdapter(
        dataset_loader=dataset_loader,
        caption_loader=_caption_loader,
    ).prepare(SVGeniusPreparationConfig(output_dir=tmp_path))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.num_records == 3
    assert manifest["records_by_split"] == {"easy": 1, "medium": 1, "hard": 1}
    assert manifest["join_stats"]["medium"] == {
        "total_rows": 2,
        "selected_rows": 2,
        "total_captions": 1,
        "joined": 1,
        "missing_caption_rows": 1,
        "known_excluded_rows": 1,
        "unexpected_missing_caption_rows": 0,
        "unused_captions": 0,
    }
    assert manifest["applied_known_exclusions"] == [
        {
            **SVGENIUS_KNOWN_EXCLUSIONS[0].to_manifest(),
            "filename": f"{known_asset_key}.svg",
            "upstream_id": "254",
        }
    ]


@pytest.mark.parametrize(
    ("hf_revision", "task_revision"),
    [
        ("a" * 40, SVGENIUS_TASK_REVISION),
        (SVGENIUS_HF_REVISION, "b" * 40),
    ],
)
def test_svgenius_known_exclusion_rejects_changed_revision_pair(
    tmp_path: Path,
    hf_revision: str,
    task_revision: str,
) -> None:
    known_asset_key = "page_38_ant_design_48353_icon_95"

    def dataset_loader(**kwargs):
        if kwargs["split"] != "medium":
            return _dataset_loader(**kwargs)
        return [
            {
                "id": 254,
                "filename": f"{known_asset_key}.svg",
                "difficulty": "medium",
                "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
            }
        ]

    adapter = SVGeniusAdapter(
        dataset_loader=dataset_loader,
        caption_loader=lambda difficulty, revision: (
            [] if difficulty == "medium" else _caption_loader(difficulty, revision)
        ),
    )

    with pytest.raises(ValueError, match="unexpected.*no matching caption"):
        adapter.prepare(
            SVGeniusPreparationConfig(
                output_dir=tmp_path,
                hf_revision=hf_revision,
                task_revision=task_revision,
            )
        )


def test_svgenius_known_exclusion_rejects_wrong_split(tmp_path: Path) -> None:
    known_asset_key = "page_38_ant_design_48353_icon_95"

    def dataset_loader(**kwargs):
        if kwargs["split"] != "easy":
            return _dataset_loader(**kwargs)
        return [
            {
                "id": 254,
                "filename": f"{known_asset_key}.svg",
                "difficulty": "easy",
                "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
            }
        ]

    adapter = SVGeniusAdapter(
        dataset_loader=dataset_loader,
        caption_loader=lambda difficulty, revision: (
            [] if difficulty == "easy" else _caption_loader(difficulty, revision)
        ),
    )

    with pytest.raises(ValueError, match="unexpected.*no matching caption"):
        adapter.prepare(SVGeniusPreparationConfig(output_dir=tmp_path))


def test_svgenius_known_exclusion_does_not_hide_additional_gap(tmp_path: Path) -> None:
    known_asset_key = "page_38_ant_design_48353_icon_95"

    def dataset_loader(**kwargs):
        rows = list(_dataset_loader(**kwargs))
        if kwargs["split"] == "medium":
            rows.extend(
                [
                    {
                        "id": 254,
                        "filename": f"{known_asset_key}.svg",
                        "difficulty": "medium",
                        "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
                    },
                    {
                        "id": 255,
                        "filename": "unexpected-gap.svg",
                        "difficulty": "medium",
                        "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
                    },
                ]
            )
        return rows

    adapter = SVGeniusAdapter(
        dataset_loader=dataset_loader,
        caption_loader=_caption_loader,
    )

    with pytest.raises(ValueError, match=r"1 unexpected.*unexpected-gap"):
        adapter.prepare(SVGeniusPreparationConfig(output_dir=tmp_path))


def test_svgenius_known_exclusion_detects_caption_policy_drift(tmp_path: Path) -> None:
    known_asset_key = "page_38_ant_design_48353_icon_95"

    def dataset_loader(**kwargs):
        if kwargs["split"] != "medium":
            return _dataset_loader(**kwargs)
        return [
            {
                "id": 254,
                "filename": f"{known_asset_key}.svg",
                "difficulty": "medium",
                "svg_code": '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
            }
        ]

    def caption_loader(difficulty: str, revision: str):
        if difficulty != "medium":
            return _caption_loader(difficulty, revision)
        return [
            {
                "question": ["A caption that should not exist at the pinned revision."],
                "answer": "",
                "image": f"../../../data/medium/{known_asset_key}.png",
            }
        ]

    adapter = SVGeniusAdapter(
        dataset_loader=dataset_loader,
        caption_loader=caption_loader,
    )

    with pytest.raises(ValueError, match="unexpectedly has a matching caption"):
        adapter.prepare(SVGeniusPreparationConfig(output_dir=tmp_path))
