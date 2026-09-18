"""Fail-closed identity checks for research dataset partitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


REQUIRED_PROTECTED_ROLES = ("sft", "rag", "eval")


@dataclass(frozen=True)
class IdentityManifest:
    """Stable IDs and provenance loaded from one or more manifest files."""

    role: str
    ids: frozenset[str]
    files: tuple[dict[str, Any], ...]


def load_identity_manifest(role: str, paths: Iterable[str | Path]) -> IdentityManifest:
    """Load canonical sample IDs, rejecting missing IDs and within-role duplicates."""
    normalized_role = role.strip().lower()
    source_paths = tuple(Path(path).resolve() for path in paths)
    if not source_paths:
        raise ValueError(f"At least one {normalized_role!r} identity manifest is required.")

    identities: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(f"{normalized_role} manifest not found: {source_path}")
        provenance.append(
            {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
            }
        )
        for location, row in _iter_manifest_records(source_path):
            sample_id = canonical_sample_id(row, location=location)
            if sample_id in identities:
                raise ValueError(
                    f"Duplicate sample_id {sample_id!r} in {normalized_role} manifests "
                    f"(last seen at {location})."
                )
            identities.add(sample_id)
    if not identities:
        raise ValueError(f"{normalized_role} identity manifests contain no records.")
    return IdentityManifest(
        role=normalized_role,
        ids=frozenset(identities),
        files=tuple(provenance),
    )


def validate_disjoint_manifests(
    *,
    critic_paths: Iterable[str | Path],
    protected_paths: dict[str, Iterable[str | Path]],
) -> dict[str, Any]:
    """Validate Critic/SFT/RAG/eval partitions are pairwise disjoint."""
    missing_roles = [role for role in REQUIRED_PROTECTED_ROLES if not protected_paths.get(role)]
    if missing_roles:
        raise ValueError(
            "Fail-closed split validation requires manifests for: "
            + ", ".join(missing_roles)
        )

    manifests = [load_identity_manifest("critic", critic_paths)]
    manifests.extend(
        load_identity_manifest(role, protected_paths[role])
        for role in REQUIRED_PROTECTED_ROLES
    )
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(manifests):
        for right in manifests[left_index + 1 :]:
            shared = sorted(left.ids & right.ids)
            if shared:
                overlaps.append(
                    {
                        "roles": [left.role, right.role],
                        "count": len(shared),
                        "sample_ids": shared[:100],
                        "sample_ids_truncated": len(shared) > 100,
                    }
                )
    if overlaps:
        summary = "; ".join(
            f"{item['roles'][0]}<->{item['roles'][1]}={item['count']}"
            for item in overlaps
        )
        raise ValueError(f"Dataset leakage detected: {summary}")

    return {
        "schema_version": 1,
        "status": "passed",
        "identity_contract": (
            "sample_id, or metadata.dataset_id + metadata.record_id; "
            "unqualified IDs are rejected"
        ),
        "pairwise_disjoint": True,
        "roles": {
            manifest.role: {
                "rows": len(manifest.ids),
                "files": list(manifest.files),
                "ids_sha256": hashlib.sha256(
                    "\n".join(sorted(manifest.ids)).encode("utf-8")
                ).hexdigest(),
            }
            for manifest in manifests
        },
    }


def read_manifest_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Return records with a normalized sample_id for downstream batch jobs."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path_value in paths:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        for location, row in _iter_manifest_records(path):
            sample_id = canonical_sample_id(row, location=location)
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id {sample_id!r} at {location}.")
            seen.add(sample_id)
            normalized = dict(row)
            normalized["sample_id"] = sample_id
            records.append(normalized)
    return records


def canonical_sample_id(row: dict[str, Any], *, location: str) -> str:
    """Derive one globally qualified ID without accepting ambiguous bare IDs."""
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit = row.get("sample_id", metadata.get("sample_id"))
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    dataset_id = _first_value(
        row,
        metadata,
        keys=("dataset_id", "source_dataset_id", "dataset"),
    )
    record_id = _first_value(
        row,
        metadata,
        keys=("record_id", "source_record_id", "item_id"),
    )
    if dataset_id is None or record_id is None:
        raise ValueError(
            f"Record at {location} needs sample_id or both dataset_id and record_id."
        )
    return f"{str(dataset_id).strip()}::{str(record_id).strip()}"


def _iter_manifest_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Manifest record must be an object at {path}:{line_number}.")
                yield f"{path}:{line_number}", payload
        return
    if suffix != ".json":
        raise ValueError(f"Identity manifest must be JSON or JSONL: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for index, row in enumerate(payload):
            yield _checked_record(row, f"{path}#records[{index}]")
        return
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be an object or array: {path}")
    if isinstance(payload.get("ids"), list):
        for index, sample_id in enumerate(payload["ids"]):
            yield f"{path}#ids[{index}]", {"sample_id": sample_id}
        return
    if isinstance(payload.get("records"), list):
        for index, row in enumerate(payload["records"]):
            yield _checked_record(row, f"{path}#records[{index}]")
        return
    splits = payload.get("splits")
    if isinstance(splits, dict):
        for split_name, descriptor in splits.items():
            if not isinstance(descriptor, dict) or not descriptor.get("file"):
                raise ValueError(f"Invalid split descriptor {split_name!r} in {path}.")
            split_path = (path.parent / str(descriptor["file"])).resolve()
            if not split_path.is_relative_to(path.parent.resolve()):
                raise ValueError(f"Split path escapes manifest directory: {split_path}")
            yield from _iter_manifest_records(split_path)
        return
    yield str(path), payload


def _checked_record(value: Any, location: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest record must be an object at {location}.")
    return location, value


def _first_value(
    row: dict[str, Any], metadata: dict[str, Any], *, keys: tuple[str, ...]
) -> Any:
    for mapping in (row, metadata):
        for key in keys:
            value = mapping.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
