"""Read ProteinGym files and enforce single-substitution sequence consistency.

This module uses only the standard library so inspection works before installing
the model stack. Positions in processed ProteinGym annotations are one-based.
"""

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ASSAY_ID = "BLAT_ECOLX_Stiffler_2015"
MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")


@dataclass(frozen=True)
class Substitution:
    wild_type: str
    position: int
    alternate: str


def parse_mutation(annotation: str, reference: str) -> Substitution:
    """Reject multis, indels, synonymous changes and numbering mismatches."""
    match = MUTATION.fullmatch(annotation)
    if match is None:
        raise ValueError(f"Expected one canonical substitution (e.g. H24A): {annotation!r}")
    wt, position, alt = match.groups()
    position = int(position)
    if position > len(reference):
        raise ValueError(f"{annotation}: position exceeds reference length {len(reference)}")
    if reference[position - 1] != wt:
        raise ValueError(f"{annotation}: reference residue is {reference[position - 1]}, not {wt}")
    if wt == alt:
        raise ValueError(f"{annotation}: not an amino-acid substitution")
    return Substitution(wt, position, alt)


def read_reference(path: Path, assay_id: str = ASSAY_ID) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("DMS_id") == assay_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one reference row for {assay_id}; found {len(matches)}")
    metadata = matches[0]
    sequence = metadata.get("target_seq", "")
    if not sequence or set(sequence) - set(AMINO_ACIDS):
        raise ValueError("Reference must contain only the 20 canonical amino acids")
    if int(metadata["seq_len"]) != len(sequence):
        raise ValueError("Reference seq_len disagrees with target_seq")
    if metadata.get("includes_multiple_mutants", "").upper() != "FALSE":
        raise ValueError("This experiment supports single-substitution assays only")
    return metadata


def read_assay(path: Path, metadata: dict) -> tuple[list[dict], list[Substitution], dict]:
    """Validate every row; never silently drop invalid observations."""
    reference = metadata["target_seq"]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {"mutant", "mutated_sequence", "DMS_score"}
        if not required.issubset(columns):
            raise ValueError(f"Missing columns: {sorted(required - set(columns))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Assay is empty")
    variants, seen = [], set()
    for line, row in enumerate(rows, start=2):
        try:
            variant = parse_mutation(row["mutant"], reference)
            if row["mutant"] in seen:
                raise ValueError(f"Duplicate mutant: {row['mutant']}")
            seen.add(row["mutant"])
            i = variant.position - 1
            expected = reference[:i] + variant.alternate + reference[i + 1:]
            if row["mutated_sequence"] != expected:
                raise ValueError(f"{row['mutant']}: mutated_sequence does not match annotation")
            score = float(row["DMS_score"])
            if not math.isfinite(score):
                raise ValueError("DMS_score is missing or non-finite")
            row["DMS_score"] = score
            variants.append(variant)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}, CSV line {line}: {error}") from error
    expected_count = int(metadata["DMS_number_single_mutants"])
    if len(rows) != expected_count or len(rows) != int(metadata["DMS_total_number_mutants"]):
        raise ValueError(f"Found {len(rows)} rows; reference expects {expected_count} singles")
    positions = sorted({v.position for v in variants})
    scores = [r["DMS_score"] for r in rows]
    summary = {
        "assay_id": metadata["DMS_id"], "columns": columns,
        "reference_length": len(reference), "n_variants": len(rows),
        "n_positions": len(positions), "positions": positions,
        "position_min": min(positions), "position_max": max(positions),
        "coverage_of_all_reference_substitutions": len(rows) / (19 * len(reference)),
        "coverage_at_observed_positions": len(rows) / (19 * len(positions)),
        "DMS_score_min": min(scores), "DMS_score_max": max(scores),
        "DMS_score_unique": len(set(scores)),
        "validation": "all rows passed; no filtering or numbering offsets applied",
        "examples": [{k: r[k] for k in ("mutant", "DMS_score")} for r in rows[:5]],
    }
    return rows, variants, summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
