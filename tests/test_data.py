"""Synthetic fixtures test scientific invariants, never benchmark performance."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tem1.data import ASSAY_ID, parse_mutation, read_assay, read_reference
from tem1.download import download_file, extract_assay, prepare_data, verify_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {
            "DMS_id": ASSAY_ID, "DMS_filename": f"{ASSAY_ID}.csv",
            "target_seq": "ACD", "seq_len": "3", "includes_multiple_mutants": "FALSE",
            "DMS_total_number_mutants": "3", "DMS_number_single_mutants": "3",
        }
        self.rows = [
            {"mutant": "A1C", "mutated_sequence": "CCD", "DMS_score": "-2.0"},
            {"mutant": "C2A", "mutated_sequence": "AAD", "DMS_score": "-1.0"},
            {"mutant": "D3A", "mutated_sequence": "ACA", "DMS_score": "0.5"},
        ]

    def write_csv(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def assay(self):
        return self.write_csv(self.root / "assay.csv", self.rows)

    def test_first_and_last_positions(self):
        self.assertEqual(parse_mutation("A1C", "ACD").position, 1)
        self.assertEqual(parse_mutation("D3A", "ACD").position, 3)

    def test_invalid_annotations(self):
        for annotation in ("A0C", "A01C", "A4C", "C1A", "A1A", "A1*", "A1X", "A1C:D3A", "del1", "a1c"):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                parse_mutation(annotation, "ACD")

    def test_valid_assay(self):
        rows, variants, summary = read_assay(self.assay(), self.metadata)
        self.assertEqual([v.position for v in variants], [1, 2, 3])
        self.assertEqual(summary["n_variants"], 3)
        self.assertEqual(rows[0]["DMS_score"], -2.0)

    def test_reconstruction_checks_entire_sequence(self):
        self.rows[0]["mutated_sequence"] = "CCA"  # correct annotated site, wrong background
        with self.assertRaisesRegex(ValueError, "mutated_sequence"):
            read_assay(self.assay(), self.metadata)

    def test_duplicate_mutants_rejected(self):
        self.rows[1] = self.rows[0].copy()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_assay(self.assay(), self.metadata)

    def test_missing_or_nonfinite_fitness_rejected(self):
        for value in ("", "nan", "inf", "-inf", "bad"):
            with self.subTest(value=value):
                self.rows[0]["DMS_score"] = value
                with self.assertRaises(ValueError):
                    read_assay(self.assay(), self.metadata)

    def test_count_mismatch_rejected(self):
        self.metadata["DMS_number_single_mutants"] = "4"
        with self.assertRaisesRegex(ValueError, "reference expects"):
            read_assay(self.assay(), self.metadata)

    def test_missing_column_rejected(self):
        for row in self.rows:
            del row["mutated_sequence"]
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            read_assay(self.assay(), self.metadata)

    def test_reference_length_checked(self):
        self.metadata["seq_len"] = "4"
        path = self.write_csv(self.root / "ref.csv", [self.metadata])
        with self.assertRaisesRegex(ValueError, "seq_len"):
            read_reference(path)

    def test_duplicate_reference_rejected(self):
        path = self.write_csv(self.root / "ref.csv", [self.metadata, self.metadata])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            read_reference(path)

    def test_local_archive_pipeline_and_checksum(self):
        reference = self.write_csv(self.root / "ref.csv", [self.metadata])
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(self.assay(), f"nested/{ASSAY_ID}.csv")
            handle.writestr("unused.csv", "irrelevant")
        output = self.root / "data"
        manifest = prepare_data(output, archive, reference)
        self.assertEqual(manifest["archive_member"], f"nested/{ASSAY_ID}.csv")
        self.assertEqual(verify_manifest(output), manifest)
        self.assertEqual(prepare_data(output), manifest)
        self.assertEqual(json.loads((output / "inspection.json").read_text())["n_variants"], 3)
        (output / f"{ASSAY_ID}.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_manifest(output)

    def test_ambiguous_archive_member_rejected(self):
        archive = self.root / "ambiguous.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("a/assay.csv", "a")
            handle.writestr("b/assay.csv", "b")
        with self.assertRaisesRegex(ValueError, "found 2"):
            extract_assay(archive, "assay.csv", self.root / "out.csv")

    def test_download_file_and_failure_cleanup(self):
        source, target = self.root / "source.txt", self.root / "target.txt"
        source.write_text("complete")
        download_file(source.as_uri(), target)
        self.assertEqual(target.read_text(), "complete")
        with self.assertRaises(OSError):
            download_file((self.root / "missing.txt").as_uri(), target)
        self.assertEqual(target.read_text(), "complete")
        self.assertFalse(target.with_suffix(".txt.part").exists())
