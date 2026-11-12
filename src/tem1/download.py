"""Retrieve a versioned official archive and extract only the requested assay."""

import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .data import ASSAY_ID, read_assay, read_reference, sha256, write_json

RELEASE = "v1.3"
REFERENCE_COMMIT = "62eab5b3c83a3727b5b57bfbe377b24ef7545ccc"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{REFERENCE_COMMIT}"
    "/reference_files/DMS_substitutions.csv"
)
ARCHIVE_URL = (
    f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{RELEASE}/"
    "DMS_ProteinGym_substitutions.zip"
)


def download_file(url: str, destination: Path) -> None:
    """Stream to a temporary file; incomplete downloads never become cached inputs."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tem1-zero-shot/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_assay(archive: Path, filename: str, destination: Path) -> str:
    """Copy one exact basename, avoiding archive paths and ambiguous matches."""
    with zipfile.ZipFile(archive) as handle:
        matches = [n for n in handle.namelist() if Path(n).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Expected one archive member named {filename}; found {len(matches)}")
        with handle.open(matches[0]) as source, destination.open("wb") as out:
            shutil.copyfileobj(source, out)
    return matches[0]


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}; use a fresh data directory")
    return manifest


def prepare_data(directory: Path, archive: Path | None = None, reference: Path | None = None) -> dict:
    """Accept official local files for offline use; record their hashes and origin."""
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        if archive is not None or reference is not None:
            raise ValueError("Use a new data directory when importing local files")
        manifest = verify_manifest(directory)
        if manifest["assay_id"] != ASSAY_ID:
            raise ValueError("Cached data are for a different assay")
        return manifest
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as work:
        work = Path(work)
        reference_path = work / "DMS_substitutions.csv"
        if reference is None:
            print("Downloading pinned ProteinGym reference metadata...", flush=True)
            download_file(REFERENCE_URL, reference_path)
        else:
            shutil.copyfile(reference, reference_path)
        metadata = read_reference(reference_path)
        archive_path = archive
        if archive_path is None:
            archive_path = work / "substitutions.zip"
            print("Downloading official substitution archive (all assays; extracting TEM-1 only)...", flush=True)
            download_file(ARCHIVE_URL, archive_path)
        assay_path = work / f"{ASSAY_ID}.csv"
        member = extract_assay(archive_path, metadata["DMS_filename"], assay_path)
        _, _, summary = read_assay(assay_path, metadata)
        write_json(work / "inspection.json", summary)
        write_json(work / "assay_metadata.json", metadata)
        (work / "reference.fasta").write_text(f">{ASSAY_ID}\n{metadata['target_seq']}\n")
        files = [reference_path, assay_path, work / "reference.fasta", work / "assay_metadata.json", work / "inspection.json"]
        manifest = {
            "assay_id": ASSAY_ID, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_source": ARCHIVE_URL if archive is None else str(archive.resolve()),
            "reference_source": REFERENCE_URL if reference is None else str(reference.resolve()),
            "official_release": RELEASE if archive is None else "user-supplied; verify release",
            "archive_member": member, "archive_sha256": sha256(archive_path),
            "sha256": {p.name: sha256(p) for p in files},
        }
        for path in files:
            shutil.move(path, directory / path.name)
        write_json(directory / "manifest.json", manifest)
    return manifest
