"""Three explicit stages: download, inspect, and run."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import ASSAY_ID, read_assay, read_reference, sha256, write_json
from .download import prepare_data, verify_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="TEM-1 / ProteinGym / ESM-2 zero-shot baseline")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("download", "inspect", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, default=Path("data/tem1"))
        if name == "download":
            sub.add_argument("--archive", type=Path, help="Already downloaded official ZIP")
            sub.add_argument("--reference", type=Path, help="Already downloaded reference CSV")
        if name == "run":
            sub.add_argument("--output", type=Path, default=Path("results/esm2_35m"))
            sub.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
            sub.add_argument("--revision", help="HF commit; required when changing the default model")
            sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            sub.add_argument("--batch-size", type=int, default=4)
            sub.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
            sub.add_argument("--warmup-batches", type=int, default=1)
            sub.add_argument("--limit-positions", type=int, help="Smoke test: first N observed positions; not a full benchmark")
            sub.add_argument("--local-files-only", action="store_true", help="Use cached model/tokenizer without downloading")
    return root


def run(args, metadata, rows, variants, summary, manifest):
    if args.batch_size < 1 or args.threads < 1 or args.warmup_batches < 0:
        raise ValueError("Batch size/threads must be positive; warmup must be nonnegative")
    if args.limit_positions is not None and args.limit_positions < 1:
        raise ValueError("--limit-positions must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; select a new --output to preserve previous runs")
    from .scoring import MODEL_ID, MODEL_REVISION, load_model, score_variants, select_device
    from .evaluation import evaluate, plot_results
    import torch

    revision = args.revision
    if revision is None:
        if args.model != MODEL_ID:
            raise ValueError("Supply --revision with the commit for your chosen model")
        revision = MODEL_REVISION
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = select_device(args.device)
    full_count = len(rows)
    if args.limit_positions is not None:
        selected = set(sorted({v.position for v in variants})[:args.limit_positions])
        pairs = [(r, v) for r, v in zip(rows, variants) if v.position in selected]
        rows, variants = map(list, zip(*pairs))
    smoke = args.limit_positions is not None
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "inspection.json", summary)
    write_json(args.output / "input_manifest.json", manifest)
    write_json(args.output / "assay_metadata.json", metadata)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["revision"] = revision
    write_json(args.output / "config.json", config)
    start = time.perf_counter()
    try:
        model, tokenizer, model_info = load_model(args.model, revision, device, args.local_files_only)
        predictions, performance = score_variants(
            model, tokenizer, metadata["target_seq"], variants, args.batch_size, args.warmup_batches,
        )
        observed = [r["DMS_score"] for r in rows]
        metrics = evaluate(observed, predictions)
        metrics.update({"assay_id": ASSAY_ID, "scope": "smoke_test" if smoke else "full_assay", "full_assay_n": full_count})
        with (args.output / "predictions.csv").open("w", newline="") as handle:
            fields = ["mutant", "position", "wild_type", "alternate", "DMS_score", "esm2_score"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, variant, score in zip(rows, variants, predictions):
                writer.writerow({"mutant": row["mutant"], "position": variant.position,
                                 "wild_type": variant.wild_type, "alternate": variant.alternate,
                                 "DMS_score": row["DMS_score"], "esm2_score": score})
        plot_results(observed, predictions, [v.position for v in variants], args.output,
                     ASSAY_ID + (" — SMOKE TEST ONLY" if smoke else ""))
        source_files = sorted(Path(__file__).parent.glob("*.py"))
        environment = {
            "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch_cuda_version": torch.version.cuda, "seed": 0, "tf32": False,
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scipy", "matplotlib")},
            "source_sha256": {p.name: sha256(p) for p in source_files},
        }
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        if freeze.returncode == 0:
            (args.output / "environment.txt").write_text(freeze.stdout)
        else:
            # uv-created environments need not include pip.
            packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
            (args.output / "environment.txt").write_text("\n".join(packages) + "\n")
        write_json(args.output / "metrics.json", metrics)
        write_json(args.output / "run.json", {
            "status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_info, "performance": performance, "environment": environment,
            "run_seconds_including_load_and_outputs": time.perf_counter() - start,
        })
        print(json.dumps(metrics, indent=2))
        print(f"Results saved to {args.output}")
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main():
    args = parser().parse_args()
    try:
        if args.command == "download":
            manifest = prepare_data(args.data_dir, args.archive, args.reference)
            print(json.dumps(manifest, indent=2))
            return
        manifest = verify_manifest(args.data_dir)
        metadata = read_reference(args.data_dir / "DMS_substitutions.csv")
        rows, variants, summary = read_assay(args.data_dir / f"{ASSAY_ID}.csv", metadata)
        if args.command == "inspect":
            print(json.dumps({"metadata": metadata, "inspection": summary}, indent=2))
        else:
            run(args, metadata, rows, variants, summary, manifest)
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
