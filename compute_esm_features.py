"""Compute frozen ESM-2 embeddings and zero-shot scores for TEM-1 single mutants.

Inference only: the model is loaded in eval mode and never updated. No model is
trained and the fold assignments in ``stiffler_single_folds.csv`` are read and
copied verbatim.

Pipeline
--------
0. Reconstruct the DMS reference sequence by reverting the mutation in the raw
   ProteinGym ``mutated_sequence`` column, and verify it against the folds table.
1. One unmasked forward pass of the wild type -> ``esm_wt.npz`` (fp16 logits and
   per-residue hidden states), then a zero-shot score per variant:
   log P(mut_aa) - log P(wt_aa) from the WT logits at the mutated position.
2. One unmasked forward pass per mutated sequence (batched, padded), keeping only
   five fp16 vectors per variant -> ``esm_variant_features.parquet``.
3. Sanity checks on the first variant and on the completed run.

Run from the repository root:

    python compute_esm_features.py

The mutant loop is resumable: variants already present in the parquet file are
skipped.

Zero-shot convention
--------------------
The WT forward pass is unmasked, so the score is the model's preference for the
mutant residue over the wild-type residue in the *wild-type* context. This is
cheap (one forward pass) but is not the masked-marginal score used elsewhere in
this repository (``src/tem1/scoring.py``), which masks the site before scoring.
"""

from __future__ import annotations

import argparse
import io
import json
import hashlib
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmForMaskedLM, EsmTokenizer

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")

DEFAULT_CHECKPOINT = "facebook/esm2_t30_150M_UR50D"
DEFAULT_REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
DEFAULT_FOLDS = "stiffler_single_folds.csv"
DEFAULT_ZIP = "proteingym_substitutions.zip"
DEFAULT_RAW_MEMBER = "DMS_ProteinGym_substitutions/BLAT_ECOLX_Stiffler_2015.csv"
DEFAULT_WT_OUT = "esm_wt.npz"
DEFAULT_VARIANTS_OUT = "esm_variant_features.parquet"

WINDOWS = (5, 10, 20)
FEATURE_COLUMNS = ["emb_mut_pos", "mean_pooled", "mean_local_5", "mean_local_10", "mean_local_20"]
BASE_COLUMNS = ["mutant", "position", "fold", "zeroshot"]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


# --------------------------------------------------------------------------- #
# Step 0 - reference sequence
# --------------------------------------------------------------------------- #
def read_raw_assay(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.is_file():
        fail(f"missing {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            fail(f"{member} not found in {zip_path}")
        with archive.open(member) as handle:
            raw = pd.read_csv(io.BytesIO(handle.read()))
    required = {"mutant", "mutated_sequence", "DMS_score"}
    if not required.issubset(raw.columns):
        fail(f"raw assay lacks columns {sorted(required - set(raw.columns))}")
    return raw


def revert_mutation(mutant: str, mutated_sequence: str) -> str:
    """Return the reference sequence implied by one single-mutant row."""
    match = MUTATION.fullmatch(mutant)
    if match is None:
        fail(f"not a canonical single substitution: {mutant!r}")
    wt_aa, position, mut_aa = match.group(1), int(match.group(2)), match.group(3)
    if position > len(mutated_sequence):
        fail(f"{mutant}: position {position} exceeds sequence length {len(mutated_sequence)}")
    if mutated_sequence[position - 1] != mut_aa:
        fail(
            f"{mutant}: mutated_sequence has {mutated_sequence[position - 1]!r} at {position}, "
            f"expected {mut_aa!r}"
        )
    return mutated_sequence[: position - 1] + wt_aa + mutated_sequence[position:]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_folds(raw: pd.DataFrame, folds: pd.DataFrame) -> None:
    """Join by variant identity, never assume input row order matches."""
    if raw.empty or folds.empty or raw['mutant'].duplicated().any() or folds['mutant'].duplicated().any():
        fail('empty table or duplicate variants')
    if set(raw['mutant']) != set(folds['mutant']):
        fail('folds variant set differs from raw assay')
    for column in ('position', 'fold'):
        values = pd.to_numeric(folds[column], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            fail(f'{column} must contain finite integers')
        folds[column] = values.astype(np.int64)
    if set(folds['fold']) != set(range(5)):
        fail('expected five folds numbered 0..4')
    if not folds.groupby('position')['fold'].nunique().eq(1).all():
        fail('a position belongs to more than one fold')
    for row in folds.itertuples():
        match = MUTATION.fullmatch(str(row.mutant))
        if match is None or (match[1], int(match[2]), match[3]) != (row.wt_aa, row.position, row.mut_aa):
            fail(f'{row.mutant}: annotation disagrees with fold columns')
        if row.wt_aa == row.mut_aa:
            fail(f'{row.mutant}: synonymous substitution')
    observed = pd.to_numeric(folds['DMS_score'], errors='coerce').to_numpy(dtype=float)
    expected = pd.to_numeric(raw.set_index('mutant').loc[folds['mutant'], 'DMS_score'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(expected).all() or not np.allclose(observed, expected, atol=1e-12, rtol=0):
        fail('DMS scores differ from raw assay or are non-finite')


def validate_completed(completed: pd.DataFrame, folds: pd.DataFrame, hidden_size: int) -> None:
    if completed.empty:
        return
    if completed['mutant'].duplicated().any() or not set(completed['mutant']).issubset(set(folds['mutant'])):
        fail('completed variants are duplicated or absent from folds')
    expected = folds.set_index('mutant').loc[completed['mutant']]
    for column in ('position', 'fold'):
        if not np.array_equal(completed[column].to_numpy(), expected[column].to_numpy()):
            fail(f'completed {column} differs from folds')
    if not np.isfinite(completed['zeroshot'].to_numpy(dtype=float)).all():
        fail('non-finite completed zeroshot')
    for column in FEATURE_COLUMNS:
        for vector in completed[column]:
            array = np.asarray(vector)
            if array.shape != (hidden_size,) or array.dtype != np.float16 or not np.isfinite(array).all():
                fail(f'{column}: expected finite float16 vector of length {hidden_size}')


def reconstruct_reference(raw: pd.DataFrame, folds: pd.DataFrame, n_random: int, seed: int) -> str:
    n_random = min(n_random, len(raw))
    sampled = raw.sample(n=n_random, random_state=seed)
    reconstructions = {revert_mutation(row.mutant, row.mutated_sequence) for row in sampled.itertuples()}
    if len(reconstructions) != 1:
        fail(f"{n_random} random rows produced {len(reconstructions)} distinct reference sequences")
    wt_seq = reconstructions.pop()

    # Every raw row must agree, not just the sampled ones.
    for row in raw.itertuples():
        if revert_mutation(row.mutant, row.mutated_sequence) != wt_seq:
            fail(f"{row.mutant}: reverted sequence disagrees with the reconstructed reference")

    if set(wt_seq) - set(AMINO_ACIDS):
        fail(f"reference contains non-canonical residues: {sorted(set(wt_seq) - set(AMINO_ACIDS))}")

    positions = folds["position"].to_numpy()
    if positions.min() < 1 or positions.max() > len(wt_seq):
        fail(f"folds positions {positions.min()}..{positions.max()} outside reference 1..{len(wt_seq)}")
    observed = np.array([wt_seq[p - 1] for p in positions])
    mismatched = folds.loc[observed != folds["wt_aa"].to_numpy(), "mutant"].tolist()
    if mismatched:
        fail(f"{len(mismatched)} rows have wt_seq[position-1] != wt_aa, e.g. {mismatched[:5]}")
    print(
        f"[step 0] reference length {len(wt_seq)}; {n_random} random rows and all "
        f"{len(raw)} raw rows revert to the same sequence; "
        f"{len(folds)} folds rows match wt_seq[position-1] == wt_aa"
    )
    return wt_seq


# --------------------------------------------------------------------------- #
# Model and tokenizer
# --------------------------------------------------------------------------- #
def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        fail("CUDA requested but unavailable; use --device cpu or --device auto")
    return torch.device(requested)


def load_model(checkpoint: str, device: torch.device, revision: str, local_only: bool = False):
    tokenizer = EsmTokenizer.from_pretrained(checkpoint, revision=revision, local_files_only=local_only)
    model = EsmForMaskedLM.from_pretrained(checkpoint, revision=revision,
                                         local_files_only=local_only, use_safetensors=True)
    if model.config.model_type != "esm":
        fail(f"{checkpoint} is not an ESM masked language model")
    model = model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def amino_acid_token_ids(tokenizer) -> dict[str, int]:
    """Map the 20 canonical residues to their single-residue token ids."""
    ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS}
    special = set(tokenizer.all_special_ids)
    unknown = tokenizer.unk_token_id
    for aa, token_id in ids.items():
        if token_id is None or token_id < 0 or token_id == unknown or token_id in special:
            fail(f"residue {aa!r} does not map to a valid non-special token (got {token_id})")
    if len(set(ids.values())) != 20:
        fail("the 20 canonical residues do not map to 20 distinct token ids")
    return ids


def residue_slice(tokenizer, sequence: str) -> slice:
    """Return the slice of the tokenized sequence that holds the residues."""
    encoded = tokenizer(sequence, return_special_tokens_mask=True, truncation=False)
    special = np.asarray(encoded["special_tokens_mask"], dtype=bool)
    residue_index = np.flatnonzero(~special)
    if len(residue_index) != len(sequence):
        fail(f"tokenizer produced {len(residue_index)} residues for a {len(sequence)}-residue sequence")
    if not np.array_equal(residue_index, np.arange(residue_index[0], residue_index[0] + len(sequence))):
        fail("residue tokens are not contiguous in the tokenized sequence")
    expected = tokenizer.convert_tokens_to_ids(list(sequence))
    actual = np.asarray(encoded["input_ids"])[residue_index].tolist()
    if actual != expected:
        fail("tokenized residues do not match the input sequence")
    return slice(int(residue_index[0]), int(residue_index[-1]) + 1)


def window_means(hidden: np.ndarray, center: int, windows=WINDOWS) -> dict[str, np.ndarray]:
    """Mean of residues within +/-w of ``center`` (0-indexed), clipped to bounds."""
    length = hidden.shape[0]
    means = {}
    for w in windows:
        lo, hi = max(0, center - w), min(length, center + w + 1)
        means[f"mean_local_{w}"] = hidden[lo:hi].astype(np.float32).mean(axis=0)
    return means


# --------------------------------------------------------------------------- #
# Step 1 - wild-type forward pass
# --------------------------------------------------------------------------- #
def compute_wild_type(model, tokenizer, wt_seq: str, device: torch.device, checkpoint: str,
                      out_path: Path) -> dict:
    import transformers

    sl = residue_slice(tokenizer, wt_seq)
    encoded = tokenizer(wt_seq, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, sl, :].float().cpu().numpy()
    logits = outputs.logits[0, sl, :].float().cpu().numpy()
    if hidden.shape != (len(wt_seq), model.config.hidden_size):
        fail(f"unexpected WT hidden shape {hidden.shape}")
    if logits.shape != (len(wt_seq), model.config.vocab_size):
        fail(f"unexpected WT logits shape {logits.shape}")

    np.savez_compressed(
        out_path,
        wt_logits=logits.astype(np.float16),
        wt_hidden=hidden.astype(np.float16),
        checkpoint_name=np.array(checkpoint),
        transformers_version=np.array(transformers.__version__),
    )
    print(
        f"[step 1] wild type forward pass: logits {logits.shape}, hidden {hidden.shape}; "
        f"saved {out_path} ({out_path.stat().st_size / 2**20:.2f} MiB)"
    )
    return {"hidden": hidden, "logits": logits}


def zero_shot_scores(wt_logits: np.ndarray, folds: pd.DataFrame, aa_ids: dict[str, int]) -> np.ndarray:
    """log P(mut_aa) - log P(wt_aa) at the mutated position, from unmasked WT logits."""
    log_probs = wt_logits.astype(np.float64)
    log_probs -= log_probs.max(axis=1, keepdims=True)
    log_probs -= np.log(np.exp(log_probs).sum(axis=1, keepdims=True))

    positions = folds["position"].to_numpy()
    for column in ("wt_aa", "mut_aa"):
        invalid = sorted(set(folds[column]) - set(AMINO_ACIDS))
        if invalid:
            fail(f"column {column} contains non-canonical residues: {invalid}")

    wt_ids = np.array([aa_ids[aa] for aa in folds["wt_aa"]])
    mut_ids = np.array([aa_ids[aa] for aa in folds["mut_aa"]])
    scores = log_probs[positions - 1, mut_ids] - log_probs[positions - 1, wt_ids]
    if not np.all(np.isfinite(scores)):
        fail("non-finite zero-shot scores")
    return scores


# --------------------------------------------------------------------------- #
# Step 2 - mutated sequences
# --------------------------------------------------------------------------- #
def load_completed(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=BASE_COLUMNS + FEATURE_COLUMNS)
    completed = pd.read_parquet(path)
    missing = set(BASE_COLUMNS + FEATURE_COLUMNS) - set(completed.columns)
    if missing:
        fail(f"{path} exists but lacks columns {sorted(missing)}; remove it to recompute")
    return completed


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame[BASE_COLUMNS + FEATURE_COLUMNS].copy()
    ordered["mutant"] = ordered["mutant"].astype(str)
    ordered["position"] = ordered["position"].astype("int64")
    ordered["fold"] = ordered["fold"].astype("int64")
    ordered["zeroshot"] = ordered["zeroshot"].astype("float64")
    for column in FEATURE_COLUMNS:
        ordered[column] = ordered[column].apply(lambda v: np.asarray(v, dtype=np.float16))
    tmp = path.with_suffix(path.suffix + ".tmp")
    ordered.to_parquet(tmp, index=False)
    tmp.replace(path)


def compute_variants(model, tokenizer, wt_seq: str, folds: pd.DataFrame, zeroshot: np.ndarray,
                     device: torch.device, batch_size: int, out_path: Path,
                     completed: pd.DataFrame) -> pd.DataFrame:
    done = set(completed["mutant"]) if len(completed) else set()
    # zeroshot is aligned to folds row order; keep the original row index explicitly.
    folds = folds.reset_index(drop=True)
    zeroshot_by_row = pd.Series(np.asarray(zeroshot, dtype=np.float64), index=folds.index)
    pending = folds[~folds["mutant"].isin(done)].reset_index(drop=True)
    print(
        f"[step 2] {len(done)} variants already in {out_path}, "
        f"{len(pending)} to compute in batches of {batch_size} on {device}"
    )
    if pending.empty:
        return completed

    sl = residue_slice(tokenizer, wt_seq)
    rows: list[dict] = []
    started = time.perf_counter()
    progress = tqdm(total=len(pending), desc="ESM-2 variants", unit="variant", dynamic_ncols=True)

    # Row index into the folds table, carried alongside the pending rows.
    pending_fold_rows = folds.index[folds["mutant"].isin(pending["mutant"])].to_numpy()

    for offset in range(0, len(pending), batch_size):
        batch = pending.iloc[offset : offset + batch_size]
        fold_rows = pending_fold_rows[offset : offset + len(batch)]
        sequences = []
        for row in batch.itertuples():
            index = row.position - 1
            sequence = wt_seq[:index] + row.mut_aa + wt_seq[index + 1 :]
            if sequence == wt_seq:
                fail(f"{row.mutant}: mutated sequence equals the wild type")
            sequences.append(sequence)

        encoded = tokenizer(sequences, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, sl, :].float().cpu().numpy()

        for i, row in enumerate(batch.itertuples()):
            index = row.position - 1
            length = len(sequences[i])
            residues = hidden[i, :length]
            features = {
                "emb_mut_pos": residues[index],
                "mean_pooled": residues.mean(axis=0),
                **window_means(residues, index),
            }
            rows.append(
                {
                    "mutant": row.mutant,
                    "position": int(row.position),
                    "fold": int(row.fold),
                    "zeroshot": float(zeroshot_by_row[fold_rows[i]]),
                    **{key: value.astype(np.float16) for key, value in features.items()},
                }
            )
        progress.update(len(batch))

        # Checkpoint after every batch so an interrupted run resumes cheaply.
        write_parquet(pd.concat([completed, pd.DataFrame(rows)], ignore_index=True), out_path)

    progress.close()
    elapsed = time.perf_counter() - started
    print(f"[step 2] {len(rows)} variants in {elapsed:.1f}s ({len(rows) / elapsed:.1f} variants/s)")
    return pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------------------------------- #
# Step 3 - sanity checks
# --------------------------------------------------------------------------- #
def check_first_variant(model, tokenizer, wt_seq: str, row, wt_hidden: np.ndarray,
                        wt_logits: np.ndarray, aa_ids: dict[str, int],
                        device: torch.device) -> dict:
    index = row.position - 1
    sequence = wt_seq[:index] + row.mut_aa + wt_seq[index + 1 :]
    differences = [i for i, (a, b) in enumerate(zip(wt_seq, sequence)) if a != b]
    assert differences == [index], f"{row.mutant}: mutated sequence differs at {differences}, expected [{index}]"

    sl = residue_slice(tokenizer, wt_seq)
    encoded = tokenizer(sequence, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, sl, :].float().cpu().numpy()
    logits = outputs.logits[0, sl, :].float().cpu().numpy()

    wt_vector = wt_hidden[index]
    emb_mut_pos = hidden[index]
    delta = emb_mut_pos - wt_vector
    l2 = lambda v: float(np.linalg.norm(v))
    assert not np.allclose(emb_mut_pos, wt_vector), f"{row.mutant}: mutant and WT hidden states are identical"

    wt_mean = wt_hidden.mean(axis=0)
    mut_mean = hidden.mean(axis=0)
    pooled_delta = l2(mut_mean - wt_mean)

    log_probs = logits[index].astype(np.float64)
    log_probs -= log_probs.max()
    log_probs -= np.log(np.exp(log_probs).sum())
    score = float(log_probs[aa_ids[row.mut_aa]] - log_probs[aa_ids[row.wt_aa]])
    wt_log_probs = wt_logits[index].astype(np.float64)
    wt_log_probs -= wt_log_probs.max()
    wt_log_probs -= np.log(np.exp(wt_log_probs).sum())
    wt_score = float(wt_log_probs[aa_ids[row.mut_aa]] - wt_log_probs[aa_ids[row.wt_aa]])

    print(
        f"[step 3] {row.mutant} at {row.position}: sequence differs only at index {index}\n"
        f"         ||wt_hidden[pos]||      = {l2(wt_vector):.4f}\n"
        f"         ||emb_mut_pos||         = {l2(emb_mut_pos):.4f}\n"
        f"         ||emb_mut_pos - wt||    = {l2(delta):.4f}\n"
        f"         ||mean_pooled(mut)-wt|| = {pooled_delta:.4f}\n"
        f"         zeroshot from WT logits = {wt_score:+.4f} "
        f"(mutant-sequence logits give {score:+.4f}; not used, informational)"
    )
    return {"delta_l2": l2(delta), "pooled_delta_l2": pooled_delta}


def report(folds: pd.DataFrame, results: pd.DataFrame, wt_path: Path, variants_path: Path,
           checkpoint: str, tokenizer) -> None:
    import transformers

    if len(results) != len(folds):
        fail(f"processed {len(results)} variants but the folds table has {len(folds)} rows")
    if results["mutant"].duplicated().any():
        fail("duplicate variants in the output")
    if set(results["mutant"]) != set(folds["mutant"]):
        fail("output variants do not match the folds table")

    merged = folds.merge(results, on="mutant", suffixes=("", "_out"), validate="one_to_one")
    for column in ("position", "fold"):
        if not (merged[column] == merged[f"{column}_out"]).all():
            fail(f"{column} changed between the folds table and the output")

    # Recompute zero-shot independently from the saved logits and require agreement.
    # esm_wt.npz stores fp16 logits (~1e-3 relative), so the tolerance covers that
    # rounding; a misaligned column is off by orders of magnitude more.
    aa_ids = amino_acid_token_ids(tokenizer)
    expected = zero_shot_scores(np.load(wt_path)["wt_logits"], merged.reset_index(drop=True), aa_ids)
    stored_zeroshot = merged["zeroshot"].to_numpy(dtype=np.float64)
    zeroshot_deviation = float(np.max(np.abs(expected - stored_zeroshot)))
    if zeroshot_deviation > 1e-2:
        worst = int(np.argmax(np.abs(expected - stored_zeroshot)))
        fail(
            "stored zeroshot disagrees with a recomputation from esm_wt.npz "
            f"(worst row {merged['mutant'].iloc[worst]}: "
            f"{stored_zeroshot[worst]:.6f} vs {expected[worst]:.6f})"
        )
    if merged["zeroshot"].nunique() < 0.9 * len(merged):
        fail("zeroshot has too few distinct values; the column is probably misaligned")

    validate_completed(results, folds, np.load(wt_path)["wt_hidden"].shape[1])
    features = {column: np.stack(merged[column].to_numpy()) for column in FEATURE_COLUMNS}
    for column, matrix in features.items():
        if matrix.dtype != np.float16:
            fail(f"{column} is {matrix.dtype}, expected float16")
        if not np.all(np.isfinite(matrix.astype(np.float32))):
            fail(f"{column} contains non-finite values")

    wt = np.load(wt_path)
    wt_hidden = wt["wt_hidden"].astype(np.float32)
    wt_means = wt_hidden.mean(axis=0)
    delta_norms = np.linalg.norm(
        features["mean_pooled"].astype(np.float32) - wt_means, axis=1
    )
    pos_delta = np.linalg.norm(
        features["emb_mut_pos"].astype(np.float32)
        - wt_hidden[merged["position"].to_numpy() - 1],
        axis=1,
    )
    zeroshot = merged["zeroshot"].to_numpy()

    def describe(name: str, values: np.ndarray) -> str:
        quantiles = np.percentile(values, [0, 1, 25, 50, 75, 99, 100])
        return (
            f"  {name:<28} n={len(values)} mean={values.mean():+.4f} sd={values.std():.4f} "
            f"min={quantiles[0]:+.4f} p1={quantiles[1]:+.4f} p25={quantiles[2]:+.4f} "
            f"median={quantiles[3]:+.4f} p75={quantiles[4]:+.4f} p99={quantiles[5]:+.4f} "
            f"max={quantiles[6]:+.4f}"
        )

    print("\n[step 3] post-run checks")
    print(f"  variants processed {len(results)} / rows in folds table {len(folds)}: match")
    print(f"  folds preserved, positions preserved, no duplicates, all vectors fp16 and finite")
    print(f"  zeroshot recomputed from esm_wt.npz matches the stored column "
          f"(max |diff| {zeroshot_deviation:.2e}, fp16 storage of the logits)")
    print(f"  checkpoint {wt['checkpoint_name']} (transformers {wt['transformers_version']})")
    print(f"  wt_logits {wt['wt_logits'].shape} fp16, wt_hidden {wt['wt_hidden'].shape} fp16")
    print("  distributions")
    print(describe("zeroshot", zeroshot))
    print(describe("||emb_mut_pos - wt_hidden||", pos_delta))
    print(describe("||mean_pooled - wt_mean||", delta_norms))
    print(
        f"  mean_pooled shift is {delta_norms.mean() / np.linalg.norm(wt_means):.2%} of "
        f"||wt_mean_pooled|| = {np.linalg.norm(wt_means):.4f}"
    )
    print("  file sizes")
    for path in (wt_path, variants_path):
        print(f"    {path}: {path.stat().st_size / 2**20:.2f} MiB")
    print(
        f"\n  checkpoint: {checkpoint}\n"
        f"  transformers: {transformers.__version__}\n"
        f"  torch: {torch.__version__}\n"
        f"  numpy: {np.__version__}  pandas: {pd.__version__}  pyarrow: {_pyarrow_version()}"
    )


def _pyarrow_version() -> str:
    import pyarrow

    return pyarrow.__version__


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=Path, default=Path(DEFAULT_FOLDS))
    parser.add_argument("--zip", type=Path, default=Path(DEFAULT_ZIP))
    parser.add_argument("--raw-member", default=DEFAULT_RAW_MEMBER)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", help="Immutable HF commit; default pins ESM-2 150M")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Check existing artifacts without inference or writes")
    parser.add_argument("--wt-out", type=Path, default=Path(DEFAULT_WT_OUT))
    parser.add_argument("--variants-out", type=Path, default=Path(DEFAULT_VARIANTS_OUT))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-random", type=int, default=100,
                        help="rows sampled to verify the reconstructed reference (>=100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip the single-variant sanity check (step 3)")
    args = parser.parse_args(argv)

    if args.n_random < 100:
        fail("--n-random must be at least 100")
    if args.batch_size < 1:
        fail("--batch-size must be positive")
    for path in (args.folds,):
        if not path.is_file():
            fail(f"missing {path}")

    folds = pd.read_csv(args.folds)
    required = {"mutant", "wt_aa", "position", "mut_aa", "DMS_score", "fold"}
    if not required.issubset(folds.columns):
        fail(f"{args.folds} lacks columns {sorted(required - set(folds.columns))}")
    if folds["mutant"].duplicated().any():
        fail(f"{args.folds} contains duplicate mutants")
    print(f"[input] {args.folds}: {len(folds)} variants, {folds['position'].nunique()} positions, "
          f"folds {sorted(folds['fold'].unique().tolist())} (copied unchanged)")

    raw = read_raw_assay(args.zip, args.raw_member)
    validate_folds(raw, folds)
    wt_seq = reconstruct_reference(raw, folds, args.n_random, args.seed)
    revision = args.revision or (DEFAULT_REVISION if args.checkpoint == DEFAULT_CHECKPOINT else None)
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        fail("supply an immutable --revision for the chosen checkpoint")
    if args.validate_only:
        tokenizer = EsmTokenizer.from_pretrained(args.checkpoint, revision=revision,
                                                 local_files_only=args.local_files_only)
        with np.load(args.wt_out) as saved:
            if str(saved['checkpoint_name']) != args.checkpoint:
                fail("WT checkpoint differs from requested model")
            if saved['wt_hidden'].shape[0] != len(wt_seq) or saved['wt_logits'].shape[0] != len(wt_seq):
                fail("WT length differs from reference")
            if not all(np.isfinite(saved[k]).all() for k in ('wt_hidden','wt_logits')):
                fail("non-finite WT arrays")
        report(folds, load_completed(args.variants_out), args.wt_out, args.variants_out,
               args.checkpoint, tokenizer)
        return 0
    # Never mix partial results from different inputs/checkpoints or overwrite a legacy WT.
    manifest_path = args.variants_out.with_suffix(args.variants_out.suffix + '.run.json')
    signature = {'checkpoint': args.checkpoint, 'revision': revision,
                 'folds_sha256': file_hash(args.folds), 'archive_sha256': file_hash(args.zip),
                 'raw_member': args.raw_member, 'source_sha256': file_hash(Path(__file__)),
                 'wt_out': str(args.wt_out.resolve()), 'precision': 'float32', 'tf32': False,
                 'torch': torch.__version__}
    if args.variants_out.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != signature:
            fail("resume provenance missing or changed; validate legacy artifacts with --validate-only or choose fresh output paths")
        if not args.wt_out.exists():
            fail("resume requires the matching WT artifact")
    elif args.wt_out.exists():
        fail("WT output already exists; choose fresh output paths")
    completed = load_completed(args.variants_out)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = select_device(args.device)
    print(f"[model] loading {args.checkpoint} on {device}")
    model, tokenizer = load_model(args.checkpoint, device, revision, args.local_files_only)
    if len(wt_seq) + 2 > model.config.max_position_embeddings:
        fail("reference exceeds model context; truncation forbidden")
    validate_completed(completed, folds, model.config.hidden_size)
    for path in (args.wt_out, args.variants_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(signature, indent=2) + "\n")
    aa_ids = amino_acid_token_ids(tokenizer)
    print(f"[model] hidden size {model.config.hidden_size}, vocab {model.config.vocab_size}, "
          f"parameters {sum(p.numel() for p in model.parameters()):,}")

    wt = compute_wild_type(model, tokenizer, wt_seq, device, args.checkpoint, args.wt_out)
    zeroshot = zero_shot_scores(wt["logits"], folds, aa_ids)
    print(f"[step 1] zero-shot scores for {len(folds)} variants "
          f"(mean {zeroshot.mean():+.4f}, sd {zeroshot.std():.4f})")

    if not args.skip_sanity:
        check_first_variant(model, tokenizer, wt_seq, folds.iloc[0], wt["hidden"], wt["logits"],
                            aa_ids, device)

    if len(completed):
        expected = pd.Series(zeroshot, index=folds['mutant']).loc[completed['mutant']].to_numpy()
        if not np.allclose(expected, completed['zeroshot'], atol=1e-4, rtol=0):
            fail("resume zeroshot differs from current WT scores")
    results = compute_variants(model, tokenizer, wt_seq, folds, zeroshot, device,
                               args.batch_size, args.variants_out, completed)
    # A resumed run may append missing rows after completed ones; restore folds order.
    results = results.set_index('mutant').loc[folds['mutant']].reset_index()
    write_parquet(results, args.variants_out)
    report(folds, results, args.wt_out, args.variants_out, args.checkpoint, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
