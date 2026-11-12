"""ESM-2 masked marginals: one masked wild-type context per observed position."""

import math
import time

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .data import AMINO_ACIDS, Substitution

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a CUDA-enabled PyTorch or use --device cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(model_id: str, revision: str, device: torch.device, local_only: bool = False):
    """Load the original masked-language-model head, with no fine-tuning."""
    if model_id in ('biohub/esmc-600m-2024-12', 'EvolutionaryScale/esmc-600m-2024-12'):
        from .esmc import load_esmc
        return load_esmc(model_id, revision, device, local_only)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_only, trust_remote_code=False,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, revision=revision, local_files_only=local_only,
        trust_remote_code=False, use_safetensors=True,
    )
    if model.config.model_type != "esm":
        raise ValueError("This implementation expects an ESM masked language model")
    model = model.float().to(device).eval()
    synchronize(device)
    return model, tokenizer, {
        "model_id": model_id, "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_load_seconds_including_download": time.perf_counter() - start,
        "precision": "float32", "scoring_method": "masked_marginals",
    }


def encode_reference(tokenizer, sequence: str):
    """Locate residues from the special-token mask instead of assuming a BOS offset."""
    encoded = tokenizer(
        sequence, return_tensors="pt", return_special_tokens_mask=True,
        add_special_tokens=True, truncation=False,
    )
    special = encoded.pop("special_tokens_mask")[0]
    residue_tokens = torch.where(special == 0)[0]
    expected = tokenizer.convert_tokens_to_ids(list(sequence))
    actual = encoded["input_ids"][0, residue_tokens].tolist()
    if len(residue_tokens) != len(sequence) or actual != expected:
        raise ValueError("Tokenizer does not map each reference residue to exactly one expected token")
    aa_ids = tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS))
    if len(set(aa_ids)) != 20 or tokenizer.unk_token_id in aa_ids or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer lacks the required amino-acid or mask vocabulary")
    return encoded, residue_tokens, dict(zip(AMINO_ACIDS, aa_ids))


def score_variants(model, tokenizer, sequence: str, variants: list[Substitution],
                   batch_size: int = 4, warmup_batches: int = 1, progress: bool = True):
    """Return scores in input order and synchronized timing/CUDA allocator statistics.

    Score = log P(alternate | reference with site masked)
          - log P(wild type | reference with site masked).
    Each batch row masks exactly one site. Batch size counts positions, not variants.
    """
    if batch_size < 1 or warmup_batches < 0 or not variants:
        raise ValueError("Require variants, positive batch_size, and nonnegative warmup_batches")
    for variant in variants:
        if not 1 <= variant.position <= len(sequence) or sequence[variant.position - 1] != variant.wild_type:
            raise ValueError("Variant does not match reference")
        if variant.alternate not in AMINO_ACIDS or variant.alternate == variant.wild_type:
            raise ValueError("Expected a canonical nonsynonymous substitution")
    device = next(model.parameters()).device
    model.eval()
    encoded, residue_tokens, amino_ids = encode_reference(tokenizer, sequence)
    if encoded["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("Reference exceeds model context limit; truncation is forbidden")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    positions = sorted({v.position for v in variants})

    def forward(batch_positions):
        columns = residue_tokens[[p - 1 for p in batch_positions]].to(device)
        rows = torch.arange(len(batch_positions), device=device)
        inputs = {key: value.repeat(len(batch_positions), 1) for key, value in encoded.items()}
        inputs["input_ids"][rows, columns] = tokenizer.mask_token_id
        logits = model(**inputs).logits[rows, columns, :].float()
        # Compute in float32 over the full vocabulary, as in the reference method.
        return torch.log_softmax(logits, dim=-1)

    with torch.inference_mode():
        warmup_start = time.perf_counter()
        for _ in range(warmup_batches):
            forward(positions[:batch_size])
        synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        marginals = {}
        for offset in range(0, len(positions), batch_size):
            batch = positions[offset:offset + batch_size]
            log_probs = forward(batch).cpu()
            marginals.update(zip(batch, log_probs))
            if progress:
                print(f"Scored {offset + len(batch)}/{len(positions)} positions", flush=True)
        synchronize(device)
        seconds = time.perf_counter() - start
    scores = [
        float(marginals[v.position][amino_ids[v.alternate]] - marginals[v.position][amino_ids[v.wild_type]])
        for v in variants
    ]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Model produced non-finite scores")
    performance = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size_positions": batch_size, "n_masked_positions": len(positions),
        "n_forward_batches": math.ceil(len(positions) / batch_size),
        "warmup_batches": warmup_batches, "warmup_seconds": warmup_seconds,
        "inference_seconds": seconds, "positions_per_second": len(positions) / seconds,
        "timing_scope": "masked batch construction, forward passes, log_softmax, CPU transfers and progress; excludes load, warmup, evaluation",
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "cuda_memory_scope": "PyTorch allocator peaks during scoring, including resident model; not total device usage",
    }
    return scores, performance
