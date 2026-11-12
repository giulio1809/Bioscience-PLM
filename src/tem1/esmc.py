"""Pinned official ESM-C backend; direct FP32 forward, never SDK autocast."""
import importlib.metadata
import inspect
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

from .data import sha256

MODEL_ID = 'biohub/esmc-600m-2024-12'
ORIGINAL_MODEL_ID = 'EvolutionaryScale/esmc-600m-2024-12'
MODEL_REVISION = 'e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c'
WEIGHTS_FILE = 'data/weights/esmc_600m_2024_12_v0.pth'
WEIGHTS_SHA256 = '8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324'
SDK_VERSION = '3.2.1'


class ESMCAdapter(torch.nn.Module):
    """Expose logits to the shared scorer; only unpadded, single-chain batches."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Match ProteinGym's window policy; no implicit truncation/windowing.
        self.config = SimpleNamespace(max_position_embeddings=1024)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or (input_ids == self.model.tokenizer.pad_token_id).any():
            raise ValueError('ESM-C adapter requires unpadded batches')
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not attention_mask.bool().all()):
            raise ValueError('ESM-C adapter requires an all-valid attention mask')
        if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
            raise ValueError('ESM-C benchmark forbids autocast')
        if next(self.model.parameters()).dtype != torch.float32:
            raise ValueError('ESM-C benchmark requires FP32 parameters')
        output = self.model(sequence_tokens=input_ids)
        if output.sequence_logits.dtype != torch.float32:
            raise ValueError('ESM-C returned non-FP32 logits')
        return SimpleNamespace(logits=output.sequence_logits)


def load_esmc(model_id, revision, device, local_only=False):
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    if model_id not in (MODEL_ID, ORIGINAL_MODEL_ID) or revision != MODEL_REVISION:
        raise ValueError('ESM-C loader requires the verified model and immutable revision')
    if importlib.metadata.version('esm') != SDK_VERSION:
        raise ValueError(f'Install esm=={SDK_VERSION} with requirements-esmc.txt')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Expected immutable HF commit')
    start = time.perf_counter()
    weights = Path(hf_hub_download(MODEL_ID, WEIGHTS_FILE, revision=revision, local_files_only=local_only))
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Official checkpoint checksum mismatch')
    tokenizer = get_esmc_model_tokenizers()
    model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=tokenizer, use_flash_attn=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = ESMCAdapter(model.float().to(device).eval()).eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    sdk_root = Path(inspect.getfile(ESMC)).parents[1]
    sdk_sources = {str(p.relative_to(sdk_root)): sha256(p)
                   for folder in ('models', 'layers', 'tokenization')
                   for p in (sdk_root / folder).rglob('*.py')}
    return model, tokenizer, {
        'family': 'esmc', 'model_id': MODEL_ID, 'original_model_id': ORIGINAL_MODEL_ID,
        'requested_revision': revision, 'resolved_revision': revision,
        'weights_file': WEIGHTS_FILE, 'weights_sha256': WEIGHTS_SHA256,
        'sdk_version': SDK_VERSION, 'sdk_source_sha256': sdk_sources,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'model_load_seconds_including_download': time.perf_counter() - start,
        'precision': 'float32', 'autocast': False, 'flash_attention': False,
        'scoring_method': 'masked_marginals', 'context_limit_tokens': 1024,
        'tokenizer_class': type(tokenizer).__name__, 'tokenizer_vocab': tokenizer.get_vocab(),
        'special_token_ids': {k: getattr(tokenizer, k) for k in ('cls_token_id','eos_token_id','pad_token_id','mask_token_id','unk_token_id')},
    }
