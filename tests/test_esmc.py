"""Offline ESM-C checks with the official tokenizer and a tiny random model."""
import importlib.util
import unittest
from unittest.mock import patch

HAS_ESMC = importlib.util.find_spec('esm') is not None


@unittest.skipUnless(HAS_ESMC, 'Install the separate ESM-C environment')
class ESMCTests(unittest.TestCase):
    def setUp(self):
        import torch
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
        from tem1.esmc import ESMCAdapter
        self.torch = torch
        torch.manual_seed(7)
        self.tokenizer = get_esmc_model_tokenizers()
        self.raw = ESMC(d_model=32, n_heads=2, n_layers=2,
                        tokenizer=self.tokenizer, use_flash_attn=False).float().eval()
        self.model = ESMCAdapter(self.raw).eval()

    def test_tokenizer_alignment_and_special_tokens(self):
        from tem1.scoring import encode_reference
        from tem1.data import AMINO_ACIDS
        encoded, positions, aa = encode_reference(self.tokenizer, AMINO_ACIDS)
        self.assertEqual(positions.tolist(), list(range(1, 21)))
        self.assertEqual(encoded['input_ids'][0, 0], 0)
        self.assertEqual(encoded['input_ids'][0, -1], 2)
        self.assertEqual(self.tokenizer.mask_token_id, 32)
        self.assertEqual(len(set(aa.values())), 20)
        self.assertEqual(self.tokenizer.decode(list(aa.values())).replace(' ', ''), AMINO_ACIDS)

    def test_batching_matches_direct_single_site_logits(self):
        from tem1.scoring import score_variants
        from tem1.data import parse_mutation
        torch = self.torch
        sequence = 'ACD'
        variants = [parse_mutation(v, sequence) for v in ['D3A','A1D','C2D','A1C']]
        expected = []
        with torch.inference_mode():
            for v in variants:
                ids = [0] + self.tokenizer.convert_tokens_to_ids(list(sequence)) + [2]
                ids[v.position] = self.tokenizer.mask_token_id
                logits = self.raw(sequence_tokens=torch.tensor([ids])).sequence_logits[0, v.position]
                # Independent oracle: log probability ratio equals logit difference.
                expected.append(float(logits[self.tokenizer.convert_tokens_to_ids(v.alternate)] - logits[self.tokenizer.convert_tokens_to_ids(v.wild_type)]))
        for size in [1, 2, 4]:
            scores, _ = score_variants(self.model,self.tokenizer,sequence,variants,size,0,False)
            torch.testing.assert_close(torch.tensor(scores),torch.tensor(expected),atol=2e-6,rtol=1e-5)

    def test_padding_and_wrong_precision_rejected(self):
        torch = self.torch
        with self.assertRaises(ValueError): self.model(torch.tensor([[0,5,1,2]]))
        with self.assertRaises(ValueError): self.model(torch.tensor([[0,5,2]]),torch.tensor([[1,0,1]]))
        self.raw.half()
        with self.assertRaises(ValueError): self.model(torch.tensor([[0,5,2]]))

    def test_autocast_rejected(self):
        with self.torch.autocast('cpu',dtype=self.torch.bfloat16):
            with self.assertRaises(ValueError): self.model(self.torch.tensor([[0,5,2]]))

    def test_unverified_revision_rejected_before_download(self):
        from tem1.esmc import load_esmc, MODEL_ID
        with patch('tem1.esmc.hf_hub_download') as download:
            with self.assertRaises(ValueError): load_esmc(MODEL_ID,'main',self.torch.device('cpu'))
            download.assert_not_called()
