"""Offline numerical tests using a tiny random ESM and an analytical fake model.

No pretrained weights are downloaded. These are correctness tests, not fitness results.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tem1.data import AMINO_ACIDS, parse_mutation

HAS_STACK = all(importlib.util.find_spec(p) is not None for p in ("torch", "transformers"))


@unittest.skipUnless(HAS_STACK, "Install project dependencies to test PyTorch/Transformers scoring")
class ScoringTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *AMINO_ACIDS, "<mask>"]
        path = Path(self.temp.name) / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n")
        self.tokenizer = EsmTokenizer(vocab_file=str(path))
        config = EsmConfig(
            vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, pad_token_id=1, mask_token_id=len(vocab) - 1,
            position_embedding_type="rotary", token_dropout=True,
        )
        torch.manual_seed(7)
        self.model = EsmForMaskedLM(config).eval()
        self.sequence = "ACD"
        # Deliberately unsorted, with two substitutions sharing position 1.
        self.variants = [parse_mutation(s, self.sequence) for s in ("D3A", "A1D", "C2D", "A1C")]

    def test_batched_scores_match_independent_single_site_oracle(self):
        from tem1.scoring import score_variants
        torch = self.torch
        expected = []
        with torch.inference_mode():
            for variant in self.variants:
                # Independent reference: explicit BOS + residues + EOS.
                ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(list(self.sequence)) + [self.tokenizer.eos_token_id]
                ids[variant.position] = self.tokenizer.mask_token_id
                inputs = torch.tensor([ids])
                logits = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
                distribution = torch.log_softmax(logits[0, variant.position], -1)
                expected.append(float(distribution[self.tokenizer.convert_tokens_to_ids(variant.alternate)] - distribution[self.tokenizer.convert_tokens_to_ids(variant.wild_type)]))
        for size in (1, 2, 8):
            with self.subTest(batch_size=size):
                scores, metrics = score_variants(self.model, self.tokenizer, self.sequence, self.variants, size, 0, False)
                torch.testing.assert_close(torch.tensor(scores), torch.tensor(expected), atol=2e-6, rtol=1e-5)
                self.assertEqual(metrics["n_masked_positions"], 3)
                self.assertIsNone(metrics["cuda_peak_allocated_mib"])

    def test_exact_masking_reuse_and_sign(self):
        from tem1.scoring import score_variants
        torch, tokenizer = self.torch, self.tokenizer
        original = tokenizer(self.sequence, return_tensors="pt")["input_ids"]

        class AnalyticalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(max_position_embeddings=32)
                self.seen = []

            def forward(self, input_ids, attention_mask):
                for row in input_ids:
                    masked = torch.where(row == tokenizer.mask_token_id)[0]
                    if len(masked) != 1:
                        raise AssertionError("Exactly one mask required")
                    position = int(masked[0])
                    restored = row.clone()
                    restored[position] = original[0, position]
                    torch.testing.assert_close(restored, original[0])
                    self.seen.append(position)
                # Vocabulary logits have known differences, independent of softmax.
                logits = torch.arange(len(tokenizer), dtype=torch.float32).repeat(*input_ids.shape, 1)
                return SimpleNamespace(logits=logits)

        model = AnalyticalModel()
        scores, _ = score_variants(model, tokenizer, self.sequence, self.variants, 2, 0, False)
        expected = [tokenizer.convert_tokens_to_ids(v.alternate) - tokenizer.convert_tokens_to_ids(v.wild_type) for v in self.variants]
        torch.testing.assert_close(torch.tensor(scores), torch.tensor(expected, dtype=torch.float32))
        self.assertEqual(model.seen, [1, 2, 3])

    def test_truncation_forbidden(self):
        from tem1.scoring import score_variants
        self.model.config.max_position_embeddings = 4
        with self.assertRaisesRegex(ValueError, "context limit"):
            score_variants(self.model, self.tokenizer, self.sequence, self.variants)
