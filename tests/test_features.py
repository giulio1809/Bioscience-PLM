"""Alignment and leakage checks for the feature preparation script."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import compute_esm_features as f


class FeatureTests(unittest.TestCase):
    def setUp(self):
        seq = 'ACDEF'
        self.folds = pd.DataFrame([
            dict(mutant=f'{wt}{i+1}{alt}', wt_aa=wt, position=i+1, mut_aa=alt,
                 DMS_score=float(i), fold=i)
            for i, wt in enumerate(seq) for alt in 'GH'])
        self.raw = self.folds[['mutant', 'DMS_score']].copy()

    def test_shuffled_rows_align_by_variant(self):
        f.validate_folds(self.raw.sample(frac=1, random_state=3), self.folds)

    def test_position_cannot_span_folds(self):
        self.folds.loc[0, 'fold'] = 1
        with self.assertRaisesRegex(SystemExit, 'more than one fold'):
            f.validate_folds(self.raw, self.folds)

    def test_target_annotation_and_fractional_position_rejected(self):
        for column, value in [('DMS_score', 999.), ('mut_aa', 'Y'), ('position', 1.5)]:
            frame = self.folds.copy()
            frame[column] = frame[column].astype(object)
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(SystemExit):
                f.validate_folds(self.raw, frame)

    def test_missing_or_duplicate_variants_rejected(self):
        for frame in [self.folds.iloc[1:], pd.concat([self.folds,self.folds.iloc[:1]])]:
            with self.assertRaises(SystemExit):f.validate_folds(self.raw,frame)

    def test_completed_feature_shapes_and_fold_alignment(self):
        frame = self.folds[['mutant','position','fold']].iloc[[5,1]].copy()
        frame['zeroshot'] = [1.,2.]
        for column in f.FEATURE_COLUMNS:frame[column] = [np.zeros(4,dtype=np.float16) for _ in range(2)]
        f.validate_completed(frame,self.folds,4)
        frame.iloc[0,frame.columns.get_loc('fold')] = 0
        with self.assertRaises(SystemExit):f.validate_completed(frame,self.folds,4)
        frame.iloc[0,frame.columns.get_loc('fold')] = 2
        frame.at[frame.index[0],'emb_mut_pos'] = np.zeros(3,dtype=np.float16)
        with self.assertRaises(SystemExit):f.validate_completed(frame,self.folds,4)

    def test_unmasked_score_equals_wt_logit_difference(self):
        logits = np.arange(5*20,dtype=float).reshape(5,20)/10
        ids = dict(zip(f.AMINO_ACIDS,range(20)))
        scores = f.zero_shot_scores(logits,self.folds,ids)
        expected = [logits[r.position-1,ids[r.mut_aa]]-logits[r.position-1,ids[r.wt_aa]] for r in self.folds.itertuples()]
        np.testing.assert_allclose(scores,expected,atol=1e-12)

    def test_resume_with_gaps_keeps_variant_score_alignment(self):
        import torch
        from transformers import EsmConfig,EsmForMaskedLM,EsmTokenizer
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); vocab=['<cls>','<pad>','<eos>','<unk>',*f.AMINO_ACIDS,'<mask>']
            (p/'vocab.txt').write_text('\n'.join(vocab)+'\n')
            tokenizer=EsmTokenizer(vocab_file=str(p/'vocab.txt'))
            model=EsmForMaskedLM(EsmConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,pad_token_id=1,mask_token_id=len(vocab)-1,
                position_embedding_type='rotary',max_position_embeddings=32)).eval()
            scores=np.arange(len(self.folds),dtype=float)
            completed=self.folds[['mutant','position','fold']].iloc[[7,1]].copy()
            completed['zeroshot']=scores[[7,1]]
            for column in f.FEATURE_COLUMNS: completed[column]=[np.zeros(16,dtype=np.float16) for _ in range(2)]
            result=f.compute_variants(model,tokenizer,'ACDEF',self.folds,scores,torch.device('cpu'),3,p/'features.parquet',completed)
            f.validate_completed(result,self.folds,16)
            aligned=result.set_index('mutant').loc[self.folds.mutant]
            np.testing.assert_array_equal(aligned.zeroshot,scores)
