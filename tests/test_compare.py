import csv
import json
import tempfile
import unittest
from pathlib import Path
from tem1.compare import compare
from tem1.evaluation import evaluate


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / n for n in ['a', 'b']]
        for path in self.paths:
            path.mkdir()
            rows = [dict(mutant=f'A{i}C', position=i, wild_type='A', alternate='C',
                         DMS_score=i, esm2_score=i) for i in range(1, 6)]
            with (path / 'predictions.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            (path / 'metrics.json').write_text(json.dumps(dict(scope='full_assay', full_assay_n=5,
                assay_id='synthetic', spearman_rho=evaluate(range(5), range(5))['spearman_rho'])))
            (path / 'run.json').write_text(json.dumps(dict(status='complete', model={}, performance={})))

    def test_paired_identical_predictions_have_zero_difference(self):
        result = compare(self.paths, replicates=20)
        self.assertEqual(result['runs'][1]['delta_ci95'], [0., 0.])

    def test_smoke_rejected(self):
        p = self.paths[1] / 'metrics.json'
        m = json.loads(p.read_text()); m['scope'] = 'smoke_test'; p.write_text(json.dumps(m))
        with self.assertRaises(ValueError): compare(self.paths, 20)

    def test_changed_labels_rejected(self):
        p = self.paths[1] / 'predictions.csv'
        p.write_text(p.read_text().replace('A1C,1,A,C,1,1', 'A1C,1,A,C,9,1'))
        with self.assertRaises(ValueError): compare(self.paths, 20)

    def test_metric_mismatch_rejected(self):
        p = self.paths[1] / 'metrics.json'
        m = json.loads(p.read_text()); m['spearman_rho'] = -1; p.write_text(json.dumps(m))
        with self.assertRaises(ValueError): compare(self.paths, 20)
