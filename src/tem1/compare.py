"""Compare full-assay predictions with paired position-cluster bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .evaluation import evaluate


def read_run(path):
    metrics = json.loads((path / 'metrics.json').read_text())
    run = json.loads((path / 'run.json').read_text())
    if metrics['scope'] != 'full_assay' or run['status'] != 'complete':
        raise ValueError('Only completed full assays can be compared')
    with (path / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    by_id = {r['mutant']: r for r in rows}
    if len(by_id) != len(rows) or len(rows) != metrics['full_assay_n']:
        raise ValueError('Duplicate or missing variants')
    return by_id, metrics, run


def compare(paths, replicates=2000, seed=0):
    if replicates < 1 or len(paths) < 2:
        raise ValueError('Require two runs and positive bootstrap count')
    runs = [read_run(p) for p in paths]
    keys = sorted(runs[0][0])
    reference = runs[0][0]
    y = np.array([float(reference[k]['DMS_score']) for k in keys])
    positions = np.array([int(reference[k]['position']) for k in keys])
    scores = []
    for rows, metrics, _ in runs:
        if set(rows) != set(keys) or metrics['assay_id'] != runs[0][1]['assay_id']:
            raise ValueError('Assays/variant sets differ')
        for k in keys:
            if any(rows[k][f] != reference[k][f] for f in ['position', 'wild_type', 'alternate']):
                raise ValueError('Variant annotations differ')
            if float(rows[k]['DMS_score']) != float(reference[k]['DMS_score']):
                raise ValueError('Experimental scores differ')
        score = np.array([float(rows[k]['esm2_score']) for k in keys])
        rho = evaluate(y, score)['spearman_rho']
        if not np.isclose(rho, metrics['spearman_rho'], atol=1e-12, rtol=0):
            raise ValueError('Saved metric disagrees with predictions')
        scores.append(score)
    groups = [np.flatnonzero(positions == p) for p in np.unique(positions)]
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, len(scores)))
    for b in range(replicates):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        boot[b] = [spearmanr(y[idx], x[idx]).statistic for x in scores]
    if not np.isfinite(boot).all():
        raise ValueError('Undefined bootstrap correlation')
    result = []
    for i, (path, (_, metrics, run)) in enumerate(zip(paths, runs)):
        result.append(dict(path=str(path), model=run['model'], performance=run['performance'],
            spearman_rho=metrics['spearman_rho'],
            rho_ci95=np.quantile(boot[:, i], [.025, .975]).tolist(),
            delta_vs_first=metrics['spearman_rho']-runs[0][1]['spearman_rho'],
            delta_ci95=np.quantile(boot[:, i]-boot[:, 0], [.025, .975]).tolist()))
    return dict(assay_id=runs[0][1]['assay_id'], n_variants=len(keys), seed=seed,
        bootstrap_replicates=replicates, bootstrap_unit='position; paired across models',
        limitation='Within-assay uncertainty only; not generalization to other proteins or assay conditions.',
        runs=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.runs, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
