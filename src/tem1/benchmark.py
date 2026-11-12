"""Run pinned model sizes sequentially, each in an isolated process."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/esm2-scale.json'))
    parser.add_argument('--sizes', nargs='+', default=['35m', '150m', '650m'])
    parser.add_argument('--output', type=Path, default=Path('results/scale'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/tem1'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit-positions', type=int)
    args = parser.parse_args()
    presets = {p['name']: p for p in json.loads(args.config.read_text())}
    if len(set(args.sizes)) != len(args.sizes) or set(args.sizes) - presets.keys():
        parser.error('Sizes must be unique names in the configuration')
    for size in args.sizes:
        destination = args.output / size
        if destination.exists() and any(destination.iterdir()):
            parser.error(f'Output is not empty: {destination}; choose a fresh output root')
    for size in args.sizes:
        p = presets[size]
        command = [sys.executable, '-m', 'tem1', 'run', '--model', p['model'],
                   '--revision', p['revision'], '--batch-size', str(p['batch_size']),
                   '--device', args.device, '--data-dir', str(args.data_dir),
                   '--output', str(args.output / size)]
        if args.limit_positions is not None:
            command += ['--limit-positions', str(args.limit_positions)]
        print(f'Running {size}', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
