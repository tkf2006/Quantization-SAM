"""Regenerate all archived RoBERTa figures without training or external assets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    output = args.output_root.resolve()
    records = ROOT / 'records'
    tuning_figures = output / 'param_tuning' / 'figures'
    jobs = [
        ['plot_e2_e5_results.py', '--e2-e5-run-dir', records / 'e2_e5_rte20_20260921_032917',
         '--e3-e4-run-dir', records / 'e3_e4_rte20_20260923_032434', '--output-dir', output / 'figures'],
        ['plot_qr_svd_diagnostics.py', '--qr-run-dir', records / 'stage1_diag_rte20_seed42_20260927_121223/e5_seed42',
         '--output-dir', tuning_figures],
    ]
    for script, stage in [
        ('plot_stage2_rho_search.py', 'stage2_rho_rte20_20260928_071417'),
        ('plot_stage3_rank_search.py', 'stage3_rank_rte20_20260928_145523'),
        ('plot_stage45_spectral_weighting.py', 'stage45_weighting_rte20_20261008_035610'),
        ('plot_stage67_formal_matrix.py', 'stage67_formal_matrix_20261008_085232'),
    ]:
        jobs.append([script, '--run-dir', records / stage, '--output-dir', tuning_figures])
    for script, *options in jobs:
        print(f'Running {script}', flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / 'analysis' / script), *map(str, options)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            result.check_returncode()

if __name__ == '__main__':
    main()
