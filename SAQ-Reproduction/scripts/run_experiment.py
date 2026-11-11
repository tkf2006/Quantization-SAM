"""Launch the archived CIFAR-100/W4A4 settings without machine-specific paths."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'saq': 'QSAM_SGD', 'qat-sgd': 'SGD'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, default='saq')
    p.add_argument('--seed', nargs='+', type=int, default=[1])
    p.add_argument('--data-root', type=Path, required=True, help='Contains cifar/cifar-100-python/')
    p.add_argument('--pretrained', type=Path, required=True, help='Original FP32 r20_w32a32.pth')
    p.add_argument('--output', type=Path, default=ROOT / 'runs')
    p.add_argument('--gpu', default='0')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--projection-seed', type=int, help='Random controls default to 10000 + training seed')
    p.add_argument('--print-command', action='store_true', help='Show arguments without launching training')
    args = p.parse_args()
    data, checkpoint = args.data_root.resolve(), args.pretrained.resolve()
    if not args.print_command:
        for file in [data / 'cifar/cifar-100-python/train', data / 'cifar/cifar-100-python/test', checkpoint]:
            if not file.is_file():
                p.error(f'Required input does not exist: {file}')
    if args.epochs < 1 or args.rank < 1:
        p.error('epochs and rank must be positive')
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '8')
    env.setdefault('MKL_NUM_THREADS', '8')
    for seed in args.seed:
        qat = args.method == 'qat-sgd'
        cmd = [sys.executable, '-u', str(ROOT / 'train_sam.py'),
               '--save_path', str(args.output.resolve() / args.method),
               '--data_path', str(data), '--pretrained', str(checkpoint),
               '--dataset', 'cifar100', '--network', 'qpreresnet20' if qat else 'qsampreresnet20',
               '--quan_type', 'LIQ_wn' if qat else 'LIQ_wn_qsam',
               '--qw', '4.0', '--qa', '4.0', '--opt_type', METHODS[args.method],
               '--rho', '0.0' if qat else '0.4', '--lr', '0.01', '--clip_lr', '0.01',
               '--momentum', '0.9', '--weight_decay', '0.0001', '--n_epochs', str(args.epochs),
               '--batch_size', '128', '--n_threads', '4', '--step', '80,120',
               '--lr_scheduler_type', 'multi_step', '--seed', str(seed), '--gpu', args.gpu,
               '--date', datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
               '--experiment_id', f'{args.method}_seed{seed}']
        if args.method in ('rs', 'qr', 'svd-rs', 'svd-qr'):
            projection_seed = args.projection_seed
            if projection_seed is None:
                projection_seed = 10000 + seed if args.method in ('rs', 'svd-rs') else 12345
            cmd += ['--projection_rank', str(args.rank), '--projection_seed', str(projection_seed)]
        print(json.dumps(cmd, ensure_ascii=False), flush=True)
        if not args.print_command:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
