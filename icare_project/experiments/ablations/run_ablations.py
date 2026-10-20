import os
import sys
import argparse

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.ablations.train_eval_ablations import run_all_ablations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run strictly leak-free ablation study.")
    parser.add_argument('--output_csv', type=str, default='ablation_table2.csv', help="Output CSV path for results")
    parser.add_argument('--eval_only', action='store_true', help="Evaluate existing checkpoints instead of retraining fresh")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_ablations', help="Directory to store or load checkpoints")
    args = parser.parse_args()
    run_all_ablations(output_csv=args.output_csv, eval_only=args.eval_only, checkpoint_dir=args.checkpoint_dir)
