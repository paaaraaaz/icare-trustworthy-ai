import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.ablations.train_eval_ablations import run_all_ablations

if __name__ == '__main__':
    run_all_ablations()
