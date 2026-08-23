"""
    Aggregate and plot all experiments.
"""

from glob import glob
from rpg_baselines.common.plotting import Plotter
from pathlib import Path
import argparse

EVAL_PATH = Path("./eval")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs', type=str, default='gt',
                        help="What type of runs (ground truth, vision inference) to aggregate.")
    return parser

def main():
    args = parser().parse_args()
    data = [fn for fn in glob(str(EVAL_PATH / args.obs / "*" / "rollout_*"))]
    data_keys = ["traj", "hits", "crosses", "residual", "gt_dist", "detections", "n_gates"]
    
    all_path = EVAL_PATH / args.obs / "all"
    all_path.mkdir(parents=True, exist_ok=True)
    
    plotter = Plotter(
        gates=[],
        rotations=[],
        sim_dt=0.00833333333,
        gate_dims=(0.75, 0.75),
        save_path=str(all_path)
    )
    plotter.load_data(npz=data, keys=data_keys)
    
    plotter.plot_residual()
    plotter.plot_completion()

if __name__ == "__main__":
    main()