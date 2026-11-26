"""Save single images with their labels as PNGs, to compare raw / preprocessed data.

    python src/visualizer.py                                              # raw simulation
    python src/visualizer.py --path data/original_103k_norm16bin/exp.h5 --output samples/experiment
    python src/visualizer.py --path data/cache/<settings>/sim.h5 --output samples/sim_pp --indices 0 5 42
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INDICES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--path", default="data/original_103k_norm16bin/sim.h5", help="h5 file with X and Y")
ap.add_argument("--output", default="samples/simulation")
ap.add_argument("--indices", type=int, nargs="+", default=INDICES, help="image indices, e.g. --indices 0 5 42")
a = ap.parse_args()

out_dir = Path(a.output)
out_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(a.path, "r") as f:
    X, y = f["X"], f["Y"]
    print(f"{X.shape[0]} images total")
    for k, i in enumerate(a.indices):
        img = X[i].astype(np.float32)  # (H, W)
        label = y[i]  # (3,)
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="inferno")
        plt.title(f"y = [{label[0]:.3f}, {label[1]:.3f}, {label[2]:.3f}]", fontsize=9)
        plt.axis("off")
        out = out_dir / f"sample_{k:02d}_{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {out}")
