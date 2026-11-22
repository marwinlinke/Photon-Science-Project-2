import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

INDICES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ap = argparse.ArgumentParser()
ap.add_argument(
    "--path", type=str, default="data/project2_simulation_103k_norm16bin.h5"
)
ap.add_argument("--output", type=str, default="samples/simulation")
ap.add_argument("--num", type=int, default=10)
ap.add_argument("--indices", type=list, default=INDICES)
a = ap.parse_args()

out_dir = Path(a.output)
out_dir.mkdir(exist_ok=True)

with h5py.File(a.path, "r") as f:
    X = f["X"]
    y = f["Y"]
    assert isinstance(X, h5py.Dataset)
    assert isinstance(y, h5py.Dataset)

    total = X.shape[0]
    print(f"{total} images total")

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
