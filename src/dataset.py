"""Load a preprocessed cache and the fixed train/val/test splits.

    Xs, Ys, Xe, Ye, splits = load("data/cache/<settings>")

X: images (N, H, W) as float16 (convert each batch to float32 before the model),
Y: labels (N, 3) = X, Y, Z as float32. Everything is loaded into memory once.

The splits are created on first use and stored in data/splits.json, so every run uses
exactly the same images. They are fixed by the constants below (change them only on
purpose: results from different splits are not comparable).
    sim_train / sim_val    simulation, random per image; VAL_FRAC of it is sim_val
    exp_adapt / exp_test   experiment, whole TILE x TILE squares of the X-Y plane per side,
                           so neighbouring scan images (near-copies) stay together;
                           EXP_TEST_FRAC of it is exp_test (final evaluation only)

Check a cache:  python src/dataset.py data/cache/<settings>
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

SPLIT_FILE = Path("data/splits.json")
VAL_FRAC = 0.1        # share of the simulation used for validation
EXP_TEST_FRAC = 0.3  # share of the experiment kept for the final test
TILE = 0.1            # tile size in label units for the experiment split
SEED = 0


def load_h5(path):
    with h5py.File(path, "r") as f:
        X = torch.from_numpy(f["X"][:].astype(np.float16, copy=False))   # no extra copy if already float16
        Y = torch.from_numpy(f["Y"][:].astype(np.float32, copy=False))
    return X, Y


def make_splits(n_sim, y_exp):
    """Load data/splits.json, or create it on first use. Returns dict of index tensors."""
    y_exp = np.asarray(y_exp)
    meta = {"n_sim": int(n_sim), "n_exp": len(y_exp), "val_frac": VAL_FRAC,
            "exp_test_frac": EXP_TEST_FRAC, "tile": TILE, "seed": SEED}

    if SPLIT_FILE.exists():
        d = json.loads(SPLIT_FILE.read_text())
        if d.pop("_meta") != meta:
            raise ValueError(f"{SPLIT_FILE} was made for other data or constants; delete it to recreate.")
        return {k: torch.tensor(v) for k, v in d.items()}

    perm = np.random.default_rng(SEED).permutation(n_sim)
    n_val = int(VAL_FRAC * n_sim)

    # whole tiles go to exp_test, in random order, until it holds ~EXP_TEST_FRAC of the images
    _, tile, counts = np.unique(np.floor(y_exp[:, :2] / TILE).astype(int), axis=0,
                                return_inverse=True, return_counts=True)
    tile, test, n = tile.ravel(), [], 0
    for t in np.random.default_rng(SEED).permutation(len(counts)):   # own generator: independent of n_sim
        if n >= EXP_TEST_FRAC * len(y_exp):
            break
        test.append(t)
        n += counts[t]
    is_test = np.isin(tile, test)

    splits = {"sim_train": perm[n_val:], "sim_val": perm[:n_val],
              "exp_adapt": np.where(~is_test)[0], "exp_test": np.where(is_test)[0]}
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({**{k: v.tolist() for k, v in splits.items()}, "_meta": meta}))
    return {k: torch.from_numpy(v) for k, v in splits.items()}


def split_info():
    """How data/splits.json was made (its _meta) + a SHA-256 of the file: two runs with the
    same hash used exactly the same images."""
    raw = SPLIT_FILE.read_bytes()
    return {"file": SPLIT_FILE.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()[:16],
            **json.loads(raw)["_meta"]}


def load(cache):
    """sim.h5 + exp.h5 from a cache folder, plus the splits."""
    Xs, Ys = load_h5(Path(cache)  /"sim.h5")
    Xe, Ye = load_h5(Path(cache) / "exp.h5")
    return Xs, Ys, Xe, Ye, make_splits(len(Xs), Ye.numpy())


if __name__ == "__main__":
    import sys
    Xs, Ys, Xe, Ye, splits = load(sys.argv[1])
    print(f"sim {tuple(Xs.shape)} {Xs.dtype}, exp {tuple(Xe.shape)} {Xe.dtype}")
    for k, v in splits.items():
        print(f"  {k:10s} {len(v):6d}")