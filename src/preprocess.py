"""Deterministic preprocessing for Project 2 (simulation + experiment).

The same `preprocess()` is applied to both domains, in training and at inference.
Random augmentations do NOT belong here (-> augment.py, training only).

Steps (each can be switched off via its parameter):
  1. background  experiment only: subtract a per-pixel background template
                 (low percentile over all exp images = detector/stray-light background;
                 the beam moves during the scan, so every pixel is dark in many images)
  2. crop        cut detector edge artefacts (bright left columns / top rows in exp)
  3. offset      subtract per-image low percentile, clip at 0 (the only step that sets the dark level)
  4. sqrt        optional, compresses bright/saturated regions (sim values are counts)
  5. blur        Gaussian blur, smooths the ray-hit noise of the simulation
  6. downsample  k x k average pooling
  7. scale       divide each image by its high percentile -> [0, 1]
                 (min_range: images with less signal than this are not stretched)
  8. dark_cut    soft threshold after scaling: x -> max(x - t, 0) / (1 - t).
                 Removes dark noise that the scaling amplified; no hard jump at t.

Usage (from repo root):
  python src/preprocess.py --sim data/project2_simulation_103k_norm16bin.h5 \
                           --exp data/project2_experiment_norm16bin.h5 \
                           --preview
  python src/preprocess.py ... --no-write --limit 3000      # only preview + domain-gap check

Output files are named after all settings, e.g.
  data/sim_pp_bg5_crop4-124-8-128_off1_sqrt0_blur1_ds2_sc0-99.5_mr0.001.h5
If a file with exactly these settings already exists, it is reused (cache); --force recomputes.
In training, open the file and read the settings back from f.attrs["params"].
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULTS = dict(
    bg_quantile=5.0,          # percentile for the exp background template; None = off
    crop=None,                # (row0, row1, col0, col1), e.g. (4, 124, 8, 128); None = off.
                              # Off by default: the background template already removes the bright
                              # detector edges of the experiment, and cropping cuts real sim features.
    offset_pct=1.0,           # per-image percentile subtracted; None = off
    sqrt=False,
    blur_sigma=1.0,           # in pixels of the cropped image; 0 = off
    downsample=2,             # 1 = off
    scale_pct=99.5,           # divide each image by this percentile; None = off
    min_range=1e-3,           # images with smaller dynamic range are not stretched (pure noise)
    dark_cut=None,            # soft threshold after scaling, e.g. 0.1-0.2; None = off
)


# --------------------------------------------------------------------------- steps
def estimate_background(X_exp, q=5.0, chunk=256):
    """Per-pixel q-th percentile over experiment images -> (H, W) template."""
    X_exp = np.asarray(X_exp, dtype=np.float32)
    return np.percentile(X_exp, q, axis=0).astype(np.float32)


def preprocess(X, p=DEFAULTS, background=None):
    """X: (N, H, W) raw images -> (N, h, w) float32 in [0, 1].
    `background` (H, W) is only passed for experiment data."""
    X = np.asarray(X, dtype=np.float32).copy()

    if background is not None:
        X -= background[None]
        np.clip(X, 0, None, out=X)

    if p.get("crop"):
        r0, r1, c0, c1 = p["crop"]
        X = X[:, r0:r1, c0:c1]

    if p.get("offset_pct") is not None:
        X -= np.percentile(X.reshape(len(X), -1), p["offset_pct"], axis=1)[:, None, None]
        np.clip(X, 0, None, out=X)

    if p.get("sqrt"):
        np.sqrt(X, out=X)

    if p.get("blur_sigma", 0) > 0:
        X = gaussian_filter(X, sigma=(0, p["blur_sigma"], p["blur_sigma"]), mode="nearest")

    k = p.get("downsample", 1)
    if k > 1:
        N, H, W = X.shape
        H, W = H - H % k, W - W % k
        X = X[:, :H, :W].reshape(N, H // k, k, W // k, k).mean(axis=(2, 4))

    if p.get("scale_pct") is not None:
        # dark level is already 0 (step 3), so only the upper end is set here
        hi = np.percentile(X.reshape(len(X), -1), p["scale_pct"], axis=1)[:, None, None]
        X = np.clip(X / np.maximum(hi, p.get("min_range", 1e-3)), 0, 1)

    t = p.get("dark_cut")
    if t:
        X = np.clip((X - t) / (1 - t), 0, 1)

    return X.astype(np.float32)


# --------------------------------------------------------------------------- io
def _params_json(p):
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}, sort_keys=True)


def run_name(p):
    """File-name tag built from ALL settings, e.g.
    bg5_crop4-124-8-128_off1_sqrt0_blur1_ds2_sc0-99.5_mr0.001
    Same settings -> same name, so the name itself identifies the cache."""
    f = lambda v: f"{v:g}"
    return "_".join([
        f"bg{f(p['bg_quantile'])}" if p.get("bg_quantile") is not None else "bg-off",
        "crop" + "-".join(map(str, p["crop"])) if p.get("crop") else "crop-off",
        f"off{f(p['offset_pct'])}" if p.get("offset_pct") is not None else "off-off",
        f"sqrt{int(bool(p.get('sqrt')))}",
        f"blur{f(p.get('blur_sigma', 0))}",
        f"ds{p.get('downsample', 1)}",
        f"sc{f(p['scale_pct'])}" if p.get("scale_pct") is not None else "sc-off",
        f"mr{f(p.get('min_range', 1e-3))}",
        f"dc{f(p['dark_cut'])}" if p.get("dark_cut") else "dc-off",
    ])


def output_paths(out_dir, p):
    """(exp_file, sim_file) for these settings; used by the CLI and the tuning app."""
    name = run_name(p)
    out = Path(out_dir)
    return out / f"exp_pp_{name}.h5", out / f"sim_pp_{name}.h5"


def cache_ok(dst, p):
    """True if `dst` exists and was written with exactly these parameters."""
    dst = Path(dst)
    if not dst.exists():
        return False
    try:
        with h5py.File(dst, "r") as f:
            return f.attrs.get("params") == _params_json(p) and f.attrs.get("complete", False)
    except OSError:          # half-written / corrupt file
        return False


def convert(src, dst, p, background=None, chunk=4096, progress=None):
    """Write preprocessed copy of `src` to `dst` (float16 images, float32 labels).
    `progress(n)` is called after each chunk with the number of images just written."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")        # write to temp file, rename when complete
    with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        N = fi["X"].shape[0]
        h, w = preprocess(fi["X"][:1], p, background).shape[1:]
        dX = fo.create_dataset("X", (N, h, w), dtype="float16", chunks=(min(256, N), h, w))
        fo.create_dataset("Y", data=fi["Y"][:].astype(np.float32))
        if background is not None:
            fo.create_dataset("background", data=background)
        for s in range(0, N, chunk):
            dX[s:s + chunk] = preprocess(fi["X"][s:s + chunk], p, background).astype(np.float16)
            print(f"  {dst.name}: {min(s + chunk, N)}/{N}", end="\r")
            if progress is not None:
                progress(min(chunk, N - s))
        fo.attrs["source"] = str(src)
        fo.attrs["params"] = _params_json(p)
        fo.attrs["complete"] = True
    tmp.replace(dst)
    print(f"  {dst.name}: done, shape {(N, h, w)}")


def read_subset(path, n, seed=0):
    with h5py.File(path, "r") as f:
        N = f["X"].shape[0]
        idx = np.arange(N) if n >= N else np.sort(np.random.default_rng(seed).choice(N, n, replace=False))
        return f["X"][idx].astype(np.float32), f["Y"][idx].astype(np.float32)


# --------------------------------------------------------------------------- checks
def domain_gap(Xs, Xe, n_pca=50, seed=0):
    """ROC-AUC of a linear sim-vs-exp classifier on PCA features (0.5 = no gap, 1.0 = trivial)."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None
    n = min(len(Xs), len(Xe))
    A, B = Xs[:n].reshape(n, -1), Xe[:n].reshape(n, -1)
    Z = PCA(n_pca, random_state=seed).fit(A).transform(np.r_[A, B])
    y = np.r_[np.zeros(n), np.ones(n)]
    return cross_val_score(LogisticRegression(max_iter=3000), Z, y, cv=5, scoring="roc_auc").mean()


def preview(Xs_raw, Xe_raw, Xs, Xe, Ys, Ye, out, n=4, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.random.default_rng(seed)
    si, ei = r.choice(len(Xs), n, replace=False), r.choice(len(Xe), n, replace=False)
    fig, ax = plt.subplots(4, n + 1, figsize=(3.2 * (n + 1), 11),
                           gridspec_kw=dict(width_ratios=[1] * n + [1.3]))
    rows = [("sim raw", Xs_raw, si, Ys), ("sim pp", Xs, si, Ys),
            ("exp raw", Xe_raw, ei, Ye), ("exp pp", Xe, ei, Ye)]
    for r_, (name, X, idx, Y) in enumerate(rows):
        for c, i in enumerate(idx):
            ax[r_, c].imshow(X[i], cmap="inferno", vmin=0, vmax=1)
            ax[r_, c].set_title(f"{name} #{i}\n{np.array2string(Y[i], precision=2)}", fontsize=8)
            ax[r_, c].axis("off")
        ax[r_, n].hist(X.ravel(), 100, density=True, log=True, color="C0" if "sim" in name else "C1")
        ax[r_, n].set_title(f"{name} pixel histogram", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"preview -> {out}")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="data/project2_simulation_103k_norm16bin.h5")
    ap.add_argument("--exp", default="data/project2_experiment_norm16bin.h5")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--force", action="store_true", help="recompute even if a matching cache file exists")
    ap.add_argument("--preview", nargs="?", const="auto", type=str,
                    help="PNG with before/after examples (default name: samples/preprocess_<settings>.png)")
    ap.add_argument("--limit", type=int, default=3000, help="sim images used for preview/gap check")
    ap.add_argument("--no-write", action="store_true", help="only preview + domain-gap check")
    # parameters (defaults in DEFAULTS)
    ap.add_argument("--bg-quantile", type=float, default=DEFAULTS["bg_quantile"], help="-1 = off")
    ap.add_argument("--crop", type=int, nargs=4, default=list(DEFAULTS["crop"] or (0, 0, 0, 0)),
                    help="r0 r1 c0 c1 (e.g. 4 124 8 128), or 0 0 0 0 = off")
    ap.add_argument("--offset-pct", type=float, default=DEFAULTS["offset_pct"], help="-1 = off")
    ap.add_argument("--sqrt", action="store_true")
    ap.add_argument("--blur", type=float, default=DEFAULTS["blur_sigma"])
    ap.add_argument("--downsample", type=int, default=DEFAULTS["downsample"])
    ap.add_argument("--scale", type=float, default=DEFAULTS["scale_pct"], help="high percentile, -1 = off")
    ap.add_argument("--min-range", type=float, default=DEFAULTS["min_range"])
    ap.add_argument("--dark-cut", type=float, default=-1, help="soft threshold after scaling (e.g. 0.15), -1 = off")
    a = ap.parse_args()

    p = dict(DEFAULTS)
    p.update(
        bg_quantile=None if a.bg_quantile < 0 else a.bg_quantile,
        crop=None if not any(a.crop) else tuple(a.crop),
        offset_pct=None if a.offset_pct < 0 else a.offset_pct,
        sqrt=a.sqrt, blur_sigma=a.blur, downsample=a.downsample,
        scale_pct=None if a.scale < 0 else a.scale,
        min_range=a.min_range,
        dark_cut=None if a.dark_cut <= 0 else a.dark_cut,
    )
    name = run_name(p)
    out = Path(a.out_dir)
    dst_exp, dst_sim = output_paths(out, p)
    print("params:", _params_json(p))
    print("name:  ", name)

    todo = [d for d in (dst_exp, dst_sim) if a.force or not cache_ok(d, p)]
    if not a.no_write and not todo and not a.preview:
        print(f"cache up to date: {dst_exp}, {dst_sim}  (use --force to recompute)")
        return

    # background template from the (unlabeled) experiment images
    with h5py.File(a.exp, "r") as f:
        Xe_raw = f["X"][:].astype(np.float32)
        Ye = f["Y"][:].astype(np.float32)
    bg = estimate_background(Xe_raw, p["bg_quantile"]) if p["bg_quantile"] is not None else None

    # quick check on a subset
    Xs_raw, Ys = read_subset(a.sim, a.limit)
    Xs, Xe = preprocess(Xs_raw, p), preprocess(Xe_raw, p, bg)
    print(f"output shape {Xs.shape[1:]}  |  sim mean {Xs.mean():.3f} zeros {np.mean(Xs == 0):.1%}"
          f"  |  exp mean {Xe.mean():.3f} zeros {np.mean(Xe == 0):.1%}")
    raw_auc, pp_auc = domain_gap(Xs_raw, Xe_raw), domain_gap(Xs, Xe)
    if pp_auc is not None:
        print(f"domain-classifier ROC-AUC  raw: {raw_auc:.3f}  ->  preprocessed: {pp_auc:.3f}")
    if a.preview:
        pv = Path(f"samples/preprocess_{name}.png") if a.preview == "auto" else Path(a.preview)
        preview(Xs_raw, Xe_raw, Xs, Xe, Ys, Ye, pv)

    if not a.no_write:
        out.mkdir(parents=True, exist_ok=True)
        for dst, src, b in ((dst_exp, a.exp, bg), (dst_sim, a.sim, None)):
            if dst in todo:
                convert(Path(src), dst, p, b)
            else:
                print(f"  {dst.name}: cache up to date, skipped")


if __name__ == "__main__":
    main()