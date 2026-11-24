"""Statistics, figures and index for exported preprocessing caches.

Layout written by preprocess.py / data_tuing.py:
    data/cache/
    ├── index.csv                 one row per cache (statistics only)
    └── <settings-name>/
        ├── exp.h5, sim.h5
        ├── stats.json            statistics of all preprocessed images + settings/sources
        ├── stats.md              the same as a short readable report
        └── figures/              examples.png, comparison.png, histograms.png

Statistics, comparison and histograms are computed on ALL images while convert() runs
(Accumulator), original and preprocessed. The tuning app shows the same figures live on
its sample, using summary() / fig_examples() / fig_comparison().
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- statistics
class Accumulator:
    """Collects statistics and figure data while convert() runs, for ALL images, for the
    original ("raw") and the preprocessed ("pp") data. Nothing is read a second time.
    Exact: moments, min/max, zero/saturated shares, mean image, per-image totals, histogram.
    Sampled: pixel percentiles (pixel_sample pixels) and PCA images (pca_sample images)."""

    def __init__(self, n_total, pca_sample=2000, pixel_sample=2_000_000, bins=200, seed=0):
        self.n_total, self.rng = n_total, np.random.default_rng(seed)
        self.pca_sample, self.pixel_sample = pca_sample, pixel_sample
        self.edges = np.linspace(0, 1, bins + 1)
        self.labels = []
        self.st = {k: {"n": 0, "s": 0.0, "s2": 0.0, "lo": np.inf, "hi": -np.inf, "zeros": 0, "sat": 0,
                       "img_sum": None, "totals": [], "hist": np.zeros(bins, np.int64),
                       "pix": [], "imgs": []} for k in ("raw", "pp")}

    def update(self, raw, pp, Y):
        k_img = int(round(self.pca_sample * len(raw) / self.n_total))
        pick = self.rng.choice(len(raw), min(k_img, len(raw)), replace=False) if k_img else []
        self.labels.append(np.asarray(Y, np.float64))
        for key, X in (("raw", raw), ("pp", pp)):
            X = np.asarray(X, np.float32)
            d, flat = self.st[key], X.reshape(len(X), -1)
            d["n"] += len(X)
            d["s"] += float(flat.sum(dtype=np.float64)); d["s2"] += float((flat.astype(np.float64) ** 2).sum())
            d["lo"], d["hi"] = min(d["lo"], float(flat.min())), max(d["hi"], float(flat.max()))
            d["zeros"] += int((flat == 0).sum()); d["sat"] += int((flat >= 0.999).sum())
            d["img_sum"] = X.sum(0, dtype=np.float64) + (0 if d["img_sum"] is None else d["img_sum"])
            d["totals"].append(flat.sum(1))
            d["hist"] += np.histogram(np.clip(flat, 0, 1), self.edges)[0]
            k_pix = max(1, int(round(self.pixel_sample * len(X) / self.n_total)))
            d["pix"].append(flat.ravel()[self.rng.integers(0, flat.size, k_pix)])
            if len(pick):
                d["imgs"].append(X[pick])

    # -- results
    def mean_image(self, key):
        return self.st[key]["img_sum"] / self.st[key]["n"]

    def totals(self, key):
        return np.concatenate(self.st[key]["totals"])

    def images(self, key):
        return np.concatenate(self.st[key]["imgs"]) if self.st[key]["imgs"] else None

    def stats(self, key="pp"):
        d = self.st[key]
        n_pix = d["n"] * d["img_sum"].size
        mean = d["s"] / n_pix
        pix, tot, Y = np.concatenate(d["pix"]), self.totals(key), np.concatenate(self.labels)
        q = lambda a, ps: {f"p{p:g}": float(v) for p, v in zip(ps, np.percentile(a, ps))}
        return {
            "n_images": d["n"], "shape": list(d["img_sum"].shape),
            "pixel": {"mean": mean, "std": float(np.sqrt(max(d["s2"] / n_pix - mean ** 2, 0))),
                      "min": d["lo"], "max": d["hi"], **q(pix, [1, 50, 99]),
                      "zeros_frac": d["zeros"] / n_pix, "saturated_frac": d["sat"] / n_pix},
            "total_intensity": {"mean": float(tot.mean()), **q(tot, [5, 50, 95])},
            "labels": {c: {"min": float(Y[:, j].min()), "max": float(Y[:, j].max()),
                           "mean": float(Y[:, j].mean()), "n_unique": int(len(np.unique(Y[:, j])))}
                       for j, c in enumerate("XYZ")},
        }


def file_info(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p.resolve()), "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}


# --------------------------------------------------------------------------- figures
def summary(Xs, Xe, n_pca=2000, seed=0):
    """Mean images, row profiles, total intensities and a PCA (fit on sim) for one stage."""
    from sklearn.decomposition import PCA
    n = min(len(Xs), n_pca)
    A = Xs[np.random.default_rng(seed).permutation(len(Xs))[:n]]
    pca = PCA(2, random_state=seed).fit(A.reshape(n, -1))
    return {
        "mean_s": Xs.mean(0), "mean_e": Xe.mean(0),
        "prof_s": Xs.mean((0, 2)), "prof_e": Xe.mean((0, 2)),
        "tot_s": Xs.reshape(len(Xs), -1).sum(1), "tot_e": Xe.reshape(len(Xe), -1).sum(1),
        "pca_s": pca.transform(A.reshape(n, -1)), "pca_e": pca.transform(Xe.reshape(len(Xe), -1)),
    }


def summary_from_acc(acc_sim, acc_exp, key, seed=0):
    """Same dict as summary(), but means/profiles/totals from ALL images; PCA fitted on the
    sampled sim images and applied to the sampled exp images."""
    from sklearn.decomposition import PCA
    ms, me = acc_sim.mean_image(key), acc_exp.mean_image(key)
    A, B = acc_sim.images(key), acc_exp.images(key)
    pca = PCA(2, random_state=seed).fit(A.reshape(len(A), -1))
    return {
        "mean_s": ms, "mean_e": me, "prof_s": ms.mean(1), "prof_e": me.mean(1),
        "tot_s": acc_sim.totals(key), "tot_e": acc_exp.totals(key),
        "pca_s": pca.transform(A.reshape(len(A), -1)), "pca_e": pca.transform(B.reshape(len(B), -1)),
    }


def fig_examples(Xs_raw, Xe_raw, Xs, Xe, Ys, Ye, n=4, seed=0):
    import matplotlib.pyplot as plt
    r = np.random.default_rng(seed)
    si, ei = r.choice(len(Xs), n, replace=False), r.choice(len(Xe), n, replace=False)
    rows = [("sim raw", Xs_raw, si, Ys), ("sim pp", Xs, si, Ys),
            ("exp raw", Xe_raw, ei, Ye), ("exp pp", Xe, ei, Ye)]
    fig, ax = plt.subplots(4, n + 1, figsize=(2.8 * (n + 1), 10.5))
    for k, (nm, X, idx, Y) in enumerate(rows):
        for c, i in enumerate(idx):
            ax[k, c].imshow(X[i], cmap="inferno", vmin=0, vmax=1)
            ax[k, c].set_title(f"{nm} #{i}\n{np.array2string(Y[i], precision=2)}", fontsize=8)
            ax[k, c].axis("off")
        flat = X.ravel()
        flat = flat[r.integers(0, flat.size, min(flat.size, 200_000))]   # sample: fast, same shape
        ax[k, n].hist(flat, 100, density=True, log=True, color="C0" if "sim" in nm else "C1")
        ax[k, n].set_title(f"{nm} pixels", fontsize=8)
    fig.tight_layout()
    return fig


def fig_comparison(raw, pp):
    """2 x 6 grid, top original, bottom preprocessed; difference images share one colour scale."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 6, figsize=(26, 8))
    vmax = max([float(np.abs(S["mean_e"] - S["mean_s"]).max())
                for S in (raw, pp) if S["mean_s"].shape == S["mean_e"].shape] or [1.0])
    for r, (lbl, S) in enumerate([("original", raw), ("preprocessed", pp)]):
        ax[r, 0].imshow(S["mean_s"], cmap="inferno"); ax[r, 0].set_title(f"{lbl}: sim mean")
        ax[r, 1].imshow(S["mean_e"], cmap="inferno"); ax[r, 1].set_title(f"{lbl}: exp mean")
        if S["mean_s"].shape == S["mean_e"].shape:
            im = ax[r, 2].imshow(S["mean_e"] - S["mean_s"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax[r, 2], fraction=0.046)
        ax[r, 2].set_title(f"{lbl}: exp − sim (same scale)")
        ax[r, 3].plot(S["prof_s"], label="sim"); ax[r, 3].plot(S["prof_e"], label="exp")
        ax[r, 3].set_title(f"{lbl}: row profile (top → bottom)"); ax[r, 3].legend(fontsize=7)
        b = np.histogram_bin_edges(np.r_[S["tot_s"], S["tot_e"]], 60)
        ax[r, 4].hist(S["tot_s"], b, density=True, alpha=.6, label="sim")
        ax[r, 4].hist(S["tot_e"], b, density=True, alpha=.6, label="exp")
        ax[r, 4].set_title(f"{lbl}: total intensity per image"); ax[r, 4].legend(fontsize=7)
        ax[r, 5].scatter(S["pca_s"][:, 0], S["pca_s"][:, 1], s=2, alpha=.3, label="sim")
        ax[r, 5].scatter(S["pca_e"][:, 0], S["pca_e"][:, 1], s=3, alpha=.5, c="C1", label="exp")
        ax[r, 5].set_title(f"{lbl}: PCA (fit on sim)"); ax[r, 5].legend(fontsize=7)
        for a in ax[r, :3]:
            a.axis("off")
    fig.tight_layout()
    return fig


def fig_histograms(acc_sim, acc_exp):
    """Pixel histograms of ALL pixels, sim vs exp, original (left) and preprocessed (right)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    e = acc_sim.edges
    for a, (key, lbl) in zip(ax, [("raw", "original"), ("pp", "preprocessed")]):
        for acc, nm in ((acc_sim, "sim"), (acc_exp, "exp")):
            h = acc.st[key]["hist"]
            a.stairs(h / max(h.sum(), 1) / np.diff(e), e, fill=True, alpha=.5, label=nm)
        a.set_yscale("log"); a.set_title(f"{lbl}: pixel values, all images (log)"); a.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- report
def _md(report):
    st, p = report["preprocessed"], report["settings"]
    L = [f"# Cache `{report['name']}`", "",
         f"- created: {report['created']}",
         f"- example images: {report.get('examples') or 'n/a'}", "",
         "## Settings", "", "| step | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in p.items()]
    L += ["", "## Preprocessed data (the complete cached sim.h5 / exp.h5)", "", "| | sim | exp |", "|---|---|---|"]
    row = lambda lbl, f: L.append(f"| {lbl} | {f(st['sim'])} | {f(st['exp'])} |")
    row("images", lambda d: d["n_images"])
    row("shape", lambda d: "×".join(map(str, d["shape"])))
    row("pixel mean ± std", lambda d: f"{d['pixel']['mean']:.3f} ± {d['pixel']['std']:.3f}")
    row("pixel min / max", lambda d: f"{d['pixel']['min']:.3f} / {d['pixel']['max']:.3f}")
    row("pixel p1 / p50 / p99", lambda d: f"{d['pixel']['p1']:.3f} / {d['pixel']['p50']:.3f} / {d['pixel']['p99']:.3f}")
    row("zeros", lambda d: f"{d['pixel']['zeros_frac']:.1%}")
    row("saturated (≥ 0.999)", lambda d: f"{d['pixel']['saturated_frac']:.1%}")
    row("total intensity p5 / p50 / p95",
        lambda d: f"{d['total_intensity']['p5']:.0f} / {d['total_intensity']['p50']:.0f} / {d['total_intensity']['p95']:.0f}")
    for c in "XYZ":
        row(f"label {c} range (unique)",
            lambda d, c=c: f"{d['labels'][c]['min']:.3f} – {d['labels'][c]['max']:.3f} ({d['labels'][c]['n_unique']})")
    L += ["", "## Sources", ""]
    L += [f"- {k}: `{v.get('path')}` ({v.get('size_bytes', 0) / 1e6:.0f} MB, modified {v.get('modified', '?')})"
          for k, v in report["sources"].items()]
    if report.get("conversion_seconds"):
        L += ["", "conversion time: " + ", ".join(f"{k} {v:.0f}s" for k, v in report["conversion_seconds"].items() if v)]
    if report.get("figures"):
        L += ["", "## Figures (comparison + histograms: all images; examples: random images)", ""]
        L += [f"![{f}](figures/{f}.png)" for f in report["figures"]]
    return "\n".join(L) + "\n"


def update_index(cache_root, report):
    """Add or replace this cache's row in <cache_root>/index.csv."""
    st = report["preprocessed"]
    row = {"name": report["name"], "created": report["created"],
           "shape": "×".join(map(str, st["sim"]["shape"])),
           "n_sim": st["sim"]["n_images"], "n_exp": st["exp"]["n_images"]}
    for d in ("sim", "exp"):
        row[f"{d}_mean"] = round(st[d]["pixel"]["mean"], 4)
        row[f"{d}_zeros"] = round(st[d]["pixel"]["zeros_frac"], 4)
        row[f"{d}_saturated"] = round(st[d]["pixel"]["saturated_frac"], 4)
        row[f"{d}_total_p50"] = round(st[d]["total_intensity"]["p50"], 1)
    row.update({f"set_{k}": json.dumps(v) if isinstance(v, (list, tuple)) else v
                for k, v in report["settings"].items()})
    path = Path(cache_root) / "index.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("name") != row["name"]]
    rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return path


def write_report(cache_dir, name, settings, sources, accs, examples_fig=None,
                 conversion_seconds=None, examples_note=None):
    """Write stats.json, stats.md, figures/*.png and the row in ../index.csv for one cache.
    accs = {"sim": Accumulator, "exp": Accumulator} filled during convert()."""
    import matplotlib.pyplot as plt
    cache_dir = Path(cache_dir)
    report = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
        "sources": {k: file_info(v) for k, v in sources.items()},
        "conversion_seconds": conversion_seconds or {},
        "examples": examples_note,
        "preprocessed": {d: accs[d].stats("pp") for d in ("sim", "exp")},
        "figures": [],
    }
    figs = {"comparison": fig_comparison(summary_from_acc(accs["sim"], accs["exp"], "raw"),
                                         summary_from_acc(accs["sim"], accs["exp"], "pp")),
            "histograms": fig_histograms(accs["sim"], accs["exp"])}
    if examples_fig is not None:
        figs = {"examples": examples_fig, **figs}
    (cache_dir / "figures").mkdir(exist_ok=True)
    for fname, fig in figs.items():
        fig.savefig(cache_dir / "figures" / f"{fname}.png", dpi=110)
        plt.close(fig)
        report["figures"].append(fname)
    (cache_dir / "stats.json").write_text(json.dumps(report, indent=2))
    (cache_dir / "stats.md").write_text(_md(report), encoding="utf-8")
    update_index(cache_dir.parent, report)
    return report