"""Everything train.py writes, plots or prints for a run (no training code here).

Run folder runs/<date-time>_<model>[_<name>]/:
    config.json   all settings: arguments, constants, cache, split (incl. hash), model CONFIG
    model.txt     the model as built, layer by layer
    log.csv       one row per epoch: train_loss_std (running, standardized units), train_* (fixed
                  sample of sim_train, evaluated like sim_val -> directly comparable), val_*, exp_*, lr
    loss.png      MSE over epochs (mean, X, Y, Z) for train / sim_val / exp_adapt, updated every epoch
    best.pt       best checkpoint (lowest sim_val MSE), saved by train.py
    metrics.json  best/last epoch; final R² / MAE / MSE per split incl. AdaBN, each with a
                  baseline (always predicting the sim_train mean)
    predictions_<split>.csv   true and predicted X/Y/Z per image of the best checkpoint
    pred_<split>.png          predicted vs. true X/Y/Z
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

LABELS = "XYZ"


# --------------------------------------------------------------------------- metrics
def metrics(pred, true):
    """R², MAE and MSE per label + overall MSE. pred, true: (N, 3) arrays in label units."""
    out = {}
    for j, c in enumerate(LABELS):
        err = pred[:, j] - true[:, j]
        out[f"R2_{c}"] = float(1 - (err ** 2).sum() / ((true[:, j] - true[:, j].mean()) ** 2).sum())
        out[f"MAE_{c}"] = float(np.abs(err).mean())
        out[f"MSE_{c}"] = float((err ** 2).mean())
    out["MSE"] = float(((pred - true) ** 2).mean())
    return out


def baseline(y_mean, true):
    """Metrics of always predicting y_mean (the sim_train mean)."""
    return metrics(np.broadcast_to(y_mean, true.shape), true)


# --------------------------------------------------------------------------- run folder
def new_run(root, model_name, name=""):
    """Create and return runs/<date-time>_<model>[_<name>]/."""
    run = Path(root) / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{model_name}" + (f"_{name}" if name else ""))
    run.mkdir(parents=True)
    return run


def write_config(run, config, model):
    (run / "config.json").write_text(json.dumps(config, indent=2))
    (run / "model.txt").write_text(str(model))


def epoch_row(ep, train_loss, mt, mv, me, lr, seconds):
    """One log.csv row from the metrics of train sample / sim_val / exp_adapt."""
    return {"epoch": ep, "train_loss_std": train_loss,
            **{f"train_{k}": v for k, v in mt.items()},
            **{f"val_{k}": v for k, v in mv.items()},
            **{f"exp_{k}": v for k, v in me.items()},
            "lr": lr, "seconds": seconds}


def log_epoch(run, rows, best_ep, is_best):
    """Rewrite log.csv and loss.png with all rows so far, print one line for the last epoch."""
    with open(run / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    plot_losses(rows, best_ep, run / "loss.png")
    r = rows[-1]
    print(f"ep {r['epoch']:3d} | train MSE {r['train_MSE']:.5f} | val MSE {r['val_MSE']:.5f} "
          f"R² {r['val_R2_X']:.3f}/{r['val_R2_Y']:.3f}/{r['val_R2_Z']:.3f} | "
          f"exp R² {r['exp_R2_X']:.3f}/{r['exp_R2_Y']:.3f}/{r['exp_R2_Z']:.3f} | "
          f"{r['seconds']:.0f}s{' *' if is_best else ''}")


def save_split(run, name, idx, true, pred):
    """predictions_<name>.csv + pred_<name>.png for one evaluated split."""
    scatter(pred, true, name, run / f"pred_{name}.png")
    np.savetxt(run / f"predictions_{name}.csv", np.c_[idx, true, pred], delimiter=",",
               header="index,true_X,true_Y,true_Z,pred_X,pred_Y,pred_Z", comments="",
               fmt=["%d"] + ["%.6f"] * 6)


def write_results(run, results):
    """metrics.json + printed summary."""
    (run / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nbest checkpoint: epoch {results['best_epoch']} of {results['last_epoch']}")
    for name, r in results["splits"].items():
        b = r["baseline"]
        print(f"  {name:16s} R² X/Y/Z {r['R2_X']:7.3f} {r['R2_Y']:7.3f} {r['R2_Z']:7.3f} | "
              f"MAE X/Y/Z {r['MAE_X']:.4f} {r['MAE_Y']:.4f} {r['MAE_Z']:.4f} "
              f"(baseline {b['MAE_X']:.4f} {b['MAE_Y']:.4f} {b['MAE_Z']:.4f})")


# --------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def scatter(pred, true, title, path):
    """Predicted vs. true, one panel per label."""
    _, plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for j, a in enumerate(ax):
        a.scatter(true[:, j], pred[:, j], s=3, alpha=.4)
        lo, hi = true[:, j].min(), true[:, j].max()
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlabel("true"); a.set_ylabel("predicted"); a.set_title(f"{title}: {LABELS[j]}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_losses(rows, best_ep, path):
    """1 x 4: MSE over epochs (mean, X, Y, Z) for train / sim_val / exp_adapt; dotted line = best epoch."""
    matplotlib, plt = _plt()
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, key, title in zip(ax, ["MSE", "MSE_X", "MSE_Y", "MSE_Z"], ["mean", "X", "Y", "Z"]):
        for split, label in (("train", "train"), ("val", "sim_val"), ("exp", "exp_adapt")):
            a.plot(ep, [r[f"{split}_{key}"] for r in rows], marker=".", label=label)
        if best_ep:
            a.axvline(best_ep, color="k", ls=":", lw=1)
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(f"MSE {title}")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].set_ylabel("MSE (label units)"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
