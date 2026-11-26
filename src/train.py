"""Train a model on the simulation (Task 1) and evaluate it on the experiment (Task 2).

    python src/train.py --cache data/cache/<settings>                    # conv_mixer
    python src/train.py --cache data/cache/<settings> --model conv_mixer --epochs 50 --name long
    python src/train.py --cache data/cache/<settings> --limit 2000 --epochs 1    # quick test
    python src/train.py --cache data/cache/<settings> --final                    # + exp_test, once at the end

--model <name> imports src/models/<name>.py and calls its build().
The best checkpoint is chosen on sim_val only. exp_test is evaluated only with --final.
Targets are standardized with the sim_train mean/std (each of X, Y, Z gets the same weight in the
loss); predictions are converted back, so all metrics and CSVs are in the original label units.
Each run writes runs/<date-time>_<model>[_<name>]/:
    config.json   all settings: arguments, constants, cache, split (incl. hash), model CONFIG
    model.txt     the model as built, layer by layer
    log.csv       one row per epoch: train_loss (running, during the epoch), train_* (fixed
                  sample of sim_train, evaluated like sim_val -> directly comparable), val_*, exp_*, lr
    best.pt       best checkpoint (lowest sim_val MSE)
    metrics.json  best/last epoch; final R² / MAE / MSE per split incl. AdaBN, each with a
                  baseline (always predicting the sim_train mean)
    predictions_<split>.csv   true and predicted X/Y/Z per image of the best checkpoint
    pred_*.png    predicted vs. true X/Y/Z
"""
import argparse
import contextlib
import copy
import csv
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import load, split_info

BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
WEIGHT_DECAY = 1e-4
RUNS_DIR = Path("runs")
TRAIN_EVAL_SIZE = 10_000   # fixed sample of sim_train evaluated after every epoch (like sim_val)
LABELS = "XYZ"
GPU_DATA_SHARE = 0.5  # move all data to the GPU if it needs less than this share of the free GPU memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bfloat16 mixed precision on GPUs that support it (faster, less memory), otherwise float32
AUTOCAST = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else contextlib.nullcontext


def place_data(tensors):
    """All data to the GPU if it fits comfortably (< GPU_DATA_SHARE of free GPU memory),
    otherwise it stays in RAM and each batch is copied over. Returns (tensors, info)."""
    need = sum(t.numel() * t.element_size() for t in tensors)
    info = {"data_on": "cpu", "data_gb": round(need / 1e9, 2)}
    if DEVICE.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        info.update(gpu_free_gb=round(free / 1e9, 1), gpu_total_gb=round(total / 1e9, 1))
        if need < GPU_DATA_SHARE * free:
            tensors, info["data_on"] = [t.to(DEVICE) for t in tensors], "gpu"
    return tensors, info


def batches(X, Y, idx, batch_size, shuffle=False):
    """Batches of (x, y) on DEVICE; images go from float16 to float32 (B, 1, H, W) here."""
    if shuffle:
        idx = idx[torch.randperm(len(idx))]
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size].to(X.device)
        yield X[j].to(DEVICE).float().unsqueeze(1), Y[j].to(DEVICE)


@torch.no_grad()
def predict(model, X, Y, idx, norm):
    """Predictions in original label units. norm = (mean, std) of the sim_train labels on DEVICE."""
    model.eval()
    mu, sd = norm
    preds = []
    for x, _ in batches(X, Y, idx, EVAL_BATCH_SIZE):
        with AUTOCAST():
            preds.append((model(x).float() * sd + mu).cpu())
    return torch.cat(preds).numpy()


def metrics(pred, true):
    out = {}
    for j, c in enumerate(LABELS):
        err = pred[:, j] - true[:, j]
        out[f"R2_{c}"] = float(1 - (err ** 2).sum() / ((true[:, j] - true[:, j].mean()) ** 2).sum())
        out[f"MAE_{c}"] = float(np.abs(err).mean())
    out["MSE"] = float(((pred - true) ** 2).mean())
    return out


@torch.no_grad()
def adabn(model, X, Y, idx):
    """Copy of the model with BatchNorm statistics re-estimated on the given images (no labels used).
    None if the model has no BatchNorm."""
    m = copy.deepcopy(model)
    bns = [b for b in m.modules() if isinstance(b, nn.BatchNorm2d)]
    if not bns:
        return None
    m.eval()
    for b in bns:
        b.reset_running_stats()
        b.momentum = None      # plain average over all batches
        b.train()
    for x, _ in batches(X, Y, idx, BATCH_SIZE):
        m(x)
    return m.eval()


def scatter(pred, true, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for j, a in enumerate(ax):
        a.scatter(true[:, j], pred[:, j], s=3, alpha=.4)
        lo, hi = true[:, j].min(), true[:, j].max()
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlabel("true"); a.set_ylabel("predicted"); a.set_title(f"{title}: {LABELS[j]}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="data/cache/<settings> with sim.h5 + exp.h5")
    ap.add_argument("--model", default="conv_mixer", help="file name in src/models/ (without .py)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    ap.add_argument("--name", default="", help="optional label for the run folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="also evaluate exp_test (do this once, at the end)")
    ap.add_argument("--limit", type=int, default=0, help="use only N sim images for training and validation")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    # ---- data
    t0 = time.time()
    Xs, Ys, Xe, Ye, s = load(a.cache)
    (Xs, Ys, Xe, Ye), placement = place_data([Xs, Ys, Xe, Ye])
    train, val = s["sim_train"], s["sim_val"]
    if a.limit:
        train, val = train[:a.limit], val[:a.limit]
    print(f"data: sim {tuple(Xs.shape)}, exp {tuple(Xe.shape)} | train {len(train)}, val {len(val)}, "
          f"exp_adapt {len(s['exp_adapt'])}, exp_test {len(s['exp_test'])} | {time.time() - t0:.0f}s")
    print(f"data on {placement['data_on']} ({placement['data_gb']} GB"
          + (f", GPU {placement['gpu_free_gb']} of {placement['gpu_total_gb']} GB free)" if "gpu_free_gb" in placement else ")"))

    # ---- model, optimiser
    module = importlib.import_module(f"models.{a.model}")
    model = module.build().to(DEVICE)
    model_config = getattr(module, "CONFIG", {})   # sizes from the model file, {} if it has none
    split = split_info()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    loss_fn = nn.MSELoss()
    y_train = Ys[train.to(Ys.device)].to(DEVICE)
    norm = (y_train.mean(0), y_train.std(0))          # target standardization (sim_train only)

    run = RUNS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{a.model}" + (f"_{a.name}" if a.name else ""))
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "args": vars(a), "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW + OneCycleLR", "loss": "MSE", "device": str(DEVICE),
        "bf16": AUTOCAST is not contextlib.nullcontext,
        "params": sum(p.numel() for p in model.parameters()),
        "target_norm": {"mean": norm[0].tolist(), "std": norm[1].tolist(), "labels": LABELS},
        "data_placement": placement,
        "model_config": model_config,
        "split": split,
        "split_sizes": {"sim_train": len(train), "sim_val": len(val),
                        "exp_adapt": len(s["exp_adapt"]), "exp_test": len(s["exp_test"])},
    }, indent=2))
    (run / "model.txt").write_text(str(model))
    print(f"model {a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {DEVICE} -> {run}")

    # ---- training
    y_val = Ys[val.to(Ys.device)].cpu().numpy()
    y_adapt = Ye[s["exp_adapt"].to(Ye.device)].cpu().numpy()
    train_eval = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:TRAIN_EVAL_SIZE]]
    y_train_eval = Ys[train_eval.to(Ys.device)].cpu().numpy()
    best, best_ep, rows = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in batches(Xs, Ys, train, BATCH_SIZE, shuffle=True):
            with AUTOCAST():
                loss = loss_fn(model(x).float(), (y - norm[0]) / norm[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(x)   # standardized units

        mt = metrics(predict(model, Xs, Ys, train_eval, norm), y_train_eval)
        mv = metrics(predict(model, Xs, Ys, val, norm), y_val)
        me = metrics(predict(model, Xe, Ye, s["exp_adapt"], norm), y_adapt)
        rows.append({"epoch": ep, "train_loss_std": total / len(train),
                     **{f"train_{k}": v for k, v in mt.items()},
                     **{f"val_{k}": v for k, v in mv.items()}, **{f"exp_{k}": v for k, v in me.items()},
                     "lr": sched.get_last_lr()[0], "seconds": time.time() - t0})
        with open(run / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

        mark = ""
        if mv["MSE"] < best:
            best, best_ep, mark = mv["MSE"], ep, " *"
            torch.save(model.state_dict(), run / "best.pt")
        print(f"ep {ep:3d} | train MSE {mt['MSE']:.5f} | val MSE {mv['MSE']:.5f} "
              f"R² {mv['R2_X']:.3f}/{mv['R2_Y']:.3f}/{mv['R2_Z']:.3f} | "
              f"exp R² {me['R2_X']:.3f}/{me['R2_Y']:.3f}/{me['R2_Z']:.3f} | {time.time() - t0:.0f}s{mark}")

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(run / "best.pt", map_location=DEVICE))
    exp_splits = ["exp_adapt"] + (["exp_test"] if a.final else [])
    evals = [("sim_val", model, Xs, Ys, val)] + [(k, model, Xe, Ye, s[k]) for k in exp_splits]
    ada = adabn(model, Xe, Ye, s["exp_adapt"])    # statistics from exp_adapt images, labels unused
    if ada is not None:
        evals += [(f"{k}_adabn", ada, Xe, Ye, s[k]) for k in exp_splits]

    y_mean = norm[0].cpu().numpy()   # baseline: always predict the training mean
    results = {"best_epoch": best_ep, "last_epoch": a.epochs, "splits": {}}
    for name, m, X, Y, idx in evals:
        pred, true = predict(m, X, Y, idx, norm), Y[idx.to(Y.device)].cpu().numpy()
        results["splits"][name] = {**metrics(pred, true),
                                   "baseline": metrics(np.broadcast_to(y_mean, true.shape), true)}
        scatter(pred, true, name, run / f"pred_{name}.png")
        np.savetxt(run / f"predictions_{name}.csv", np.c_[idx.numpy(), true, pred], delimiter=",",
                   header="index,true_X,true_Y,true_Z,pred_X,pred_Y,pred_Z", comments="",
                   fmt=["%d"] + ["%.6f"] * 6)
    (run / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\nbest checkpoint: epoch {best_ep} of {a.epochs}")
    for name, r in results["splits"].items():
        b = r["baseline"]
        print(f"  {name:16s} R² X/Y/Z {r['R2_X']:7.3f} {r['R2_Y']:7.3f} {r['R2_Z']:7.3f} | "
              f"MAE X/Y/Z {r['MAE_X']:.4f} {r['MAE_Y']:.4f} {r['MAE_Z']:.4f} "
              f"(baseline {b['MAE_X']:.4f} {b['MAE_Y']:.4f} {b['MAE_Z']:.4f})")


if __name__ == "__main__":
    main()