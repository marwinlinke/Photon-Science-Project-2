"""Train a model on the simulation (Task 1) and evaluate it on the experiment (Task 2).

    python src/train.py --cache data/cache/<settings>                    # conv_mixer
    python src/train.py --cache data/cache/<settings> --model conv_mixer --epochs 50 --name long
    python src/train.py --cache data/cache/<settings> --limit 2000 --epochs 1    # quick test
    python src/train.py --cache data/cache/<settings> --final                    # + exp_test, once at the end

--model <name> imports src/models/<name>.py and calls its build().
The best checkpoint is chosen on sim_val only. exp_test is evaluated only with --final.
Targets are standardized with the sim_train mean/std (each of X, Y, Z gets the same weight in the
loss); predictions are converted back, so all metrics and CSVs are in the original label units.

This file: arguments, data, model, training loop, evaluation.
Everything that is written / plotted / printed (run folder contents) is in train_log.py.
"""
import argparse
import contextlib
import copy
import importlib
import time
from pathlib import Path

import torch
import torch.nn as nn

import train_log as log
from dataset import load, split_info

BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
WEIGHT_DECAY = 1e-4
RUNS_DIR = Path("runs")
TRAIN_EVAL_SIZE = 10_000   # fixed sample of sim_train evaluated after every epoch (like sim_val)
GPU_DATA_SHARE = 0.5  # move all data to the GPU if it needs less than this share of the free GPU memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bfloat16 mixed precision only on GPUs with native bf16 (Ampere and newer: A100, H100, RTX 30xx+),
# otherwise float32. including_emulation=False: older GPUs (e.g. Turing, Quadro T2000) would only
# emulate bf16, which is slower than float32.
AUTOCAST = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) \
    else contextlib.nullcontext


# --------------------------------------------------------------------------- arguments
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="data/cache/<settings> with sim.h5 + exp.h5")
    ap.add_argument("--model", default="conv_mixer", help="file name in src/models/ (without .py)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    ap.add_argument("--name", default="", help="optional label for the run folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="also evaluate exp_test (do this once, at the end)")
    ap.add_argument("--limit", type=int, default=0, help="use only N sim images for training and validation")
    return ap.parse_args()


# --------------------------------------------------------------------------- data
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


def labels(Y, idx):
    """Labels of the images idx as a numpy array."""
    return Y[idx.to(Y.device)].cpu().numpy()


# --------------------------------------------------------------------------- training / evaluation
def train_epoch(model, opt, sched, loss_fn, X, Y, idx, norm):
    """One pass over idx; returns the mean loss (standardized units)."""
    model.train()
    total = 0.0
    for x, y in batches(X, Y, idx, BATCH_SIZE, shuffle=True):
        with AUTOCAST():
            loss = loss_fn(model(x).float(), (y - norm[0]) / norm[1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item() * len(x)
    return total / len(idx)


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


@torch.no_grad()
def adabn(model, X, Y, idx):
    """Copy of the model with BatchNorm statistics re-estimated on the given images (no labels used).
    None if the model has no BatchNorm."""
    m = copy.deepcopy(model)
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]   # 1d and 2d
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


# --------------------------------------------------------------------------- main
def main():
    a = parse_args()
    torch.manual_seed(a.seed)

    # ---- data
    t0 = time.time()
    Xs, Ys, Xe, Ye, s = load(a.cache)
    (Xs, Ys, Xe, Ye), placement = place_data([Xs, Ys, Xe, Ye])
    train, val, adapt = s["sim_train"], s["sim_val"], s["exp_adapt"]
    if a.limit:
        train, val = train[:a.limit], val[:a.limit]
    print(f"data: sim {tuple(Xs.shape)}, exp {tuple(Xe.shape)} | train {len(train)}, val {len(val)}, "
          f"exp_adapt {len(adapt)}, exp_test {len(s['exp_test'])} | {time.time() - t0:.0f}s")
    print(f"data on {placement['data_on']} ({placement['data_gb']} GB"
          + (f", GPU {placement['gpu_free_gb']} of {placement['gpu_total_gb']} GB free)" if "gpu_free_gb" in placement else ")"))

    # ---- model, optimiser, target standardization
    module = importlib.import_module(f"models.{a.model}")
    model = module.build().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    loss_fn = nn.MSELoss()
    y_train = Ys[train.to(Ys.device)].to(DEVICE)
    norm = (y_train.mean(0), y_train.std(0))          # sim_train only
    n_params = sum(p.numel() for p in model.parameters())

    run = log.new_run(RUNS_DIR, a.model, a.name)
    log.write_config(run, {
        "args": vars(a), "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW + OneCycleLR", "loss": "MSE", "device": str(DEVICE),
        "bf16": AUTOCAST is not contextlib.nullcontext,
        "params": n_params,
        "target_norm": {"mean": norm[0].tolist(), "std": norm[1].tolist(), "labels": log.LABELS},
        "data_placement": placement,
        "model_config": getattr(module, "CONFIG", {}),   # sizes from the model file, {} if it has none
        "split": split_info(),
        "split_sizes": {"sim_train": len(train), "sim_val": len(val),
                        "exp_adapt": len(adapt), "exp_test": len(s["exp_test"])},
    }, model)
    print(f"model {a.model}: {n_params / 1e6:.2f}M params on {DEVICE} -> {run}")

    # ---- training: best checkpoint on sim_val
    train_eval = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:TRAIN_EVAL_SIZE]]
    y_train_eval, y_val, y_adapt = labels(Ys, train_eval), labels(Ys, val), labels(Ye, adapt)
    best, best_ep, rows = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, opt, sched, loss_fn, Xs, Ys, train, norm)
        mt = log.metrics(predict(model, Xs, Ys, train_eval, norm), y_train_eval)
        mv = log.metrics(predict(model, Xs, Ys, val, norm), y_val)
        me = log.metrics(predict(model, Xe, Ye, adapt, norm), y_adapt)

        is_best = mv["MSE"] < best
        if is_best:
            best, best_ep = mv["MSE"], ep
            torch.save(model.state_dict(), run / "best.pt")
        rows.append(log.epoch_row(ep, train_loss, mt, mv, me, sched.get_last_lr()[0], time.time() - t0))
        log.log_epoch(run, rows, best_ep, is_best)

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(run / "best.pt", map_location=DEVICE))
    exp_splits = ["exp_adapt"] + (["exp_test"] if a.final else [])
    evals = [("sim_val", model, Xs, Ys, val)] + [(k, model, Xe, Ye, s[k]) for k in exp_splits]
    ada = adabn(model, Xe, Ye, adapt)    # statistics from exp_adapt images, labels unused
    if ada is not None:
        evals += [(f"{k}_adabn", ada, Xe, Ye, s[k]) for k in exp_splits]

    y_mean = norm[0].cpu().numpy()   # baseline: always predict the training mean
    results = {"best_epoch": best_ep, "last_epoch": a.epochs, "splits": {}}
    for name, m, X, Y, idx in evals:
        pred, true = predict(m, X, Y, idx, norm), labels(Y, idx)
        results["splits"][name] = {**log.metrics(pred, true), "baseline": log.baseline(y_mean, true)}
        log.save_split(run, name, idx.numpy(), true, pred)
    log.write_results(run, results)


if __name__ == "__main__":
    main()
