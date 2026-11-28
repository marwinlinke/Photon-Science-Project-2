# Compare training runs (marimo notebook): one table of all runs in runs/ with their settings and
# final results, then curves, final metrics and plots for the runs selected in the table.
# Start from the repo root:
#     marimo edit src/report_runs.py     # editable notebook in the browser
#     marimo run src/report_runs.py      # app view only
# Needs: pip install -r requirements-local.txt (local only; the server does not need marimo)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Proj 2 – compare runs")


@app.cell
def _():
    import csv
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    REPO = Path(__file__).resolve().parent.parent
    LABELS = "XYZ"
    return LABELS, Path, REPO, csv, json, mo, plt


@app.cell
def _(REPO, mo):
    runs_dir = mo.ui.text(value=str(REPO / "runs"), label="runs folder", full_width=True)
    reload = mo.ui.run_button(label="reload runs")
    mo.vstack([mo.md("# Compare runs"), mo.hstack([runs_dir, reload], widths=[6, 1])])
    return reload, runs_dir


@app.cell
def _(LABELS, Path, csv, json, reload, runs_dir):
    reload  # re-run this cell when "reload runs" is clicked

    def read_log(path):
        if not path.exists():
            return []
        with open(path) as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

    def load_run(d):
        """config.json (required), metrics.json and log.csv (optional: missing while running / after a crash)."""
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None
        return {"dir": d, "config": cfg, "metrics": met, "log": read_log(d / "log.csv")}

    root = Path(runs_dir.value)
    runs = {d.name: load_run(d) for d in sorted(root.glob("*")) if (d / "config.json").exists()}

    def data_name(cache):
        """Short name of the training data: the cache folder (= all preprocessing settings) or 'original'."""
        name = Path(cache).name
        return "original" if name in ("original", "original_103k_norm16bin") else name

    def row(name, r):
        c, m, log = r["config"], r["metrics"], r["log"]
        a = c["args"]
        out = {
            "run": name,
            "model": a["model"],
            "name": a.get("name", ""),
            "data": data_name(a["cache"]),
            "epochs": a["epochs"],
            "lr": a["lr"],
            "batch": c.get("batch_size"),
            "wd": c.get("weight_decay"),
            "seed": a.get("seed"),
            "limit": a.get("limit") or "",
            "params_M": round(c.get("params", 0) / 1e6, 2),
            "model_config": ", ".join(f"{k}={v}" for k, v in c.get("model_config", {}).items()),
            "device": c.get("device", "") + (" bf16" if c.get("bf16") else ""),
            "split": c.get("split", {}).get("sha256", ""),
            "status": "done" if m else f"incomplete ({len(log)}/{a['epochs']} ep)",
            "best_ep": m["best_epoch"] if m else "",
            "s_per_ep": round(sum(x["seconds"] for x in log) / len(log), 1) if log else "",
        }
        for split, short in (("sim_val", "val"), ("exp_adapt", "exp"), ("exp_adapt_adabn", "exp_adabn"), ("exp_test", "test")):
            s = (m or {}).get("splits", {}).get(split)
            out[f"{short}_MSE"] = round(s["MSE"], 5) if s else ""
            for c_ in LABELS:
                out[f"{short}_R2_{c_}"] = round(s[f"R2_{c_}"], 3) if s else ""
        return out

    rows = [row(n, r) for n, r in runs.items()]
    return root, rows, runs


@app.cell
def _(mo, root, rows):
    only_diff = mo.ui.switch(value=False, label="only columns that differ between runs")
    hide_results = mo.ui.switch(value=False, label="settings only (hide results)")
    mo.stop(not rows, mo.md(f"No runs with a config.json in `{root}`."))
    mo.hstack([only_diff, hide_results], justify="start", gap=2)
    return hide_results, only_diff


@app.cell
def _(hide_results, mo, only_diff, rows):
    _cols = list(rows[0])
    _results = [c for c in _cols if c.split("_")[0] in ("val", "exp", "test") or c in ("best_ep", "s_per_ep", "status")]
    if hide_results.value:
        _cols = [c for c in _cols if c not in _results]
    if only_diff.value and len(rows) > 1:
        _cols = [c for c in _cols if c == "run" or len({str(r[c]) for r in rows}) > 1]
    table = mo.ui.table(
        [{c: r[c] for c in _cols} for r in rows],
        selection="multi",
        page_size=25,
        freeze_columns_left=["run"],
        label=f"{len(rows)} runs – select rows to compare below (none selected = all)",
    )
    table
    return (table,)


@app.cell
def _(runs, table):
    selected = [r["run"] for r in table.value] or list(runs)
    colors = {n: f"C{i % 10}" for i, n in enumerate(selected)}   # same colour per run in every plot
    return colors, selected


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        ["MSE", "MSE_X", "MSE_Y", "MSE_Z", "MAE_X", "MAE_Y", "MAE_Z", "R2_X", "R2_Y", "R2_Z"],
        value="MSE", label="metric")
    splits = mo.ui.multiselect(["train", "val", "exp"], value=["val", "exp"], label="splits")
    log_y = mo.ui.switch(value=True, label="log y")
    mo.vstack([mo.md("## Curves over epochs"), mo.hstack([metric, splits, log_y], justify="start", gap=2)])
    return log_y, metric, splits


@app.cell
def _(colors, log_y, metric, plt, runs, selected, splits):
    _sp = splits.value or ["val"]
    _names = {"train": "train (fixed sample)", "val": "sim_val", "exp": "exp_adapt"}
    _fig, _ax = plt.subplots(1, len(_sp), figsize=(6 * len(_sp), 4), squeeze=False)
    for _a, _s in zip(_ax[0], _sp):
        for _n in selected:
            _log = runs[_n]["log"]
            _k = f"{_s}_{metric.value}"
            if _log and _k in _log[0]:
                _a.plot([r["epoch"] for r in _log], [r[_k] for r in _log], label=_n, lw=1.3, color=colors[_n])
        if log_y.value and not metric.value.startswith("R2"):
            _a.set_yscale("log")
        _a.set_title(f"{_names[_s]}: {metric.value}")
        _a.set_xlabel("epoch")
    _ax[0][0].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(LABELS, colors, mo, plt, runs, selected):
    # final metrics of the best checkpoint, grouped by split, one bar per run
    _splits = ["sim_val", "exp_adapt", "exp_adapt_adabn", "exp_test"]
    _done = [n for n in selected if runs[n]["metrics"]]
    _present = [s for s in _splits if any(s in runs[n]["metrics"]["splits"] for n in _done)]
    mo.stop(not _done, mo.md("_No finished runs selected (metrics.json missing)._"))
    _fig, _ax = plt.subplots(1, 4, figsize=(20, 4))
    _w = 0.8 / len(_done)
    for _j, _key in enumerate(["MSE"] + [f"R2_{c}" for c in LABELS]):
        for _i, _n in enumerate(_done):
            _sp = runs[_n]["metrics"]["splits"]
            _vals = [_sp[s][_key] if s in _sp else float("nan") for s in _present]
            _ax[_j].bar([k + _i * _w for k in range(len(_present))], _vals, _w, label=_n, color=colors[_n])
        if _key == "MSE":   # baseline: always predicting the sim_train mean
            _b = runs[_done[0]]["metrics"]["splits"]
            _ax[_j].scatter([k + 0.4 - _w / 2 for k in range(len(_present))],
                            [_b[s]["baseline"]["MSE"] if s in _b else float("nan") for s in _present],
                            marker="_", s=600, color="k", label="baseline (mean)", zorder=3)
        else:
            _ax[_j].axhline(0, color="k", lw=0.8)
        _ax[_j].set_xticks([k + 0.4 - _w / 2 for k in range(len(_present))], _present, rotation=15, fontsize=8)
        _ax[_j].set_title(_key)
    _ax[0].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([mo.md("## Final metrics (best checkpoint)"), _fig])
    return


@app.cell
def _(mo, runs, selected):
    # the plots each run saved itself
    _files = [("loss", "loss.png"), ("sim_val", "pred_sim_val.png"), ("exp_adapt", "pred_exp_adapt.png"),
              ("exp_adapt AdaBN", "pred_exp_adapt_adabn.png"), ("exp_test", "pred_exp_test.png")]

    def _tabs(d):
        t = {k: mo.image(str(d / f), width="100%") for k, f in _files if (d / f).exists()}
        return mo.ui.tabs(t) if t else mo.md("_no plots yet_")

    mo.vstack([mo.md("## Plots per run"),
               mo.accordion({n: _tabs(runs[n]["dir"]) for n in selected}, multiple=True)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
