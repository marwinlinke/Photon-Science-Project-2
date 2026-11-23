import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Proj 2 – preprocessing tuning")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO / "src"))
    from preprocess import (DEFAULTS, cache_ok, convert, domain_gap, estimate_background,
                            output_paths, preprocess, run_name)

    return (
        DEFAULTS,
        REPO,
        cache_ok,
        convert,
        domain_gap,
        estimate_background,
        h5py,
        json,
        mo,
        np,
        output_paths,
        plt,
        preprocess,
        run_name,
    )


@app.cell
def _(REPO, mo):
    mo.md("# Preprocessing tuning\n"
          "1. move the sliders: images update live · "
          "2. **Compute metrics**: domain gap + XYZ information, logged as a table · "
          "3. **Write to cache**: full preprocessed files for training")
    sim_path = mo.ui.text(str(REPO / "data/project2_simulation_103k_norm16bin.h5"), label="simulation h5", full_width=True)
    exp_path = mo.ui.text(str(REPO / "data/project2_experiment_norm16bin.h5"), label="experiment h5", full_width=True)
    out_dir = mo.ui.text(str(REPO / "data"), label="cache folder", full_width=True)
    n_blocks = mo.ui.slider(8, 64, value=40, step=8, label="sim subset (blocks of 128 images)", show_value=True)
    mo.vstack([sim_path, exp_path, out_dir, n_blocks])
    return exp_path, n_blocks, out_dir, sim_path


@app.cell
def _(exp_path, h5py, mo, n_blocks, np, sim_path):
    def load_sim_subset(path, n_blocks, block=128, seed=0):
        # The sim file is chunked in blocks of 128 images; reading whole random blocks
        # touches only those chunks (fast), random single images would read almost the whole file.
        with h5py.File(path, "r") as f:
            N = f["X"].shape[0]
            starts = np.sort(np.random.default_rng(seed).choice(N // block, min(n_blocks, N // block), replace=False)) * block
            X = np.concatenate([f["X"][s:s + block] for s in starts]).astype(np.float32)
            Y = np.concatenate([f["Y"][s:s + block] for s in starts]).astype(np.float32)
        return X, Y

    with mo.status.spinner("loading data ..."):
        Xs_raw, Ys = load_sim_subset(sim_path.value, n_blocks.value)
        with h5py.File(exp_path.value, "r") as _f:
            Xe_raw, Ye = _f["X"][:].astype(np.float32), _f["Y"][:].astype(np.float32)
    mo.md(f"loaded **sim** {Xs_raw.shape}, **exp** {Xe_raw.shape}")
    return Xe_raw, Xs_raw, Ye, Ys


@app.cell
def _(mo):
    mo.md("""
    ## Data overview (raw data)
    Short version of the exploration notebook, on the loaded subset. Key findings so far:
    sim = sparse ray hits (9 discrete levels, ~70% zeros); exp = smooth, with a strong background
    gradient and no zeros; exp labels lie inside the sim range; X/Y are a scan grid, Z has 3 main levels.
    The image comparison (mean images, profiles, total intensity, PCA) is further down, original vs. preprocessed.
    """)
    return


@app.cell
def _(Xe_raw, Xs_raw, Ye, Ys, mo, np):
    def _summ(name, X, Y):
        flat = X.reshape(len(X), -1)
        return {
            "data": name, "images": len(X), "shape": f"{X.shape[1]}×{X.shape[2]}",
            "pixel min / max": f"{X.min():.3f} / {X.max():.3f}",
            "pixel mean": round(float(X.mean()), 3),
            "zeros": f"{np.mean(X == 0):.1%}",
            "distinct values (1st img)": int(len(np.unique(X[0]))),
            "total intensity (median)": int(np.median(flat.sum(1))),
            **{f"{c} range": f"{Y[:, j].min():.3f} – {Y[:, j].max():.3f}" for j, c in enumerate("XYZ")},
            **{f"{c} unique": int(len(np.unique(Y[:, j]))) for j, c in enumerate("XYZ")},
        }
    mo.ui.table([_summ("sim (subset)", Xs_raw, Ys), _summ("exp", Xe_raw, Ye)], selection=None)
    return


@app.cell
def _(Ye, Ys, np, plt):
    _fig, _ax = plt.subplots(1, 5, figsize=(22, 3.6))
    for _j, _c in enumerate("XYZ"):
        _b = np.histogram_bin_edges(np.r_[Ys[:, _j], Ye[:, _j]], 50)
        _ax[_j].hist(Ys[:, _j], _b, density=True, alpha=.6, label="sim")
        _ax[_j].hist(Ye[:, _j], _b, density=True, alpha=.6, label="exp")
        _ax[_j].set_title(f"label {_c}"); _ax[_j].legend(fontsize=7)
    _ax[3].scatter(Ys[:, 0], Ys[:, 1], s=1, alpha=.3, label="sim")
    _ax[3].scatter(Ye[:, 0], Ye[:, 1], s=4, c="C1", label="exp")
    _ax[3].set_xlabel("X"); _ax[3].set_ylabel("Y"); _ax[3].set_title("X–Y: exp is a scan grid"); _ax[3].legend(fontsize=7)
    _ax[4].scatter(Ys[:, 0], Ys[:, 2], s=1, alpha=.3, label="sim")
    _ax[4].scatter(Ye[:, 0], Ye[:, 2], s=4, c="C1", label="exp")
    _ax[4].set_xlabel("X"); _ax[4].set_ylabel("Z"); _ax[4].set_title("X–Z: exp has ~3 Z levels"); _ax[4].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Settings
    """)
    return


@app.cell
def _(DEFAULTS, mo):
    d = DEFAULTS
    bg_on = mo.ui.checkbox(value=d["bg_quantile"] is not None, label="background template (exp)")
    bg_q = mo.ui.slider(1, 30, value=d["bg_quantile"] or 5, step=1, label="percentile", show_value=True)
    _crop = d["crop"] or (0, 128, 0, 128)
    crop_on = mo.ui.checkbox(value=d["crop"] is not None, label="crop")
    r0 = mo.ui.number(0, 64, value=_crop[0], label="row0")
    r1 = mo.ui.number(64, 128, value=_crop[1], label="row1")
    c0 = mo.ui.number(0, 64, value=_crop[2], label="col0")
    c1 = mo.ui.number(64, 128, value=_crop[3], label="col1")
    off_on = mo.ui.checkbox(value=d["offset_pct"] is not None, label="per-image offset")
    off_pct = mo.ui.slider(0, 20, value=d["offset_pct"] or 1, step=0.5, label="percentile", show_value=True)
    sqrt = mo.ui.checkbox(value=d["sqrt"], label="sqrt")
    blur = mo.ui.slider(0, 4, value=d["blur_sigma"], step=0.25, label="blur σ", show_value=True)
    ds = mo.ui.dropdown(["1", "2", "4"], value=str(d["downsample"]), label="downsample")
    sc_on = mo.ui.checkbox(value=d["scale_pct"] is not None, label="scaling")
    sc_hi = mo.ui.slider(90, 100, value=d["scale_pct"] or 99.5, step=0.1, label="divide by pct", show_value=True)
    min_range = mo.ui.number(0.0, 0.5, value=d["min_range"], step=0.001, label="min_range")
    dc_on = mo.ui.checkbox(value=bool(d["dark_cut"]), label="dark cutoff")
    dc_t = mo.ui.slider(0.01, 0.5, value=d["dark_cut"] or 0.15, step=0.01, label="threshold t", show_value=True)

    mo.hstack([
        mo.vstack([mo.md("**1. background**"), bg_on, bg_q]),
        mo.vstack([mo.md("**2. crop**"), crop_on, mo.hstack([r0, r1]), mo.hstack([c0, c1])]),
        mo.vstack([mo.md("**3./4. offset (dark level), sqrt**"), off_on, off_pct, sqrt]),
        mo.vstack([mo.md("**5./6. blur, downsample**"), blur, ds]),
        mo.vstack([mo.md("**7. scaling (brightness)**"), sc_on, sc_hi, min_range]),
        mo.vstack([mo.md("**8. dark cutoff (after scaling)**"), dc_on, dc_t]),
    ], justify="start", gap=2)
    return (
        bg_on,
        bg_q,
        blur,
        c0,
        c1,
        crop_on,
        dc_on,
        dc_t,
        ds,
        min_range,
        off_on,
        off_pct,
        r0,
        r1,
        sc_hi,
        sc_on,
        sqrt,
    )


@app.cell
def _(
    bg_on,
    bg_q,
    blur,
    c0,
    c1,
    crop_on,
    dc_on,
    dc_t,
    ds,
    min_range,
    mo,
    off_on,
    off_pct,
    r0,
    r1,
    run_name,
    sc_hi,
    sc_on,
    sqrt,
):
    p = dict(
        bg_quantile=float(bg_q.value) if bg_on.value else None,
        crop=(int(r0.value), int(r1.value), int(c0.value), int(c1.value)) if crop_on.value else None,
        offset_pct=float(off_pct.value) if off_on.value else None,
        sqrt=bool(sqrt.value),
        blur_sigma=float(blur.value),
        downsample=int(ds.value),
        scale_pct=float(sc_hi.value) if sc_on.value else None,
        min_range=float(min_range.value),
        dark_cut=float(dc_t.value) if dc_on.value else None,
    )
    name = run_name(p)
    cmd = ("python src/preprocess.py"
           f" --bg-quantile {p['bg_quantile'] if p['bg_quantile'] is not None else -1:g}"
           f" --crop {' '.join(map(str, p['crop'])) if p['crop'] else '0 0 0 0'}"
           f" --offset-pct {p['offset_pct'] if p['offset_pct'] is not None else -1:g}"
           f"{' --sqrt' if p['sqrt'] else ''}"
           f" --blur {p['blur_sigma']:g} --downsample {p['downsample']}"
           f" --scale {p['scale_pct'] if p['scale_pct'] is not None else -1:g}"
           f" --min-range {p['min_range']:g}"
           f" --dark-cut {p['dark_cut'] if p['dark_cut'] else -1:g}")
    mo.md(f"**name:** `{name}`\n\n**same conversion from the command line:**\n```bash\n{cmd}\n```")
    return name, p


@app.cell
def _(Xe_raw, estimate_background, p):
    # recomputed only when the background percentile changes; uses ALL exp images,
    # exactly like the command line and the cache writer below
    bg = estimate_background(Xe_raw, p["bg_quantile"]) if p["bg_quantile"] is not None else None
    return (bg,)


@app.cell
def _(Xe_raw, Xs_raw, bg, mo, np, p, preprocess):
    with mo.status.spinner("preprocessing ..."):
        Xs = preprocess(Xs_raw, p)
        Xe = preprocess(Xe_raw, p, bg)
    mo.md(f"output **{Xs.shape[1]}×{Xs.shape[2]}** | sim mean {Xs.mean():.3f}, zeros {np.mean(Xs == 0):.1%} "
          f"| exp mean {Xe.mean():.3f}, zeros {np.mean(Xe == 0):.1%}")
    return Xe, Xs


@app.cell
def _(mo):
    seed = mo.ui.number(0, 9999, value=0, label="example seed")
    n_ex = mo.ui.slider(2, 6, value=4, label="examples", show_value=True)
    mo.hstack([seed, n_ex], justify="start")
    return n_ex, seed


@app.cell
def _(Xe, Xe_raw, Xs, Xs_raw, Ye, Ys, n_ex, np, plt, seed):
    _r = np.random.default_rng(seed.value)
    _n = n_ex.value
    _si, _ei = _r.choice(len(Xs), _n, replace=False), _r.choice(len(Xe), _n, replace=False)
    _rows = [("sim raw", Xs_raw, _si, Ys), ("sim pp", Xs, _si, Ys),
             ("exp raw", Xe_raw, _ei, Ye), ("exp pp", Xe, _ei, Ye)]
    _fig, _ax = plt.subplots(4, _n + 1, figsize=(2.8 * (_n + 1), 10.5))
    for _k, (_nm, _X, _idx, _Y) in enumerate(_rows):
        for _c, _i in enumerate(_idx):
            _ax[_k, _c].imshow(_X[_i], cmap="inferno", vmin=0, vmax=1)
            _ax[_k, _c].set_title(f"{_nm} #{_i}\n{np.array2string(_Y[_i], precision=2)}", fontsize=8)
            _ax[_k, _c].axis("off")
        _ax[_k, _n].hist(_X.ravel(), 100, density=True, log=True, color="C0" if "sim" in _nm else "C1")
        _ax[_k, _n].set_title(f"{_nm} pixels", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Xe_raw, Xs_raw, np):
    # raw side of the comparison: computed once, not on every slider change
    def summary(Xs_, Xe_, n_pca=2000, seed=0):
        from sklearn.decomposition import PCA
        n = min(len(Xs_), n_pca)
        A = Xs_[np.random.default_rng(seed).permutation(len(Xs_))[:n]]
        pca = PCA(2, random_state=seed).fit(A.reshape(n, -1))
        return {
            "mean_s": Xs_.mean(0), "mean_e": Xe_.mean(0),
            "prof_s": Xs_.mean((0, 2)), "prof_e": Xe_.mean((0, 2)),
            "tot_s": Xs_.reshape(len(Xs_), -1).sum(1), "tot_e": Xe_.reshape(len(Xe_), -1).sum(1),
            "pca_s": pca.transform(A.reshape(n, -1)), "pca_e": pca.transform(Xe_.reshape(len(Xe_), -1)),
        }

    raw_summary = summary(Xs_raw, Xe_raw)
    return raw_summary, summary


@app.cell
def _(Xe, Xs, mo, np, plt, raw_summary, summary):
    mo.md("### Original vs. preprocessed: simulation, experiment and their difference")
    _pp = summary(Xs, Xe)
    _fig, _ax = plt.subplots(2, 6, figsize=(26, 8))
    # same colour scale for both difference images, so the reduction is visible directly
    _vmax = max(float(np.abs(_S["mean_e"] - _S["mean_s"]).max())
                for _S in (raw_summary, _pp) if _S["mean_s"].shape == _S["mean_e"].shape)
    for _r, (_lbl, _S) in enumerate([("original", raw_summary), ("preprocessed", _pp)]):
        _ax[_r, 0].imshow(_S["mean_s"], cmap="inferno"); _ax[_r, 0].set_title(f"{_lbl}: sim mean")
        _ax[_r, 1].imshow(_S["mean_e"], cmap="inferno"); _ax[_r, 1].set_title(f"{_lbl}: exp mean")
        if _S["mean_s"].shape == _S["mean_e"].shape:
            _d = _S["mean_e"] - _S["mean_s"]
            _im = _ax[_r, 2].imshow(_d, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            plt.colorbar(_im, ax=_ax[_r, 2], fraction=0.046)
        _ax[_r, 2].set_title(f"{_lbl}: exp − sim (same scale)")
        _ax[_r, 3].plot(_S["prof_s"], label="sim"); _ax[_r, 3].plot(_S["prof_e"], label="exp")
        _ax[_r, 3].set_title(f"{_lbl}: row profile (top → bottom)"); _ax[_r, 3].legend(fontsize=7)
        _b = np.histogram_bin_edges(np.r_[_S["tot_s"], _S["tot_e"]], 60)
        _ax[_r, 4].hist(_S["tot_s"], _b, density=True, alpha=.6, label="sim")
        _ax[_r, 4].hist(_S["tot_e"], _b, density=True, alpha=.6, label="exp")
        _ax[_r, 4].set_title(f"{_lbl}: total intensity per image"); _ax[_r, 4].legend(fontsize=7)
        _ax[_r, 5].scatter(_S["pca_s"][:, 0], _S["pca_s"][:, 1], s=2, alpha=.3, label="sim")
        _ax[_r, 5].scatter(_S["pca_e"][:, 0], _S["pca_e"][:, 1], s=3, alpha=.5, c="C1", label="exp")
        _ax[_r, 5].set_title(f"{_lbl}: PCA (fit on sim)"); _ax[_r, 5].legend(fontsize=7)
        for _a in _ax[_r, :3]:
            _a.axis("off")
    _fig.tight_layout()
    mo.vstack([mo.md("Top row original, bottom row with the current settings. "
                     "Goal: the exp − sim difference fades, the histograms overlap, the PCA clouds merge."), _fig])
    return


@app.cell
def _(bg, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 3.6))
    if bg is not None:
        _im = _ax.imshow(bg, cmap="viridis"); plt.colorbar(_im, ax=_ax)
        _ax.set_title("exp background template\n(should be smooth, no beam structures)", fontsize=9)
    else:
        _ax.set_title("background template off")
    _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metrics
    **domain AUC**: linear sim-vs-exp classifier on PCA features (0.5 = no gap, 1.0 = trivially separable) → should go **down**.
    **R² sim**: gradient boosting trained on sim, tested on held-out sim blocks → should stay **high** (information about XYZ survives).
    **R² exp**: same model applied to exp. Only a hint, do **not** tune on it (that is tuning on the test set).
    Each value is **mean ± std over several repeats** (different sim/exp subsamples and train/test block splits).
    """)
    return


@app.cell
def _(mo):
    feat_kind = mo.ui.dropdown(["profiles + PCA20", "profiles", "PCA50"], value="profiles + PCA20",
                               label="regression features")
    repeats = mo.ui.slider(1, 5, value=3, label="repeats", show_value=True)
    run = mo.ui.run_button(label="Compute metrics")
    get_log, set_log = mo.state([])
    mo.hstack([feat_kind, repeats, run], justify="start", gap=2)
    return feat_kind, get_log, repeats, run, set_log


@app.cell
def _(domain_gap, np):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score

    def profile_features(X):
        # mean intensity per row and per column: band/edge positions appear directly as
        # steps in these curves, which tree models read much better than raw-pixel PCA
        return np.c_[X.mean(2), X.mean(1)]

    def evaluate(Xs, Ys, Xe, Ye, kind, repeats, block=128):
        blocks = np.arange(len(Xs)) // block
        ub = np.unique(blocks)
        res = {"domain_AUC": []}
        for c in "XYZ":
            res[f"R2_sim_{c}"], res[f"R2_exp_{c}"] = [], []

        for r in range(repeats):
            rng = np.random.default_rng(r)
            # domain gap on a fresh random subsample each repeat
            n = min(len(Xs), len(Xe))
            res["domain_AUC"].append(domain_gap(Xs[rng.permutation(len(Xs))[:n]],
                                                Xe[rng.permutation(len(Xe))[:n]], seed=r))
            # block-wise split: images of one 128-block never end up in train and test
            te = np.isin(blocks, rng.choice(ub, max(1, len(ub) // 5), replace=False))
            fs, fe = [], []
            if kind in ("profiles", "profiles + PCA20"):
                fs.append(profile_features(Xs)); fe.append(profile_features(Xe))
            if kind in ("PCA50", "profiles + PCA20"):
                k = 50 if kind == "PCA50" else 20
                pca = PCA(k, random_state=r).fit(Xs[~te].reshape((~te).sum(), -1))
                fs.append(pca.transform(Xs.reshape(len(Xs), -1)))
                fe.append(pca.transform(Xe.reshape(len(Xe), -1)))
            Fs, Fe = np.hstack(fs), np.hstack(fe)
            for j, c in enumerate("XYZ"):
                m = HistGradientBoostingRegressor(max_iter=300, random_state=r).fit(Fs[~te], Ys[~te, j])
                res[f"R2_sim_{c}"].append(r2_score(Ys[te, j], m.predict(Fs[te])))
                res[f"R2_exp_{c}"].append(r2_score(Ye[:, j], m.predict(Fe)))
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}

    return (evaluate,)


@app.cell
def _(
    Xe,
    Xs,
    Ye,
    Ys,
    evaluate,
    feat_kind,
    get_log,
    json,
    mo,
    name,
    p,
    repeats,
    run,
    set_log,
):
    mo.stop(not run.value, mo.md("_Click **Compute metrics** to evaluate the current settings._"))

    with mo.status.spinner(f"{repeats.value} repeats: domain gap + regression ..."):
        _m = evaluate(Xs, Ys, Xe, Ye, feat_kind.value, repeats.value)

    # which settings differ from the previous logged run?
    _prev = get_log()[-1]["_params"] if get_log() else None
    if _prev is None:
        _changed = "(first run)"
    else:
        _pp = json.loads(_prev)
        _changed = ", ".join(f"{k}: {_pp.get(k)} → {v}" for k, v in p.items()
                             if json.dumps(_pp.get(k)) != json.dumps(list(v) if isinstance(v, tuple) else v)) or "(nothing)"

    _row = {"changed vs previous": _changed, "features": feat_kind.value, "repeats": repeats.value}
    for _k, (_mu, _sd) in _m.items():
        _row[_k] = f"{_mu:.3f} ± {_sd:.3f}"
    _row["name"] = name
    _row["_params"] = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()})
    set_log(lambda log: log + [_row])

    mo.md(f"**domain AUC {_row['domain_AUC']}** · "
          f"R² sim X/Y/Z: {_row['R2_sim_X']} / {_row['R2_sim_Y']} / {_row['R2_sim_Z']} · "
          f"R² exp X/Y/Z: {_row['R2_exp_X']} / {_row['R2_exp_Y']} / {_row['R2_exp_Z']}")
    return


@app.cell
def _(get_log, mo):
    _log = [{k: v for k, v in r.items() if not k.startswith("_")} for r in get_log()]
    mo.vstack([
        mo.md("### Results log\nChange **one** setting per run; the first column shows what changed. "
              "A difference only counts if it is clearly larger than the ± values. "
              "The table can be downloaded as CSV."),
        mo.ui.table(_log, selection=None, page_size=20) if _log else mo.md("_no runs yet_"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Write to cache
    Converts the **complete** simulation (~103k) and experiment files with the current settings.
    Files are named after the settings; if a file with exactly these settings already exists it is skipped.
    The simulation takes a few minutes and needs ~2 GB RAM.
    """)
    return


@app.cell
def _(cache_ok, mo, out_dir, output_paths, p):
    dst_exp, dst_sim = output_paths(out_dir.value, p)
    force = mo.ui.checkbox(label="force recompute (overwrite existing cache)")
    write = mo.ui.run_button(label="Write full preprocessed files to cache")
    _st = lambda d: "✅ cached" if cache_ok(d, p) else "— not yet written"
    mo.vstack([
        mo.md(f"- `{dst_exp.name}`: {_st(dst_exp)}\n- `{dst_sim.name}`: {_st(dst_sim)}"),
        mo.hstack([write, force], justify="start", gap=2),
    ])
    return dst_exp, dst_sim, force, write


@app.cell
def _(
    bg,
    cache_ok,
    convert,
    dst_exp,
    dst_sim,
    exp_path,
    force,
    h5py,
    mo,
    p,
    sim_path,
    write,
):
    mo.stop(not write.value)
    dst_exp.parent.mkdir(parents=True, exist_ok=True)
    _done = []
    for _dst, _src, _b in ((dst_exp, exp_path.value, bg), (dst_sim, sim_path.value, None)):
        if cache_ok(_dst, p) and not force.value:
            _done.append(f"- `{_dst.name}`: already cached, skipped")
            continue
        with h5py.File(_src, "r") as _f:
            _N = _f["X"].shape[0]
        with mo.status.progress_bar(total=_N, title=f"writing {_dst.name}") as _bar:
            convert(_src, _dst, p, _b, progress=lambda k: _bar.update(increment=k))
        _done.append(f"- `{_dst.name}`: written ({_N} images)")
    mo.md("**Cache ready:**\n" + "\n".join(_done) +
          f"\n\nLoad in training with `h5py.File('{dst_sim}')`; settings are in `f.attrs['params']`.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
