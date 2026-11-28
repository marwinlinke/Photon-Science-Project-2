# Photon-Science-Project-2

Inverse spectrometer: a network maps a detector image to the source position X, Y, Z.
It is trained on simulation (RAYX, ~103k images) and evaluated on the experiment (~1.6k images).

Tasks: (1) train Image → XYZ and evaluate on simulation, (2) evaluate on the experiment,
(3) improve with data augmentation, (4) improve with domain adaptation.

All commands are run **from the repo root** (paths like `data/` and `runs/` are relative to it).

## Setup

```
pip install -r requirements.txt          # preprocessing + training (server and local)
pip install -r requirements-local.txt    # + marimo tuning app (local)
```

Caches can be built on the server (`preprocess.py` needs only `requirements.txt`), so the
3.4 GB cache files do not have to be copied; only the tuning app is local.

## Data

Copy the raw files from the server into `data/original_103k_norm16bin/`:

```
mkdir -p data/original_103k_norm16bin
cp /vol/tmp/feueforp/project2_simulation_103k_norm16bin.h5 data/original_103k_norm16bin/sim.h5
cp /vol/tmp/feueforp/project2_experiment_norm16bin.h5     data/original_103k_norm16bin/exp.h5
```

Each file holds `X` (images, 128×128) and `Y` (labels X, Y, Z). `data/` is not in git,
except `data/splits.json` (see below).

## 1. Preprocessing

`src/preprocess.py` applies the same deterministic steps to simulation and experiment
(background template, offset, sqrt, blur, scaling, dark cut, …; see its docstring) and
writes a cache folder named after all settings:

```
python src/preprocess.py --bg-quantile 1 --offset-pct 15 --sqrt --sim-blur 0.75 --blur 0 \
    --downsample 1 --scale 98 --min-range 0.05 --dark-cut 0.2
```

→ `data/cache/bg1_crop-off_off15_sqrt1_sb0.75_blur0_ds1_sc98_mr0.05_dc0.2_fx-off/`
with `sim.h5`, `exp.h5`, `stats.md` / `stats.json` and `figures/`. 
An existing cache with the same settings is reused (`--force` recomputes);
`--no-write --preview` only shows examples and the domain-gap check.

To tune the settings interactively (live images, domain-gap and XYZ metrics, write cache;
needs `requirements-local.txt`):

```
marimo edit src/data_tuning.py
```

## 2. Training

```
python src/train.py --cache data/cache/<settings>                          # conv_mixer, 30 epochs
python src/train.py --cache data/cache/<settings> --epochs 50 --name long
python src/train.py --cache data/cache/<settings> --limit 2000 --epochs 1  # quick test
python src/train.py --cache data/cache/<settings> --final                  # + exp_test, only at the end
```

- `--model <name>` loads `src/models/<name>.py` and calls its `build()`; sizes are in its `CONFIG`.
  New models go into `src/models/` and need `build()`, `CONFIG` and `features(x)` (activations
  before the last layer, for t-SNE / domain adaptation). The fc models have a fixed input size
  (`in_size` in their `CONFIG`, default 128): change it when training on a downsampled cache.

  | model | input | params (128×128) |
  |---|---|---|
  | `conv_mixer` (default) | image, ConvMixer (patch embedding + depthwise/pointwise conv blocks) | 0.7M |
  | `fc` | all pixels, fully connected 2048-1024-256 (best model of Project 1) | 36M |
  | `fc_profiles` | mean per row and per column (H + W values), same fully connected layers | 2.9M |
  | `cnn_coord` | image + x/y coordinate maps (CoordConv), 4 conv blocks, flattened 8×8 grid + MLP; keeps position information | 1.1M |
- Loss: MSE on standardized targets (sim_train mean/std); all metrics are in label units.
- The best checkpoint is chosen on `sim_val` only. The experiment is evaluated after every epoch
  (`exp_adapt`) for monitoring, and once more with AdaBN (BatchNorm statistics re-estimated on
  the unlabeled `exp_adapt` images).

Each run writes `runs/<date-time>_<model>[_<name>]/` (not in git):

| file | content |
|---|---|
| `config.json` | all arguments and settings, cache, split hash, model CONFIG, target normalization |
| `model.txt` | the model layer by layer |
| `log.csv` | one row per epoch: train / val / exp metrics (incl. MSE per axis), learning rate |
| `loss.png` | MSE over epochs (mean, X, Y, Z) for train / sim_val / exp_adapt, updated every epoch |
| `best.pt` | best checkpoint (lowest sim_val MSE) |
| `metrics.json` | final R² / MAE / MSE per split (incl. AdaBN), each with a mean-prediction baseline |
| `predictions_<split>.csv`, `pred_<split>.png` | true vs. predicted X/Y/Z |

## Splits

`data/splits.json` is created on first use by `src/dataset.py` and **committed**, so everyone
trains and evaluates on exactly the same images (check: `python src/dataset.py data/cache/<settings>`).

| split | data | how | use |
|---|---|---|---|
| `sim_train` / `sim_val` | simulation | random, 90 / 10 % | training / model selection |
| `exp_adapt` | experiment | whole 0.1 × 0.1 tiles of the X-Y plane, ~30 % | monitoring, AdaBN, domain adaptation (labels not used for training) |
| `exp_test` | experiment | the other tiles | final evaluation only (`--final`) |

Tiles keep neighbouring scan positions (near-identical images) on one side. Results with a different
`splits.json` are not comparable; each run records the file's hash in `config.json`.

## Layout

```
src/
  preprocess.py      preprocessing steps + cache writer (CLI)
  preprocess_log.py  statistics, figures and index.csv for each cache
  data_tuning.py     marimo app for tuning the preprocessing
  dataset.py         loads a cache + the fixed splits
  train.py           arguments, training loop and evaluation
  train_log.py       everything train.py writes: run folder, log.csv, loss.png, metrics, plots
  models/            one file per model, each with build(), CONFIG and features()
  visualizer.py      export sample images
data/                raw data, caches, splits.json
runs/                training runs
samples/             exported sample images
```
