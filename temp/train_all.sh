#!/bin/bash
# Train several models one after another on one prepared dataset (training only: the data must
# already exist, see below).
#
#   cd Photon-Science-Project-2                                   # repo root, with .venv (see README)
#   sbatch temp/train_all.sh                                      # all four models, DATA=flip
#   sbatch temp/train_all.sh fc cnn_coord                         # only these models
#   sbatch --export=ALL,DATA=preprocessed1 temp/train_all.sh      # other data (see DATA below)
#   squeue -u $USER;  tail -f slurm-train_all-<jobid>.out
#
# DATA (each run writes runs/<date-time>_<model>_<DATA>/; same images, labels and splits in all):
#   flip           original data, simulation mirrored left-right, nothing else. Build once with
#                    python src/preprocess.py --sim /vol/tmp/feueforp/project2_simulation_103k_norm16bin.h5
#                      --exp /vol/tmp/feueforp/project2_experiment_norm16bin.h5 --bg-quantile -1
#                      --offset-pct -1 --sim-blur 0 --blur 0 --downsample 1 --scale -1 --dark-cut -1 --flip-x sim
#   original       the raw files. Set up once with
#                    mkdir -p data/original
#                    ln -s /vol/tmp/feueforp/project2_simulation_103k_norm16bin.h5 data/original/sim.h5
#                    ln -s /vol/tmp/feueforp/project2_experiment_norm16bin.h5 data/original/exp.h5
#   preprocessed1  full preprocessing + flip (preprocess.py with --bg-quantile 1 --offset-pct 3
#                    --sim-blur 0.75 --blur 0 --downsample 1 --scale 98.5 --min-range 0.05
#                    --dark-cut 0.3 --flip-x sim)
#
# Expected runtime on an A100 (100 epochs): conv_mixer ~50 min, cnn_coord ~30 min, fc and
# fc_profiles ~3 min each, plus ~3 min data loading per model.
# --dependency=singleton: at most one train_all job of yours runs at a time.
#SBATCH --job-name=train_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --dependency=singleton
#SBATCH --output=slurm-%x-%j.out

EPOCHS=100
MODELS="${*:-conv_mixer fc fc_profiles cnn_coord}"   # models given to sbatch, default all four
DATA="${DATA:-flip}"

case "$DATA" in
    original)      CACHE=data/original ;;
    flip)          CACHE=data/cache/bg-off_crop-off_off-off_sqrt0_sb0_blur0_ds1_sc-off_mr0.001_dc-off_fxsim ;;
    preprocessed1) CACHE=data/cache/bg1_crop-off_off3_sqrt0_sb0.75_blur0_ds1_sc98.5_mr0.05_dc0.3_fxsim ;;
    *)             echo "unknown DATA=$DATA (original, flip, preprocessed1)"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1   # print each line immediately (otherwise the .out log fills only in chunks)

echo "job $SLURM_JOB_ID on $(hostname), $(date) | DATA=$DATA | models: $MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

[ -r "$CACHE/sim.h5" ] && [ -r "$CACHE/exp.h5" ] || {
    echo "no data in $CACHE: build it first (command in the header of this script)"; exit 1; }

# ---- training; a failed model does not stop the others
failed=""
for m in $MODELS; do
    echo; echo "===== $m ($EPOCHS epochs, DATA=$DATA), $(date)"
    python src/train.py --cache "$CACHE" --model "$m" --epochs "$EPOCHS" --name "$DATA" || failed="$failed $m"
done

echo; echo "done $(date)"
[ -z "$failed" ] || { echo "FAILED:$failed"; exit 1; }
