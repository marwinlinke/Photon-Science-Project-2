#!/bin/bash
# Train all three models (conv_mixer, fc, fc_profiles) on the ORIGINAL data (no preprocessing),
# one after another.
#
#   cd Photon-Science-Project-2      # repo root, with .venv set up (see README)
#   sbatch temp/train_all.sh                  # all three models
#   sbatch temp/train_all.sh fc fc_profiles   # only these (e.g. after a time-out)
#   squeue -u $USER                  # job state
#   tail -f slurm-train_all-<jobid>.out
#
# train.py expects a folder with sim.h5 + exp.h5, so data/original/ gets two symlinks to the raw
# files in /vol/tmp/feueforp (nothing is copied). Same images and labels as the caches, so
# data/splits.json and the split are the same as for the preprocessed runs.
# Each run writes runs/<date-time>_<model>_original/.
#
# Resources are kept small so the job starts sooner (short jobs can fill gaps in the queue).
# Expected runtime: conv_mixer ~50 min (28 s/epoch on the A100-class GPU), fc and fc_profiles
# faster, plus ~3 min data loading per model -> about 1.5 h. If the limit is hit, the models that
# already finished keep their complete run folders; only the running one is cut off.
# --dependency=singleton: at most one job named train_all of yours runs at a time; submitting it
# again just queues behind the first instead of taking a second GPU.
#SBATCH --job-name=train_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:1
# only A100s (about 2x faster than the RTX 6000 on gruenau1/2, but only 6 exist -> may wait longer):
# replace the line above by   #SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --dependency=singleton
#SBATCH --output=slurm-%x-%j.out

EPOCHS=100
MODELS="${*:-conv_mixer fc fc_profiles}"   # models given to sbatch, default all three
SIM=/vol/tmp/feueforp/project2_simulation_103k_norm16bin.h5
EXP=/vol/tmp/feueforp/project2_experiment_norm16bin.h5
DATA=data/original

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1   # print each line immediately (otherwise the .out log fills only in chunks)

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---- data folder: symlinks to the raw files
mkdir -p "$DATA"
ln -sfn "$SIM" "$DATA/sim.h5"
ln -sfn "$EXP" "$DATA/exp.h5"
[ -r "$DATA/sim.h5" ] && [ -r "$DATA/exp.h5" ] || { echo "raw files not readable: $SIM / $EXP"; exit 1; }

# ---- training; a failed model does not stop the others
failed=""
for m in $MODELS; do
    echo; echo "===== $m ($EPOCHS epochs, original data), $(date)"
    python src/train.py --cache "$DATA" --model "$m" --epochs "$EPOCHS" --name original || failed="$failed $m"
done

echo; echo "done $(date)"
[ -z "$failed" ] || { echo "FAILED:$failed"; exit 1; }
