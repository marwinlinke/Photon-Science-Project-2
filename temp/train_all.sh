#!/bin/bash
# Train all three models (conv_mixer, fc, fc_profiles) on the ORIGINAL data (no preprocessing),
# one after another.
#
#   cd Photon-Science-Project-2      # repo root, with .venv set up (see README)
#   sbatch train_all.sh
#   squeue -u $USER                  # job state
#   tail -f slurm-train_all-<jobid>.out
#
# train.py expects a folder with sim.h5 + exp.h5, so data/original/ gets two symlinks to the raw
# files in /vol/tmp/feueforp (nothing is copied). Same images and labels as the caches, so
# data/splits.json and the split are the same as for the preprocessed runs.
# Each run writes runs/<date-time>_<model>_original/.
#SBATCH --job-name=train_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

EPOCHS=100
MODELS="conv_mixer fc fc_profiles"
SIM=/vol/tmp/feueforp/project2_simulation_103k_norm16bin.h5
EXP=/vol/tmp/feueforp/project2_experiment_norm16bin.h5
DATA=data/original

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

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
