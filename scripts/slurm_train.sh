#!/bin/bash
#SBATCH --job-name=semanticbox
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=tkg5kq@virginia.edu

# Usage: sbatch --gres=gpu:N slurm_train.sh <config> <num_gpus>


cd /sfs/weka/scratch/tkg5kq/SemanticBox

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    now=$(date +"%y%m%d")
    logpath="${LOG_DIR}/$SLURM_ARRAY_JOB_ID/logs-$now-${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$logpath"
    logfile="$logpath/${SLURM_ARRAY_TASK_ID}.out"
else
    now=$(date +"%y%m%d-%H%M%S")
    mkdir -p "${LOG_DIR}/$now"
    logfile="${LOG_DIR}/$now/logs-$now-${SLURM_JOB_ID}.out"
fi

source /home/tkg5kq/.bashrc > "${logfile}" 2>&1
source activate bound >> "${logfile}" 2>&1

if [ -f "$1" ]; then
    realpath $1 >> "${logfile}" 2>&1
    config=$(realpath "$1") 
    GPU=${2:-4}
else
    echo "need a config file"
    exit 1
fi

echo "Command: $0 $@" >> "${logfile}"
echo "Config: ${config}, GPUs: ${GPU}" >> "${logfile}"

bash scripts/run_train.sh "${config}" >> "${logfile}" 2>&1

sleep 45
