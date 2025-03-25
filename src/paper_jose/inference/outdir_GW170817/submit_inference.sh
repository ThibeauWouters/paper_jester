#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./outdir_GW170817/log.out"
#SBATCH --job-name="GW170817"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jose

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python inference.py \
    --outdir ./outdir_GW170817/ \
    --sample-GW170817 True \
    --use-GW170817-posterior-agnostic-prior True \
    --n-loop-production 20 \
    --make-cornerplot True
    # --sample-radio True \ # this is no longer needed with Hauke's GW only run
    # --sample-chiEFT True \
    
python postprocessing.py outdir_GW170817

echo "DONE"