#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./numpyro/log.out"
#SBATCH --job-name="numpyro"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jose

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python numpyro_inference.py \
    --outdir ./numpyro/ \
    --sample-chiEFT True \
    --sample-GW170817 True \
    --use-GW170817-posterior-agnostic-prior True \
    --sample-radio True \
    --sample-J0030 True \
    --sample-J0740 True \
    --sample-NICER-masses True \

echo "DONE"