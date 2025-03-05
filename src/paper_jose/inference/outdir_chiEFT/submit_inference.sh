#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./outdir_chiEFT/log.out"
#SBATCH --job-name="chiEFT"

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
    --outdir ./outdir_chiEFT/ \
    --sample-chiEFT True \
    --n-loop-production 20 \
    --make-cornerplot True

python postprocessing.py outdir_chiEFT

echo "DONE"