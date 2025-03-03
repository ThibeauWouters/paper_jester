#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=30G
#SBATCH --output="./outdir_new_prior/log.out"
#SBATCH --job-name="new_prior"

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
    --outdir ./outdir_new_prior/ \
    --n-loop-production 20 \
    --make-cornerplot True \
    --which-nbreak-prior broad
    
python postprocessing.py outdir_new_prior

echo "DONE"