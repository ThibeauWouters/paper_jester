#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./CSE_systematics/outdir_10/log.out"
#SBATCH --job-name="10"

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
    --outdir ./CSE_systematics/outdir_10/ \
    --nb-cse 10 \
    --sample-chiEFT True \
    --sample-GW170817 True \
    --use-GW170817-posterior-agnostic-prior True \
    --sample-radio True \
    --sample-J0030 True \
    --sample-J0740 True \
    --sample-NICER-masses True \

echo 
echo 
echo 

echo "Postprocessing now"

python postprocessing.py ./CSE_systematics/outdir_10/

echo 
echo 
echo 

echo "Postprocessing done"

echo "DONE"