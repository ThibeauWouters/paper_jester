#!/bin/bash -l
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH -p gpu_h100
#SBATCH -t 04:00:00
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./CSE_systematics/outdir_2/log.out"
#SBATCH --job-name="2"

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
    --outdir ./CSE_systematics/outdir_2/ \
    --nb-cse 2 \
    --sample-chiEFT True \
    --sample-GW170817 True \
    --use-GW170817-posterior-agnostic-prior True \
    --sample-radio True \
    --sample-J0030 True \
    --sample-J0740 True \
    --sample-NICER-masses True \
    --n-loop-production 20 \
    --make-cornerplot True

echo 
echo 
echo 

echo "Postprocessing now"

python postprocessing.py ./CSE_systematics/outdir_2/

echo 
echo 
echo 

echo "Postprocessing done"

echo "DONE"