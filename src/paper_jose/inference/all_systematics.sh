#!/bin/bash

# Loop over each GW event ID
for NB_CSE in 8 10 20 30 40 50 60 70 80 90 100; do
  SH_FILE="./CSE_systematics/outdir_${NB_CSE}/submit.sh"

  # Submit the job to SLURM
  sbatch $SH_FILE
  echo "==== Submitted ${SH_FILE} ===="
  
  echo
  echo
  echo
done