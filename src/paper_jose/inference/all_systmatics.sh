#!/bin/bash

# List of GW event IDs (replace with actual IDs as needed)
OUTDIR="./CSE_systematics/"
TEMPLATE_FILE="template_systematics.sh" # Path to the submission bash template, located in PWD

# Loop over each GW event ID
for NB_CSE in 8 10 20 30 40 50 60 70 80 90 100; do
  EVENT_DIR="${OUTDIR}/outdir_${NB_CSE}"
  NEW_SCRIPT="${EVENT_DIR}/submit.sh"

  echo
  echo
  echo

  echo "==== Submitting job for ${EVENT_DIR} ===="
  echo
  
  # Create a unique SLURM script for each GW event
  cp "$TEMPLATE_FILE" "$NEW_SCRIPT"
  
  # Replace placeholders in the SLURM script
  sed -i "s|NB_CSE|$NB_CSE|g" "$NEW_SCRIPT"
  
  # Submit the job to SLURM
  sbatch "$NEW_SCRIPT"
  echo "==== Submitted job for ${EVENT_DIR} ===="
  
  echo
  echo
  echo
done