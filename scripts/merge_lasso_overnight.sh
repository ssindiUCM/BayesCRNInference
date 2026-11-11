#!/bin/bash
# Copy the per-channel results of run_lasso_overnight.sh back into the run folders, then print the run times.
set -e
cd "$(dirname "$0")/.."
WORK=sparse_runs/_overnight
for d in "$WORK"/*_ch*; do
  [ -f "$d/done" ] || { echo "NOT FINISHED: $d"; continue; }
  run=$(basename "$d"); ch=${run##*_ch}; run=${run%_ch*}
  cp "$d/output/omega_vec_for_channel_$ch.txt" "$d/output/iteration_for_channel_$ch.txt" "sparse_runs/$run/output/"
  echo "$run channel $ch: $(cat "$d/done") s"
done
echo "Merged. Tell Claude, and it will check the fits and update the tables."
