#!/bin/bash
# Run the sparse-learning-CRN LASSO (Zhang et al.) fits with their compiled program, to convergence, with the same long
# stopping rule as all other runs (tot_step 300000, stop after 100000 steps without a new minimum; see
# sparse_learning.cfg in each run folder). Their code is not modified.
#   * ergodic example, T = 1000 and 5000, LASSO-BB and LASSO-MA
#   * random network, LASSO-BB (the 8 runs in the paper; earlier results with the default stopping rule are kept in
#     sparse_runs/_previous_default_stopping/) and LASSO-MA at T = 100, 200, 400 (new)
#
# Each fit is split by channel (their input file output/channels_to_learn.txt lists the channels to fit), and the
# channels run in parallel. Results are merged back with merge_lasso_overnight.sh.
#
# Usage (macOS Terminal):
#   cd ~/Documents/GitHub/BayesCRNInference
#   caffeinate -i bash scripts/run_lasso_overnight.sh          # caffeinate keeps the Mac awake while it runs
# Progress: ls sparse_runs/_overnight/*/done   (one 'done' file per finished channel)

set -e
# Their compiled program. Override with:  SPARSE_EXE=/path/to/sparse_learning caffeinate -i bash scripts/run_lasso_overnight.sh
SPARSE_EXE="${SPARSE_EXE:-$HOME/Dropbox/WritingProjects/2026_06_08_CRNWork/sparse-learning-CRN/src/sparse_learning}"
# longest fits first
RUNS="example6_T5000_lam0p01_blackbox example6_T5000_lam0p01_massaction example6_T1000_lam0p01_blackbox
      example5_T400_lam0p01 example5_T200_lam0p001 example5_T200_lam0p005 example5_T200_lam0p01
      example5_T200_lam0p05 example5_T200_lam0p1 example5_T200_lam0p5 example5_T100_lam0p01
      example6_T1000_lam0p01_massaction
      example5_T400_lam0p01_massaction example5_T200_lam0p01_massaction example5_T100_lam0p01_massaction"
NPAR=$(( $(sysctl -n hw.ncpu) - 1 ))       # parallel jobs: all cores but one

cd "$(dirname "$0")/.."
[ -x "$SPARSE_EXE" ] || { echo "Cannot find $SPARSE_EXE"; exit 1; }
# check that the program can start on this Mac (e.g. its libconfig library is installed) before launching 185 fits
probe=$(mktemp -d); (cd "$probe" && "$SPARSE_EXE" > out.txt 2>&1) || true; if grep -q "Library not loaded\|dyld" "$probe/out.txt"; then
  echo "The program cannot start on this Mac:"; head -3 "$probe/out.txt"; exit 1; fi
WORK=sparse_runs/_overnight
mkdir -p "$WORK"

jobs_file="$WORK/jobs.txt"; : > "$jobs_file"
for run in $RUNS; do
  src="sparse_runs/$run"
  read -r -a chans <<< "$(sed -n 2p "$src/output/channels_to_learn.txt")"
  for ch in "${chans[@]}"; do
    d="$WORK/${run}_ch$ch"
    rm -rf "$d"; cp -R "$src" "$d"
    printf '1\n%s \n' "$ch" > "$d/output/channels_to_learn.txt"     # fit this channel only
    echo "$d" >> "$jobs_file"
  done
done

echo "$(wc -l < "$jobs_file") channel fits, $NPAR at a time. Started $(date)."
# macOS xargs limits -I replacements to 255 bytes, so the folder is passed as an argument ($1) instead
export SPARSE_EXE
cat "$jobs_file" | xargs -P "$NPAR" -n 1 bash -c '
  cd "$1" && start=$(date +%s) && "$SPARSE_EXE" > run_log.txt 2>&1 \
  && echo $(( $(date +%s) - start )) > done && echo "finished $1 in $(cat done) s"' _
echo "All done $(date). Now run: bash scripts/merge_lasso_overnight.sh"
