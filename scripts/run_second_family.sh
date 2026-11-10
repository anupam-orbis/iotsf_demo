#!/bin/bash
# SECOND-FAMILY ARM (Traffic), STEP 2 of 3: the runs results/v60_second_family/preregistration.json fixes.
#
# Refuses to start unless that registration exists and is committed. Resumes by output-file existence.
# Within a seed every arm runs back to back, so truncating this script leaves complete seed sets,
# never half a pair. A run counts only if its JSON exists; the summary lists the missing ones.
#
# Run on the GPU instance, inside tmux:  bash scripts/run_second_family.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REG=results/v60_second_family/preregistration.json
[ -f "$REG" ] || { echo "no registration at $REG"; exit 1; }
if [ -d .git ]; then
  git ls-files --error-unmatch "$REG" >/dev/null 2>&1 || { echo "$REG is not committed"; exit 1; }
fi
PY=${PY:-.venv/bin/python}
export HF_HUB_OFFLINE=1
OUT=results/v60_second_family
LOG=logs/v60_second_family
mkdir -p "$LOG"

for S in 42 43 44 45 46; do
  F="$OUT/chronos/cond_B/mse_traffic/seed$S/condition_B_s$S.json"
  [ -f "$F" ] || { echo "=== chronos B seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/chronos_mse_finetune.py --dataset Traffic --condition B --seed "$S" --epochs 30 \
      --max-train-samples 8000 --lr 1e-4 --batch-size 32 --patience 7 --device cuda \
      --results-dir "$OUT/chronos/cond_B" > "$LOG/chronos_B_s$S.log" 2>&1; }
  F="$OUT/chronos/cond_D/mse_traffic/seed$S/condition_D_s$S.json"
  [ -f "$F" ] || { echo "=== chronos D seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/chronos_frozen_converged.py --dataset Traffic --seed "$S" --lr 1e-3 \
      --max-epochs 3000 --patience 200 --max-train-samples 8000 --device cuda \
      --results-dir "$OUT/chronos/cond_D" > "$LOG/chronos_D_s$S.log" 2>&1; }
done

for S in 42 123 456 789 999; do
  for C in B D H; do
    D_="$OUT/timesfm/Traffic_h24/condition_$C"
    [ -f "$D_/condition_${C}_h24_s$S.json" ] && continue
    echo "=== timesfm $C seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/finetune_timesfm.py --dataset Traffic --condition "$C" --seed "$S" --epochs 20 \
      --batch-size 16 --max-train-samples 1000 --lr 1e-4 --patience 7 --device cuda \
      --results-dir "$D_" > "$LOG/timesfm_${C}_s$S.log" 2>&1
  done
done

echo "--- summary $(date +%H:%M:%S)"
missing=0
for S in 42 43 44 45 46; do for C in B D; do
  f="$OUT/chronos/cond_$C/mse_traffic/seed$S/condition_${C}_s$S.json"
  [ -f "$f" ] || { echo "MISSING $f"; missing=$((missing+1)); }
done; done
for S in 42 123 456 789 999; do for C in B D H; do
  f="$OUT/timesfm/Traffic_h24/condition_$C/condition_${C}_h24_s$S.json"
  [ -f "$f" ] || { echo "MISSING $f"; missing=$((missing+1)); }
done; done
echo "missing: $missing of 25"
