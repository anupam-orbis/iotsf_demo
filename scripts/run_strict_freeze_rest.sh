#!/bin/bash
# The strict-freeze control (condition H) on the SIX Moirai cells that still lack it.
#
# WHY THIS RUN EXISTS: condition H covers 16 of the 31 intervention cells (v45, v51, v53). Reviewer
# sq16 asked for the strict freeze to be the primary frozen-encoder contrast rather than a sixteen-
# cell control, because condition D leaves in_proj and mask_encoding trainable and H reverses D's
# sign on four ETTm2 cells. This run closes the Moirai gap, so that B-H exists on every Moirai cell.
# Chronos and TimesFM need their own definition of a strict freeze and are not covered here.
#
# PROTOCOL MATCHING -- read off the stored B/D runs, per cell, before writing this file:
#
#   cell                    B/D dir                                   n     epochs  early_stop       B/D seeds
#   small Electricity7 h96  v47_prospective/small_Electricity7_h96    1000  20      {enabled:False}  42/123/456
#   small Electricity7 h192 v47_prospective/small_Electricity7_h192   1000  20      {enabled:False}  42/123/456
#   base  Weather h96       v47_prospective/base_Weather_h96          1000  20      {enabled:False}  42/123/456
#   base  Weather h192      v47_prospective/base_Weather_h192         1000  20      {enabled:False}  42/123/456
#   large ETTh1 h96         v47_prospective/large_ETTh1_h96           1000  20      {enabled:False}  42/123/456
#   large Weather h96       v47_prospective/large_Weather_h96         1000  20      {enabled:False}  42/123/456
#
# Batch size and learning rate are left at the script defaults, as in the other strict-freeze
# runners; the stored B/D records do not carry either field.
#
# FAILURE IS REPORTED, NOT FILTERED. The earlier runners piped training output through grep|tail,
# which discards the exit status and can let a crash read as a completed run. Here every run keeps
# its full log, and a run counts as done only if its JSON exists afterwards; the final line names
# every missing run.
#
# Serialized -- concurrent MPS jobs contend badly. Resumes by output-file existence.
# ORDER: cheapest first, so a closed window loses whole cells at the tail.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v58_strictfreeze_rest
LOGS=logs/v58_strictfreeze_rest
mkdir -p "$LOGS"
MISSING=()

cell() {  # size dataset horizon n_train
  for S in 42 123 456; do
    D_="$OUT/$1_$2_h$3/condition_H"
    J="$D_/condition_H_h$3_s$S.json"
    [ -f "$J" ] && continue
    L="$LOGS/$1_$2_h$3_s$S.log"
    echo "=== $1/$2 h$3 n=$4 cond H seed $S  $(date '+%F %T')"
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/$2.csv" \
      --condition H --model-size "$1" --horizon "$3" --epochs 20 --seed "$S" \
      --max-train-samples "$4" --device mps --results-dir "$D_" > "$L" 2>&1
    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$J" ]; then
      echo "    FAILED (exit $RC, json $( [ -f "$J" ] && echo present || echo missing )); see $L"
      MISSING+=("$1_$2_h$3_s$S")
    else
      grep -aE 'Zero-shot|forgetting|CKA' "$L" | tail -3 | sed 's/^/    /'
    fi
  done
}

cell small Electricity7  96 1000
cell small Electricity7 192 1000
cell base  Weather       96 1000
cell base  Weather      192 1000
cell large ETTh1         96 1000
cell large Weather       96 1000

if [ ${#MISSING[@]} -eq 0 ]; then
  echo "ALL 18 STRICT-FREEZE RUNS PRESENT $(date '+%F %T')"
else
  echo "INCOMPLETE: ${#MISSING[@]} run(s) missing: ${MISSING[*]} $(date '+%F %T')"
fi
