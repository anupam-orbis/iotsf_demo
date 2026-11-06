#!/bin/bash
# OUT-OF-DOMAIN RETENTION, STEP 2 of 3: train B/D/H on the seven gate-passing Moirai cells, save each
# end-of-training checkpoint, and score it on every other dataset's held-out windows.
#
# The design, thresholds and outcomes are fixed in results/v59_ood_retention/preregistration.json,
# written by scripts/preregister_ood_retention.py and committed before this script first ran. The cell
# list is read from that file, not restated here, so the two cannot disagree.
#
# FAILURE IS REPORTED, NOT FILTERED: every training and evaluation keeps its full log, a step counts
# as done only if its JSON exists, and the final line names everything missing.
#
# DISK: Base and Large checkpoints (~0.4 / 1.2 GB) are deleted once every retention JSON for them
# exists, after their SHA-256 is appended to $OUT/deleted_checkpoints.tsv. Small checkpoints are kept.
#
# Serialized behind scripts/run_strict_freeze_rest.sh -- concurrent MPS jobs contend badly.
# Resumes by output-file existence.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v59_ood_retention
REG=$OUT/preregistration.json
LOGS=logs/v59_ood_retention
mkdir -p "$LOGS"
[ -f "$REG" ] || { echo "no registration at $REG"; exit 1; }
git log --format=%h -1 -- "$REG" | grep -q . || { echo "$REG is not committed"; exit 1; }

while pgrep -f run_strict_freeze_rest.sh > /dev/null; do sleep 300; done
echo "start $(date '+%F %T')"

MISSING=()
# size dataset horizon n_train ood1,ood2,...  -- one line per registered cell, in registered order
CELLS=$($PY -c "
import json
for c in json.load(open('$REG'))['cells']:
    print(c['size'], c['dataset'], c['horizon'], c['n_train'], ','.join(c['ood_datasets']))")

while read -r SIZE DS H N OODS; do
  for ARM in B D H; do
    for S in 42 123 456; do
      RD="$OUT/${SIZE}_${DS}_h$H/condition_$ARM/s$S"
      J="$RD/condition_${ARM}_h${H}_s$S.json"
      ST="$RD/final_state.pt"
      TAG="${SIZE}_${DS}_h${H}_${ARM}_s$S"
      if [ ! -f "$J" ]; then
        echo "=== train $TAG  $(date '+%F %T')"
        $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/$DS.csv" \
          --condition "$ARM" --model-size "$SIZE" --horizon "$H" --epochs 20 --seed "$S" \
          --max-train-samples "$N" --device mps --results-dir "$RD" --save-final-state \
          > "$LOGS/train_$TAG.log" 2>&1
        if [ ! -f "$J" ] || [ ! -f "$ST" ]; then
          echo "    TRAIN FAILED; see $LOGS/train_$TAG.log"; MISSING+=("train:$TAG"); continue
        fi
      fi
      ALL_EVAL=1
      for O in ${OODS//,/ }; do
        E="$RD/retention_$O.json"
        [ -f "$E" ] && continue
        if [ ! -f "$ST" ]; then
          echo "    checkpoint gone before eval on $O"; MISSING+=("eval:$TAG:$O"); ALL_EVAL=0; continue
        fi
        $PY -u scripts/eval_retention.py --state "$ST" --task-a-data "data/forecasting/$O.csv" \
          --model-size "$SIZE" --horizon "$H" --device mps --out "$E" \
          > "$LOGS/eval_${TAG}_$O.log" 2>&1
        if [ -f "$E" ]; then
          grep -aE '^retention_A' "$LOGS/eval_${TAG}_$O.log" | sed "s/^/    $O: /"
        else
          echo "    EVAL FAILED on $O; see $LOGS/eval_${TAG}_$O.log"; MISSING+=("eval:$TAG:$O"); ALL_EVAL=0
        fi
      done
      if [ "$SIZE" != small ] && [ $ALL_EVAL -eq 1 ] && [ -f "$ST" ]; then
        printf '%s\t%s\t%s\n' "$ST" "$(shasum -a 256 "$ST" | cut -d' ' -f1)" "$(date '+%F %T')" \
          >> "$OUT/deleted_checkpoints.tsv" && rm -f "$ST"
      fi
    done
  done
done <<< "$CELLS"

if [ ${#MISSING[@]} -eq 0 ]; then
  echo "ALL OOD-RETENTION RUNS PRESENT $(date '+%F %T')"
else
  echo "INCOMPLETE: ${#MISSING[@]} step(s) missing: ${MISSING[*]} $(date '+%F %T')"
fi
