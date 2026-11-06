#!/usr/bin/env python3
"""
OUT-OF-DOMAIN RETENTION, STEP 1 of 3. Freeze the design, the thresholds and the outcomes BEFORE any run.

WHY THIS ARM EXISTS
-------------------
Every intervention cell in the matrix is fine-tuned and scored on the same series, so the matrix
measures what encoder adaptation does to the TARGET task. A reviewer (sq16) pointed out that this is
not a test of forgetting: a model can drift, improve on its target, and lose what it could do on
everything else, and only the last of those is forgetting. This arm measures it directly. Each of the
seven gate-passing Moirai cells is fine-tuned under full fine-tuning (B), the backbone-frozen control
(D) and the strict freeze (H), and every end-of-training checkpoint is scored zero-shot on the
held-out windows of the OTHER datasets, through Moirai's own forecasting head.

The question CKA is then asked is the one the forgetting literature asks of it: does representation
change measured on a domain order the capability lost on that domain?

WHAT IS IMPORTED RATHER THAN REIMPLEMENTED: scripts/eval_retention.py, unchanged, the positive
control's instrument. It recomputes the zero-shot reference on the same windows, loads the checkpoint
with strict=True, and measures CKA on the out-of-domain dataset's own inputs.

Run once, then commit, THEN run scripts/run_ood_retention.sh. `git log` must show this file's commit
strictly before the first retention JSON.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/v59_ood_retention/preregistration.json"

DATASETS = ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity7"]

# The seven cells with gate_val >= 0.20, read from scripts/cell_matrix.build_rows() on 2026-11-06.
# n is each cell's own training size in the matrix, so these reruns repeat the matrix protocol.
CELLS = [
    dict(size="small", dataset="ETTh2", horizon=96,  n_train=500,  matrix_dir="forecasting_finetune_20ep"),
    dict(size="small", dataset="ETTh2", horizon=192, n_train=500,  matrix_dir="forecasting_finetune_20ep"),
    dict(size="small", dataset="ETTm2", horizon=192, n_train=1000, matrix_dir="v5_ettm2"),
    dict(size="large", dataset="ETTh2", horizon=96,  n_train=500,  matrix_dir="v8_etth2_large"),
    dict(size="base",  dataset="ETTh2", horizon=96,  n_train=1000, matrix_dir="v5_etth2_base"),
    dict(size="base",  dataset="ETTm2", horizon=192, n_train=1000, matrix_dir="v47_prospective"),
    dict(size="base",  dataset="ETTh2", horizon=192, n_train=1000, matrix_dir="v5_etth2_base"),
]
for c in CELLS:
    c["ood_datasets"] = [d for d in DATASETS if d != c["dataset"]]


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: a registration is written once and never edited")
    reg = dict(
        registered_by="scripts/preregister_ood_retention.py",
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        purpose="Measure loss of out-of-domain forecasting capability after target fine-tuning, and "
                "ask whether CKA orders it.",
        cells=CELLS,
        arms=dict(B="full fine-tune", D="backbone-frozen encoder (in_proj, mask_encoding trainable)",
                  H="strict freeze (only param_proj trains)"),
        seeds=[42, 123, 456],
        training=dict(epochs=20, early_stopping=False, lr=1e-4, batch_size=16, features="M",
                      checkpoint="end-of-training state, --save-final-state"),
        evaluation=dict(instrument="scripts/eval_retention.py", windows="held-out test split",
                        max_eval_sequences=300, horizon="the cell's own", lookback=96,
                        num_samples=20, point_forecast="median"),
        primary_readout="retention_pct = (MSE_finetuned - MSE_zeroshot) / MSE_zeroshot * 100 on each "
                        "out-of-domain dataset; positive = the capability got worse",
        contrasts=dict(
            forgetting_due_to_encoder="ret_B - ret_H per (cell, ood dataset), paired on seed",
            secondary="ret_B - ret_D, same pairing",
        ),
        thresholds=dict(
            harmful_forgetting_pp=10.0,
            harmful_forgetting_rule="ret_B - ret_H >= 10 pp in every seed of a (cell, ood dataset) pair",
            cka_predicts_rule="Spearman rho between (1 - CKA on the ood inputs) and ret_B over all B "
                              "(cell, ood dataset, seed) records >= 0.5, with a 95% bootstrap interval "
                              "resampling whole cells that excludes 0",
        ),
        outcomes_fixed_in_advance=[
            "O1: some (cell, ood dataset) pair meets the harmful-forgetting rule AND CKA meets the "
            "cka_predicts_rule -> out-of-domain forgetting exists here and CKA orders it; the paper's "
            "negative claim is restricted to target-task outcomes.",
            "O2: some pair meets the harmful-forgetting rule AND CKA does not meet the cka_predicts_rule "
            "-> forgetting exists and CKA does not order it; this supplies the positive cases the "
            "matrix lacks and extends the negative claim to forgetting proper.",
            "O3: no pair meets the harmful-forgetting rule -> no out-of-domain forgetting at this scale; "
            "reported as such, and the CKA association is reported descriptively only.",
        ],
        also_reported=[
            "target-task test MSE of every rerun beside the matrix record it repeats, as a "
            "reproducibility check",
            "CKA on the target inputs (from the training record) against mean ret_B, per cell",
        ],
        checkpoint_policy="Moirai-Small checkpoints are kept. Base and Large checkpoints are deleted "
                          "after every retention JSON for that checkpoint exists, with their SHA-256 "
                          "recorded first, because the disk cannot hold them.",
        what_this_cannot_show="The out-of-domain datasets are the paper's own benchmarks, not Moirai's "
                              "pre-training corpus, so this measures transfer capability on held-out "
                              "series, not retention of the pre-training distribution itself.",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
