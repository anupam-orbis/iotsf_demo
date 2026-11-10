#!/usr/bin/env python3
"""
SECOND-FAMILY ARM (Traffic), STEP 1 of 3. Freeze the design, the statistics and the outcomes BEFORE any
outcome run.

WHY THIS ARM EXISTS
-------------------
Every non-Moirai cell in the matrix fails the fitted capability screen, so the paper's evidence that
drift does not track the intervention rests on Moirai wherever there is a capability to protect
(reviewer u3rt, Q2). scripts/screen_second_family.py, run before this file was written and with no
training, found one dataset where both other backbones clear the screen on both window sets: Traffic
at h=24 (results/v60_second_family/screen.json; Chronos +0.287 selection / +0.231 held-out, TimesFM
+0.444 / +0.207). Exchange and ILI fail for both. This arm runs the intervention on the two Traffic
cells.

WHAT IS REUSED RATHER THAN REIMPLEMENTED: each backbone's existing matrix trainer and settings,
unchanged, so the new cells sit beside that backbone's existing cells (Table tab:trainers):
  Chronos B  scripts/chronos_mse_finetune.py     (MSE head, 30 ep, n=8000, lr 1e-4, batch 32, patience 7)
  Chronos D  scripts/chronos_frozen_converged.py (cached features, lr 1e-3, <=3000 ep, patience 200)
  TimesFM    scripts/finetune_timesfm.py         (native head, 20 ep, n=1000, lr 1e-4, batch 16, patience 7)

Run once, then commit, THEN run scripts/run_second_family.sh. `git log` must show this file's commit
strictly before the first outcome JSON.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/v60_second_family/preregistration.json"
SCREEN = ROOT / "results/v60_second_family/screen.json"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: a registration is written once and never edited")
    screen = json.loads(SCREEN.read_text())["cells"]
    gates = {k: round(v["V_ridge"], 4) for k, v in screen.items() if "traffic" in k}
    reg = dict(
        registered_by="scripts/preregister_second_family.py",
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        purpose="Run the B / frozen intervention on the only non-Moirai cells with a demonstrated "
                "capability, and ask whether the paper's findings hold off Moirai.",
        screen=dict(file="results/v60_second_family/screen.json", computed_before_registration=True,
                    traffic_gates=gates,
                    failed_datasets=["Exchange", "ILI"],
                    note="Both Traffic held-out gates (+0.231, +0.207) are within 0.04 of the 0.20 "
                         "threshold and are reported as marginal passes."),
        cells=[dict(backbone="chronos-t5-small", dataset="Traffic", horizon=24, lookback=96,
                    target="OT (univariate)"),
               dict(backbone="timesfm-2.5-200m", dataset="Traffic", horizon=24, lookback=96,
                    target="OT (univariate)")],
        data=dict(source="laiguokun/multivariate-time-series-data traffic.txt.gz, last series as OT",
                  split="train 12,280 / val 1,756 / test 3,508 (70/10/20)",
                  csv_sha256_prefix="30aa2d41ed1a9421be55"),
        arms=dict(
            chronos=dict(B="full fine-tune of the T5 encoder + MSE head",
                         D="encoder AND shared token embedding frozen, only the MSE head trains; this "
                           "is a strict freeze by construction (CKA 1, drift 0), so on Chronos D is "
                           "also the strict-freeze arm and no separate H is run"),
            timesfm=dict(B="full fine-tune",
                         D="transformer stack frozen, input tokenizer and output heads trainable",
                         H="strict freeze: stack AND tokenizer frozen, only the output heads train")),
        seeds=dict(chronos=[42, 43, 44, 45, 46], timesfm=[42, 123, 456, 789, 999],
                   note="each backbone's existing seed list, extended to 5"),
        commands=dict(
            chronos_B="chronos_mse_finetune.py --dataset Traffic --condition B --seed {s} --epochs 30 "
                      "--max-train-samples 8000 --lr 1e-4 --batch-size 32 --patience 7 --device cuda",
            chronos_D="chronos_frozen_converged.py --dataset Traffic --seed {s} --lr 1e-3 "
                      "--max-epochs 3000 --patience 200 --max-train-samples 8000 --device cuda",
            timesfm="finetune_timesfm.py --dataset Traffic --condition {B,D,H} --seed {s} --epochs 20 "
                    "--batch-size 16 --max-train-samples 1000 --lr 1e-4 --patience 7 --device cuda"),
        environment=dict(device="AWS g5.xlarge, NVIDIA A10G", torch="2.13.0+cu130", python="3.12",
                         note="Every run in this arm is fresh on this device. Chronos's sampled "
                              "zero-shot differs across devices (ETTh1 seed 42: 1.3245 on A10G against "
                              "1.3349 on record), so these cells are never pooled with the matrix's "
                              "existing records; they are reported as their own arm."),
        statistics=dict(
            outcome_window="held-out test windows (200, seed 0), scored once after training",
            chronos_contrast="d_enc = (test_mse_B - test_mse_D) / zs_test * 100 per seed, paired on "
                             "seed, with D's head chosen by the matrix rule (closed-form ridge if its "
                             "validation loss is within 5% of the AdamW head's, else AdamW). "
                             "Chronos forgetting against zero-shot is NOT read: zero-shot runs "
                             "through the generative decoder and B/D through the MSE head, the "
                             "confound the matrix already declares.",
            timesfm_contrast="d_enc = forgetting_pct_test(B) - forgetting_pct_test(H) per seed "
                             "(primary), and the same with D (secondary)",
            interval="two-sided 95% t-interval on the 5 paired differences; mean +- SEM reported",
            call="'freezing better' if the interval lies above 0, 'adapting better' if below, "
                 "'inconclusive' otherwise",
            degradation_definition="TimesFM only (native head on both sides): gate >= 0.20 on the "
                                   "selection split AND forgetting_B > 0 in every seed AND "
                                   "forgetting_frozen < 0 in every seed, frozen = H (primary) or D",
            cka="final_cka of each B run, encoder output against the pre-trained encoder on the "
                "selection windows, as each trainer already records it"),
        outcomes_fixed_in_advance=[
            "O1: TimesFM/Traffic meets the degradation definition -> the paper has its first harm "
            "case. Its B CKA is reported against Chronos/Traffic's and against the Moirai cells, and "
            "the abstract's 'no cell meets the definition' is narrowed to the cells it covers.",
            "O2: some cell's call is 'freezing better' without meeting the definition -> reported "
            "as a non-Moirai cell where the frozen arm wins, beside its CKA; the drift claim is "
            "tested on whether the lower-CKA cell is the one where freezing wins, as an observation "
            "on two cells, not an inference.",
            "O3: both calls are 'adapting better' or 'inconclusive' -> the Moirai finding that "
            "adaptation is not harmful where a capability exists extends to two further backbones "
            "on one dataset; reported with the marginal-gate caveat.",
        ],
        also_reported=[
            "per-seed CKA, l2 drift, best epoch and selection-window forgetting for every run",
            "the strict-freeze check: Chronos D and TimesFM H must show CKA 1.000; a run that does "
            "not is reported and excluded from the H contrast with the reason",
            "whether Traffic appears in each backbone's published pre-training corpus",
        ],
        what_this_cannot_show="Two cells on one dataset cannot estimate a correlation between CKA "
                              "and the intervention; this arm tests whether the paper's per-cell "
                              "findings hold on two more backbones where a capability exists.",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
