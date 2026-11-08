# Errata: changes from the earlier public version

An earlier version of this work was publicly available before submission. This file lists every
claim that changed. The paper itself carries one sentence pointing here.

## Corrected measurements

1. **The capability screen's linear baseline was unfitted.** The paper defines the screen against a
   lookback-96 linear map fitted on the cell's training windows. The earlier version's numbers came
   from a different estimator that shares the name: a per-window trend extrapolation that uses no
   training data. A constant training-mean predictor beats that estimator (ETTh1 h=96: 0.9937
   against 1.2207). Fitting the baseline as defined changes the screen's verdict on 17 of 21 Moirai
   cells, and 16 of those go from pass to fail.
   Consequences:
   - **(a)** The earlier version reported eight degradation cells, all on ETTh1 or Weather at
     h ∈ {96, 192}. None of the eight passes the fitted screen, and no cell now meets the
     degradation definition. No model was re-run for this correction.
   - **(b)** The earlier version reported that eight of nine non-Moirai cells pass the screen at
     h=24. Under the fitted baseline none of the ten does.
   - **(c)** The prospective arm's scorecard was TP 2, FP 6, precision 0.25, recall 1.00. It is now
     TP 0, FP 8, and recall is undefined because there are no positive cells.
   - **(d)** Batch 1 of the prospective test was frozen against the unfitted screen. It is reported
     as registered, and Batch 3 re-registers it on the fitted screen.

   See main Appendix "Fitted Versus Unfitted Linear Baseline".

2. **ILI.** The earlier version reported Moirai-Small/ILI as a 57% screen pass. The fitted screen
   gives −1.625, so the cell fails decisively. Its frozen-encoder control had never been trained:
   the script assigned the zero-shot MSE as the fine-tuned MSE, which gave "0.0% in all 10 seeds".
   Trained properly, the full-minus-frozen gap B−D is −38.6 ± 1.3 on held-out windows, not
   −80.7. Condition B is unaffected.

3. **The sample-size sweep was scored on selection windows.** These are the windows the
   early-stopping rule also chose the epoch on, and the paper itself shows that this split can flip
   the sign of the outcome. On held-out windows, forgetting falls steadily as CKA falls:
   - n=500: +5.2%
   - n=10k: −18.5 ± 2.8% (SEM), with 10/10 seeds improving

   The earlier "non-monotonic, sign reverses at n=5k" description is an artefact of the selection
   windows. All four n=10k protocols agree on held-out windows: −16% to −21%, in 27 of 28 seeds.
   The earlier figure for the sweep was hand-typed, did not match the run records, and carried two
   panels of the withdrawn probe statistic. It has been regenerated from the records.

4. **The layer-wise unfreeze ladder was read on selection windows.** There, unfreezing 3 or 6 of the
   6 layers looked indistinguishable (−5.9% against −7.9%). On held-out windows, unfreezing all six
   is better by 13.3 points (−9.5% against −22.9%, 4.0 SE), while CKA falls from 0.716 to 0.465.
   The top-up's registered statistic (selection windows) is still reported beside it.

5. **The random-initialisation CKA floor is 0.215, not 0.** The earlier version reported
   0.0000 ± 0.0001, and no script in the repository produced that number. A naive re-initialisation
   that zeroes LayerNorm's scale gives a constant output, and therefore CKA 0.

6. **The early-stopping protocol was stated globally.** It is now stated for each trainer:
   - final epoch, no early stopping: the 21 Moirai ETT/Weather/Electricity cells;
   - best epoch: ILI, TimesFM, Chronos and the n=10k sweep arm.

7. **The ten-seed random-initialisation control.** Five of its ten printed rows traced to no run
   record. It also compared a final-epoch arm against an early-stopped one. It is withdrawn along
   with the probe statistic it measured (item 8).

8. **The per-layer CKA profile.** The earlier version printed a profile that rose monotonically
   with depth. Recomputing it from the checkpoints gives a trough at layer 4, and the top three
   layers drift more than the bottom three. The earlier numbers left no run record.

## Withdrawn analyses

9. **The probe-sign asymmetry axis (linear-probe ΔR²).** Withdrawn in full. Under a temporal split
   the pre-trained probe scores R² between −164.1 and −2.7, so ΔR² measures movement between two
   failed fits. Under a randomised split on the same windows every probe generalises
   (R² 0.678–0.994), and no |ΔR²| exceeds 0.010. Everything built on the axis is withdrawn:
   - the probe sections, task-orthogonal probes, the probe noise-floor sweep and the
     random-initialisation probe control;
   - the ILI probe prediction;
   - the ΔR² columns of the sample sweep and the layer-unfreeze tables.

10. **The cross-backbone dataset ordering.** Chronos-T5 and TimesFM-2.5 rank the shared datasets
    identically on held-out windows (ρ = +1.00), and the earlier version read that as a property of
    the benchmarks. On selection windows the two disagree (ρ = −0.40). The claim is withdrawn.

11. **The gate's anti-predictive interval.** ρ = −0.518 between the gate and B−D excludes zero when
    cells are resampled, [−0.840, −0.040]. It does not exclude zero when the six
    (backbone, dataset) clusters are resampled, [−0.810, +0.500]. Only the direction is reported.

12. **Chronos-T5-Small on M4-Monthly.** The earlier appendix reported ten seeds at two sample
    sizes, a frozen arm, a random-initialisation control, probe figures and an 84.5% advantage over
    a linear baseline. None of these has a run record. The cell was re-run from the data up
    (conditions B and D, 3 seeds each, registered before the gate was computed), and fails the
    fitted screen at +0.092.

## Removed as out of scope

13. The IoT negative control (N-BaIoT) and the ETTh1 spectral pilot, both from the work's earlier
    IoT framing. Neither supported any claim in the current paper.

## Code defects that changed no published number

14. The glob that collected the pre-registered arms' zero-shot references matched every condition,
    not just `condition_A`. So for a period the registered denominators were means over the outcome
    runs. The shift was at most 0.0044 in the screen score and changed 0 of 26 admissions. The code
    now fails if an outcome file is read as a registered reference.

## Counts restated

15. **The abstract's gate-correction count.** The abstract said the unfitted baseline "puts 16 of 21
    Moirai cells on the wrong side of the screen", while the introduction and Section 3 said it
    "would move 17". Both numbers are in the records. 16 cells pass under the unfitted baseline and
    fail under the fitted one, and 1 does the reverse. A cell on the wrong side in either direction
    counts, so the abstract now says 17.
