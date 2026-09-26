#!/usr/bin/env python3
"""
What would a practitioner have saved by acting on CKA? A price, out of sample.

WHY THIS EXISTS
---------------
The paper's central claim is that measured drift is not an ACTIONABLE substitute for the
fine-tune-versus-frozen-encoder intervention. Everything supporting it so far is a statistical
statement about association: a within-Moirai rank correlation whose clustered interval is wide, a
pre-registered leave-one-cluster-out comparison in which adding CKA does not improve prediction, and a
best-case in-sample threshold that misplaces 5 of the 31 cells. A reviewer can fairly answer that a
wide interval is not evidence of no relationship -- it is not, and the paper says so -- and that the
reader is therefore being asked to accept a null.

This module reframes the same data as a DECISION rather than an association, which is the form in
which "not actionable" is falsifiable. A practitioner facing one cell chooses between two policies:
fine-tune everything, or fine-tune with the encoder frozen. $\\denc$ is the held-out contrast between
them, in percentage points of MSE, POSITIVE WHEN FREEZING IS BETTER. So the realised saving of any
policy, measured against the do-nothing-clever default of always adapting, is

    saving(policy) = mean over cells of  denc_i * 1[policy freezes cell i]

and every rule that reads a single CKA threshold is in the same family. That makes five numbers
comparable on one scale, in the units the practitioner cares about:

    always-adapt          0 by construction -- the baseline the others are priced against
    always-freeze         mean(denc)              -- the cheap policy that ignores CKA entirely
    CKA rule, out of sample  the threshold rule a practitioner could actually have followed
    CKA rule, in sample      the same rule with its threshold chosen ON the cells it is scored on
    oracle                mean(max(denc, 0))      -- per-cell hindsight, the ceiling

THE ONE THAT IS NOT CHEATING. The out-of-sample row is computed by leave-one-cluster-out: the cut and
its orientation are searched on the 14 training clusters and applied, unchanged, to the held-out
cluster. Each cell is held out exactly once, so the reported figure is a mean over all scored cells of
a decision made without seeing that cell. The in-sample row is the same search run on all cells at
once; the gap between the two IS the optimism in a threshold chosen retrospectively, and it is the
quantity \\S9's "chosen on those same cells, so a lower bound and not a validated classifier" asserts
without pricing.

WHAT THIS IS NOT. It is POST-HOC and descriptive. Unlike scripts/cka_fixed_effects.py --loco, whose
specification and deciding statistic were frozen in results/preregister_loco.json before the models
ran, nothing here was registered: the policy family, the objective and the folds were all chosen after
the outcomes were known. It lives in its own file for exactly that reason -- adding it as a flag on
cka_fixed_effects.py would blur a registered analysis with an unregistered one. It also evaluates ONE
policy family (a single global threshold on CKA, either orientation); a per-backbone threshold, a
multivariate rule or a continuous-shrinkage policy could do better, and 15 clusters cannot rule that
out. The claim it supports is about the rule practitioners are actually offered, not about every rule.

THE CANDIDATE CUTS ARE INTERIOR. Thresholds are the midpoints between consecutive distinct CKA values
in the TRAINING fold, so the search cannot return a degenerate cut that freezes everything or nothing
and then be reported as though CKA had informed it: always-freeze and always-adapt are rows of their
own, priced separately. Ties are broken deterministically -- orientations in the order (low, high),
cuts ascending, first strict maximum kept -- so the output is a function of the data alone.

INFERENCE. Clusters (backbone x dataset) are resampled with replacement and the WHOLE
leave-one-cluster-out procedure is re-run inside each draw, with duplicate copies of a cluster sharing
one fold id so that a cell can never be in both the training and the held-out side of one fold.
Unlike the out-of-sample R^2 in cka_fixed_effects.py, every statistic here is a mean of bounded
per-cell quantities, so its bootstrap percentiles are well behaved and a 95% interval IS quoted --
but only to whole percentage points. At two decimals these bounds do not reproduce across bootstrap
seeds (the out-of-sample rule's lower bound came out -3.97 / -3.61 / -3.68 / -3.67 on seeds
0/1/7/13); rounded to whole points every quoted interval and every quoted draw share is identical
across those four seeds, which `--seedcheck` verifies at the printed precision rather than asserting.
The point estimates and the decision counts are functions of the data alone and carry no seed.

Run:  python3 scripts/cka_decision_value.py
      python3 scripts/cka_decision_value.py --json --latex
      python3 scripts/cka_decision_value.py --seedcheck
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
# loco_frame() is imported rather than reimplemented: it is the function that decides which cells are
# scorable (bd_test and cka both present) and which cluster each one belongs to, so a second copy here
# could silently price a different cell set than the pre-registered LOCO analysis validates on.
from cka_fixed_effects import loco_frame  # noqa: E402

OUT_JSON = ROOT / "results/cka_decision.json"
OUT_TEX = ROOT / "paper_8/tables/cka_decision.tex"
# Fixed here rather than taken from --boot/--seed, matching cka_fixed_effects.py's TEX_BOOT/TEX_SEED
# convention, so that --latex is reproducible from one documented command.
TEX_BOOT, TEX_SEED = 10_000, 0
ORIENTS = ("low", "high")
ORIENT_GLOSS = {"low": "freeze when CKA is LOW (drift large)",
                "high": "freeze when CKA is HIGH (drift small)"}


def freeze_mask(cka, t, orient):
    """Which cells the rule freezes. `low` = freeze the drifted cells, which is the prescription the
    forgetting literature implies; `high` is its inverse, searched too so that the rule family is not
    handicapped by our expectation about its sign."""
    return (cka < t) if orient == "low" else (cka > t)


def saving(y, mask):
    """Realised saving per cell against always-adapt, in percentage points of MSE.

    y is held-out denc, positive when freezing wins, so freezing a cell collects y (which may be
    negative -- freezing a cell that wanted adaptation COSTS). Averaged over all cells offered, not
    only the frozen ones: the practitioner faces every cell, and a rule that freezes one cell well and
    abstains everywhere else is not worth more than its effect on the whole caseload.
    """
    return float((y * mask).mean())


def candidate_cuts(cka):
    """Midpoints between consecutive distinct CKA values: interior by construction."""
    u = np.unique(cka)
    return (u[:-1] + u[1:]) / 2.0


def best_rule(y, cka, orients=ORIENTS):
    """The (threshold, orientation) maximising realised saving on the cells given.

    Deterministic tie-breaking: orientations in ORIENTS order, cuts ascending, strict improvement only.
    Returns None when there are fewer than two distinct CKA values, which is the only case in which the
    family is empty. `orients` is narrowed to ("low",) to price the rule the forgetting literature
    actually prescribes, rather than the best of both directions.
    """
    cuts = candidate_cuts(cka)
    if len(cuts) == 0:
        return None
    best = None
    for orient in orients:
        for t in cuts:
            s = saving(y, freeze_mask(cka, t, orient))
            if best is None or s > best[2]:
                best = (float(t), orient, s)
    return best


def misplaced(y, mask):
    """Cells on the wrong side of the decision: frozen when adaptation helped, or adapted when
    freezing would have. The same accounting as the best-cut count in fig1_diagnostic_flow.py, which
    minimises THIS rather than the saving, so the two need not choose the same cut."""
    return int(((mask & (y < 0)) | (~mask & (y > 0))).sum())


def oos_rule(f, fold_of, orients=ORIENTS):
    """Leave-one-cluster-out: choose the rule on the training clusters, apply it to the held-out one.

    Returns the mean saving over every cell that got a fold (each cell is held out exactly once), plus
    the per-fold record of which rule was chosen -- the orientation split across folds is itself a
    result, because a rule whose DIRECTION is not stable across folds cannot be a prescription.
    """
    y, cka = f["y"], f["cka"]
    fold_of = np.asarray(fold_of)
    contrib = np.full(len(y), np.nan)
    froze = np.zeros(len(y), bool)
    chosen = {}
    for fid in sorted(set(fold_of.tolist())):
        te = np.flatnonzero(fold_of == fid)
        tr = np.flatnonzero(fold_of != fid)
        if len(tr) < 2:
            continue
        r = best_rule(y[tr], cka[tr], orients)
        if r is None:
            continue
        t, orient, s_tr = r
        m = freeze_mask(cka[te], t, orient)
        contrib[te] = y[te] * m
        froze[te] = m
        chosen[str(fid)] = dict(threshold=t, orientation=orient, train_saving=s_tr,
                                n_held_out=int(len(te)), n_frozen=int(m.sum()),
                                held_out_saving=float((y[te] * m).mean()))
    ok = ~np.isnan(contrib)
    if not ok.any():
        return None
    return dict(saving=float(contrib[ok].mean()), n_scored=int(ok.sum()), chosen=chosen,
                contrib=contrib, scored=ok, froze=froze,
                misplaced=misplaced(y[ok], froze[ok]))


def policies(f, fold_of):
    """All six rows, on one scale, for one data set (the real one or a bootstrap draw)."""
    y, cka = f["y"], f["cka"]
    o = oos_rule(f, fold_of)
    # The rule the literature actually prescribes -- freeze where the representation moved MOST -- is
    # priced separately, because "best of both orientations" grants the shortcut a direction it is only
    # entitled to in hindsight, and the direction that wins here is the inverse of the prescribed one.
    olow = oos_rule(f, fold_of, ("low",))
    if o is None or olow is None:
        return None
    ins = best_rule(y, cka)
    if ins is None:
        return None
    ins_mask = freeze_mask(cka, ins[0], ins[1])
    return dict(
        always_adapt=0.0,
        always_freeze=float(y.mean()),
        cka_oos=o["saving"],
        cka_oos_prescribed=olow["saving"],
        cka_insample=ins[2],
        oracle=float(np.maximum(y, 0.0).mean()),
        insample_threshold=ins[0], insample_orientation=ins[1],
        n_scored=o["n_scored"], chosen=o["chosen"],
        oos_misplaced=o["misplaced"], insample_misplaced=misplaced(y, ins_mask),
        oos_n_frozen=int(o["froze"][o["scored"]].sum()),
        n_positive=int((y > 0).sum()),
        contrib=o["contrib"], scored=o["scored"],
    )


def bootstrap(f, n_boot, rng):
    """Cluster bootstrap of all four non-trivial policies, and of the two gaps that matter."""
    uniq = np.unique(f["cl"])
    idx_by_cluster = {c: np.flatnonzero(f["cl"] == c) for c in uniq}
    keys = ("always_freeze", "cka_oos", "cka_oos_prescribed", "cka_insample", "oracle")
    out = {k: [] for k in keys}
    kept = 0
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx, fold = [], []
        for c in np.unique(draw):
            for rep in range(int((draw == c).sum())):
                idx.append(idx_by_cluster[c])
                fold.append(np.full(len(idx_by_cluster[c]), c))   # duplicates share a fold id
        idx = np.concatenate(idx)
        fold = np.concatenate(fold)
        sub = {k: v[idx] for k, v in f.items()}
        if len(np.unique(sub["cl"])) < 3:
            continue
        p = policies(sub, fold)
        if p is None or not all(np.isfinite(p[k]) for k in keys):
            continue
        for k in keys:
            out[k].append(p[k])
        kept += 1
    return {k: np.array(v, float) for k, v in out.items()}, kept


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def share_le0(a):
    """The share of draws in which a policy is worth nothing or less, TO THE NEAREST 5 POINTS.

    Not to the nearest point: at 10,000 draws that share came out 25/26/26/26% over seeds 0/1/7/13, so
    a whole-percent figure in the paper is one a reader re-running the script with another seed cannot
    reproduce. Rounded to 5 points it is identical across those seeds, which --seedcheck checks. The
    raw fraction is kept in the JSON for the record and is deliberately not the quoted form.
    """
    raw = float((a <= 0).mean())
    return raw, int(5 * round(100 * raw / 5))


def run(a):
    f = loco_frame()
    fold_of = list(f["cl"])
    rng = np.random.default_rng(a.seed)
    p = policies(f, fold_of)
    if p is None:
        sys.exit("no scorable folds")

    n_cells, n_folds = len(f["y"]), len(set(fold_of))
    print("\n" + "=" * 92)
    print("WHAT WOULD ACTING ON CKA HAVE SAVED? POLICY VALUE IN PERCENTAGE POINTS OF MSE")
    print("=" * 92)
    print(f"  {n_cells} cells, {n_folds} leave-one-cluster-out folds (cluster = backbone x dataset)")
    print( "  POST-HOC AND DESCRIPTIVE: unlike --loco in cka_fixed_effects.py, nothing here was")
    print( "  pre-registered, and the family searched is a single global CKA threshold, both orientations.")
    print(f"  saving = mean over all {n_cells} cells of denc * 1[policy freezes]; denc > 0 means")
    print( "  freezing wins, so freezing a cell that wanted adaptation subtracts.\n")

    boots, kept = bootstrap(f, a.boot, rng)
    rows = [("always-adapt (baseline)", "always_adapt"),
            ("always-freeze", "always_freeze"),
            ("CKA rule as prescribed, OOS", "cka_oos_prescribed"),
            ("CKA threshold, out of sample", "cka_oos"),
            ("CKA threshold, in sample", "cka_insample"),
            ("oracle (per-cell hindsight)", "oracle")]
    print(f"  {'policy':30s} {'saving':>8s}   {'95% cluster CI':>20s}")
    for lab, k in rows:
        if k == "always_adapt":
            print(f"  {lab:30s} {0.0:+8.2f}   {'(by construction)':>20s}")
            continue
        lo, hi = ci(boots[k])
        print(f"  {lab:30s} {p[k]:+8.2f}   [{lo:+8.2f}, {hi:+8.2f}]")
    print(f"\n  (kept {kept}/{a.boot} bootstrap draws; a draw is discarded when it leaves fewer than"
          f" 3 distinct clusters)")

    # The two comparisons the paper makes. Both are differences WITHIN a draw, so the interval accounts
    # for the correlation between the policies -- they are scored on the same resampled cells.
    d_vs_adapt = boots["cka_oos"]
    d_vs_freeze = boots["cka_oos"] - boots["always_freeze"]
    d_optimism = boots["cka_insample"] - boots["cka_oos"]
    d_oracle_gap = boots["oracle"] - boots["cka_oos"]
    print(f"\n  CKA rule minus always-adapt   {p['cka_oos']:+7.2f}  CI [{ci(d_vs_adapt)[0]:+.2f},"
          f" {ci(d_vs_adapt)[1]:+.2f}]   share of draws <= 0: {share_le0(d_vs_adapt)[1]}%")
    print(f"  CKA rule minus always-freeze  {p['cka_oos'] - p['always_freeze']:+7.2f}"
          f"  CI [{ci(d_vs_freeze)[0]:+.2f}, {ci(d_vs_freeze)[1]:+.2f}]"
          f"   share of draws <= 0: {share_le0(d_vs_freeze)[1]}%")
    print(f"  in-sample optimism            {p['cka_insample'] - p['cka_oos']:+7.2f}"
          f"  CI [{ci(d_optimism)[0]:+.2f}, {ci(d_optimism)[1]:+.2f}]")
    print(f"  oracle minus CKA rule         {p['oracle'] - p['cka_oos']:+7.2f}"
          f"  CI [{ci(d_oracle_gap)[0]:+.2f}, {ci(d_oracle_gap)[1]:+.2f}]")

    # A rule whose DIRECTION is not stable across folds is not a prescription, whatever its mean value.
    orients = [v["orientation"] for v in p["chosen"].values()]
    n_low = sum(1 for o in orients if o == "low")
    ths = [v["threshold"] for v in p["chosen"].values()]
    print(f"\n  across the {len(orients)} folds the chosen rule was 'freeze when CKA is low' in"
          f" {n_low} and 'freeze when CKA is high' in {len(orients) - n_low};")
    print(f"  chosen thresholds span {min(ths):.3f}-{max(ths):.3f}")
    print(f"  in-sample choice: {ORIENT_GLOSS[p['insample_orientation']]}, cut"
          f" {p['insample_threshold']:.3f}")
    print(f"\n  the out-of-sample rule freezes {p['oos_n_frozen']} of {p['n_scored']} cells and"
          f" MISPLACES {p['oos_misplaced']} of them ({p['n_positive']} cells wanted freezing);")
    print(f"  the in-sample rule misplaces {p['insample_misplaced']}. Misplacement is a different"
          f" objective from saving, so this")
    print( "  count need not match the best-misplacement cut reported in Figure 1b.")
    d_prescribed = boots["cka_oos_prescribed"]
    print(f"\n  ACTING ON CKA AS PRESCRIBED (freeze where drift is largest) is worth"
          f" {p['cka_oos_prescribed']:+.2f} pp")
    print(f"  CI [{ci(d_prescribed)[0]:+.2f}, {ci(d_prescribed)[1]:+.2f}], share of draws <= 0:"
          f" {share_le0(d_prescribed)[1]}%")

    payload = dict(
        n_cells=n_cells, n_folds=n_folds,
        post_hoc=True,
        registered=False,
        units="percentage points of MSE against the zero-shot reference, per cell offered",
        policy_family="single global threshold on CKA, both orientations, interior cuts only",
        saving={k: (0.0 if k == "always_adapt" else float(p[k])) for _, k in rows},
        ci={k: ci(boots[k]) for _, k in rows if k != "always_adapt"},
        gaps=dict(
            cka_oos_minus_always_adapt=dict(point=float(p["cka_oos"]), ci=ci(d_vs_adapt),
                                            frac_le_zero=share_le0(d_vs_adapt)[0],
                                            pct_le_zero_nearest5=share_le0(d_vs_adapt)[1]),
            cka_oos_minus_always_freeze=dict(point=float(p["cka_oos"] - p["always_freeze"]),
                                             ci=ci(d_vs_freeze),
                                             frac_le_zero=share_le0(d_vs_freeze)[0],
                                             pct_le_zero_nearest5=share_le0(d_vs_freeze)[1]),
            insample_optimism=dict(point=float(p["cka_insample"] - p["cka_oos"]),
                                   ci=ci(d_optimism)),
            oracle_minus_cka_oos=dict(point=float(p["oracle"] - p["cka_oos"]),
                                      ci=ci(d_oracle_gap)),
            prescribed_orientation=dict(point=float(p["cka_oos_prescribed"]),
                                        ci=ci(boots["cka_oos_prescribed"]),
                                        frac_le_zero=share_le0(boots["cka_oos_prescribed"])[0],
                                        pct_le_zero_nearest5=share_le0(boots["cka_oos_prescribed"])[1],
                                        gloss=("out-of-sample value of the rule the forgetting "
                                               "literature prescribes: freeze where the "
                                               "representation moved MOST")),
        ),
        decisions=dict(oos_n_frozen=int(p["oos_n_frozen"]), oos_misplaced=int(p["oos_misplaced"]),
                       insample_misplaced=int(p["insample_misplaced"]),
                       n_wanted_freezing=int(p["n_positive"]),
                       note=("misplaced = frozen where adaptation helped, or adapted where freezing "
                             "would have. This objective is NOT the one the rule maximises (saving "
                             "is), so these counts need not match the best-misplacement cut in "
                             "fig1_diagnostic_flow.py.")),
        insample_rule=dict(threshold=float(p["insample_threshold"]),
                           orientation=p["insample_orientation"]),
        fold_rules=p["chosen"],
        fold_orientation_low=n_low, fold_orientation_high=len(orients) - n_low,
        fold_threshold_range=[float(min(ths)), float(max(ths))],
        boot_kept=int(kept), boot_requested=int(a.boot), seed=int(a.seed),
        dispersion_note=("95% percentile intervals from a cluster bootstrap. Every statistic here is "
                         "a mean of bounded per-cell quantities, unlike the out-of-sample R2 in "
                         "cka_fixed_effects.py, so the extreme percentiles are reproducible: over "
                         "seeds 0/1/7/13 the reported bounds move in the second decimal place."),
        caveats=("Post-hoc, not pre-registered. One policy family (single global CKA threshold). "
                 "denc is the held-out fine-tune-minus-frozen-encoder contrast on the cells this "
                 "paper evaluated; the saving is not a forecast for other checkpoints or datasets."),
        source="scripts/cka_decision_value.py",
    )
    if a.json:
        OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\n  wrote {OUT_JSON.relative_to(ROOT)}")
    if a.latex:
        emit_tex(payload)
        print(f"  wrote {OUT_TEX.relative_to(ROOT)}")
    return payload


LABEL = {"always_adapt": r"Always adapt (fine-tune everything) \emph{-- the baseline}",
         "always_freeze": r"Always freeze the encoder, ignoring CKA",
         "cka_oos_prescribed": r"CKA rule \emph{as prescribed}: freeze where drift is largest",
         "cka_oos": r"\textbf{CKA threshold, best orientation, chosen out of sample}",
         "cka_insample": r"\quad the same rule, threshold chosen on the cells it is scored on",
         "oracle": r"Oracle: freeze exactly the cells that wanted it"}
ORDER = ("always_adapt", "always_freeze", "cka_oos_prescribed", "cka_oos", "cka_insample", "oracle")


def pp(v):
    """A bound in whole percentage points, rendering a rounded zero as `0` rather than `+0`: the sign
    on a bound that rounded to zero from either side would claim a precision this interval has not
    got."""
    r = round(v)
    return "0" if r == 0 else f"{r:+.0f}"


def emit_tex(p):
    def row(k):
        if k == "always_adapt":
            return rf"{LABEL[k]} & $0.00$ & \emph{{by construction}} \\"
        lo, hi = p["ci"][k]
        # Whole percentage points, for the reason share_le0() gives: at two decimals these bounds do
        # not reproduce across bootstrap seeds, and a number a reader cannot reproduce is not a number.
        return rf"{LABEL[k]} & ${p['saving'][k]:+.2f}$ & $[{pp(lo)}, {pp(hi)}]$ \\"
    g = p["gaps"]
    # Rendered before the caption rather than inside it, so that no interval is split across a source
    # line: a bound and its bracket have to stay together for the whole-pp convention to be legible.
    iv = {k: f"$[{pp(v['ci'][0])}, {pp(v['ci'][1])}]$" for k, v in g.items()}
    OUT_TEX.write_text(rf"""% GENERATED by scripts/cka_decision_value.py --json --latex -- do not edit by hand.
\begin{{table}}[t]
\centering
\caption{{\textbf{{What would acting on CKA have saved?}} The same
{p['n_cells']} cells read as a decision rather than a correlation. A practitioner
choosing per cell between full fine-tuning and a frozen encoder collects
$\denc$ (positive ${{=}}$ freezing better) on every cell they freeze, so
\emph{{saving}} $=$ mean over all {p['n_cells']} cells of
$\denc \cdot \mathbf{{1}}[\text{{policy freezes}}]$, in percentage points of MSE,
against the default of always adapting. \textbf{{The out-of-sample row is the only
one a practitioner could have followed}}: the threshold and its orientation are
searched on the {p['n_folds'] - 1} training clusters and applied unchanged to the
held-out cluster, so each cell's decision is made without seeing it. The row below
it is the same search run on all cells at once, and the difference between them,
${g['insample_optimism']['point']:+.2f}$~pp {iv['insample_optimism']},
prices the optimism in a retrospectively chosen cut.
\textbf{{The headline is the interval, not the point estimate}}: the best-orientation
rule's ${p['saving']['cka_oos']:+.2f}$~pp comes with
{iv['cka_oos_minus_always_adapt']} and is ${{\leq}}0$ in
{g['cka_oos_minus_always_adapt']['pct_le_zero_nearest5']}\% of draws, so it is a
policy whose value these cells cannot establish, not one they refute.
\textbf{{And the orientation that earns it is the inverse of the prescribed one}}:
freezing where drift is \emph{{largest}} -- what forgetting mitigation actually
recommends -- is worth ${p['saving']['cka_oos_prescribed']:+.2f}$~pp
{iv['prescribed_orientation']}, and the row above it buys its point
estimate by freezing the cells that moved \emph{{least}}, a direction available only
because we searched both. That rule still puts
{p['decisions']['oos_misplaced']} of the {p['n_cells']} cells on the wrong side of
its own decision out of sample ({p['decisions']['n_wanted_freezing']} of them wanted
freezing), against {p['decisions']['insample_misplaced']} for the in-sample cut.
\textbf{{This analysis is post-hoc and descriptive}}: unlike
Table~\ref{{tab:cka_loco}}, nothing here was pre-registered, and the family searched
is a single global CKA threshold in either orientation -- a per-backbone or
multivariate rule could do better, and {p['n_folds']} clusters cannot rule that out.
Candidate cuts are interior (midpoints between consecutive distinct CKA values), so
the search cannot return \emph{{always freeze}} or \emph{{always adapt}} and have it
counted as CKA informing the decision; both are priced as rows of their own.
Intervals are 95\% percentile cluster bootstrap
({p['boot_kept']:,}/{p['boot_requested']:,} draws kept, a draw discarded when it leaves
fewer than three distinct clusters), re-running the whole
leave-one-cluster-out procedure inside each draw with duplicated clusters sharing one
fold id. Every statistic is a mean of bounded per-cell quantities, so unlike
$\Delta R^2$ in Table~\ref{{tab:cka_loco}} these percentiles are reproducible --
\textbf{{to whole percentage points, which is why they are quoted that way}}: at two
decimals the bounds move across bootstrap seeds (this table's headline lower bound
came out $-3.97$, $-3.61$, $-3.68$ and $-3.67$ on seeds 0, 1, 7 and 13), and
\texttt{{-{{}}-seedcheck}} checks every figure quoted here at its printed precision.
The point estimates and the misplacement counts carry no seed at all.}}
\label{{tab:cka_decision}}
\small
\begin{{tabular}}{{lrr}}
\toprule
Policy & Saving (pp) & 95\% cluster CI \\
\midrule
{chr(10).join(row(k) for k in ORDER)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def seedcheck(a):
    """Is every number the paper quotes reproducible across bootstrap seeds, AS ROUNDED?

    Checked rather than asserted, and checked at the precision the paper prints, which is the only
    precision the question is about: at two decimals these bounds do NOT reproduce (the out-of-sample
    rule's lower bound moved -3.97 / -3.61 / -3.68 / -3.67 over seeds 0/1/7/13), which is why the table
    quotes intervals to whole percentage points. That is the same convention and the same reason as the
    b1 interval in cka_fixed_effects.py.
    """
    f = loco_frame()
    p = policies(f, list(f["cl"]))
    rounded = {}
    for s in (0, 1, 7, 13):
        boots, kept = bootstrap(f, a.boot, np.random.default_rng(s))
        vals = {}
        for k in ORDER:
            if k == "always_adapt":
                continue
            lo, hi = ci(boots[k])
            vals[f"CI {k}"] = f"[{lo:+.0f}, {hi:+.0f}]"
        d = boots["cka_oos"]
        lo, hi = ci(boots["cka_insample"] - d)
        vals["CI insample_optimism"] = f"[{lo:+.0f}, {hi:+.0f}]"
        lo, hi = ci(boots["oracle"] - d)
        vals["CI oracle_gap"] = f"[{lo:+.0f}, {hi:+.0f}]"
        vals["share cka_oos<=0"] = f"{share_le0(d)[1]}%"
        vals["share prescribed<=0"] = f"{share_le0(boots['cka_oos_prescribed'])[1]}%"
        vals["share vs always-freeze<=0"] = f"{share_le0(d - boots['always_freeze'])[1]}%"
        rounded[s] = vals
        print(f"  seed {s:3d}  kept {kept}/{a.boot}")
        for k, v in vals.items():
            print(f"      {k:28s} {v}")
    keys = list(rounded[0])
    unstable = [k for k in keys if len({rounded[s][k] for s in rounded}) > 1]
    print(f"\n  {len(keys) - len(unstable)}/{len(keys)} quoted values identical as rounded across"
          f" seeds 0/1/7/13")
    if unstable:
        print("  NOT REPRODUCIBLE AT THIS PRECISION -- do not quote: " + ", ".join(unstable))
        for k in unstable:
            print(f"      {k}: " + " ".join(f"{s}={rounded[s][k]}" for s in rounded))
    print("\n  point estimates are seed-free (functions of the data alone): "
          + ", ".join(f"{k}={p[k]:+.2f}" for k in ORDER if k != "always_adapt"))
    print(f"  as are the decision counts: out-of-sample misplaces {p['oos_misplaced']},"
          f" in-sample {p['insample_misplaced']}, of {p['n_scored']}")
    return not unstable


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--boot", type=int, default=TEX_BOOT)
    ap.add_argument("--seed", type=int, default=TEX_SEED)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--seedcheck", action="store_true",
                    help="report the bootstrap CI under four seeds and stop")
    a = ap.parse_args()
    if a.seedcheck:
        seedcheck(a)
        return
    run(a)


if __name__ == "__main__":
    main()
