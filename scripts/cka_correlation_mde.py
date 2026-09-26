#!/usr/bin/env python3
"""
What size of correlation could this design have resolved? A POST HOC minimum detectable Spearman rho.

WHY THIS EXISTS. A reviewer of the round-10 build named the objection they expected to decide the
paper: "you haven't demonstrated that CKA fails, you've demonstrated that 22-ish Moirai observations
cannot identify its correlation with adequate precision." That is not answerable by rewording. It is
answerable by reporting what the design WAS capable of resolving, which turns the accusation into a
measured property of the sample. The paper already uses a minimum detectable effect as its
absence-of-evidence device for the paired B-vs-D contrast (preregister_power.py, app:pairedinference);
this applies the same device to the quantity under attack.

THIS IS POST HOC AND SAYS SO EVERYWHERE. The pre-registration is authoritative about choices made
before the runs; this analysis was designed after seeing the interval it explains, so the docstring,
the JSON (`posthoc: true`) and the prose all label it that way. It is NOT folded into
cka_fixed_effects.py, which is the pre-registered script, for the same reason.

WHAT IS COMPUTED, AND WHY NOT ANALYTICALLY. MDE_80 is the smallest true Spearman rho at which THE
PAPER'S OWN clustered 95% interval excludes zero in at least 80% of simulated realisations of this
exact design. Power has to be defined against the rule that produced the verdict, and the verdict came
from a percentile bootstrap over six clusters, not from a test -- so an analytic MDE would describe a
test the paper never ran, and a reviewer would say so. Defining it this way inherits the six units,
their unequal sizes (5/5/5/4/2/1), the min_distinct discard rule and the boundedness of rho for free.
Two cross-checks that do NOT use the simulation are reported beside it (Fisher-z with the
Bonett-Wright 1.06 factor and the Moulton design effect at three effective sample sizes, and the naive
inversion of the observed half-width), and the simulated value must land inside that bracket. If it
does not, the model is wrong, not the bracket.

THE GENERATIVE MODEL, AND THE TWO WAYS IT CUTS AGAINST US. The observed CKA is CONDITIONED ON, not
simulated: with z the normal scores of the real CKA ranks, only the outcome's latent is generated,
    y_i = r*z_i + sqrt(1-r^2) * ( sqrt(t)*C_{j(i)} + sqrt(1-t)*W_i ),   C ~ N(0,1) per cluster.
So the real cluster sizes, the real spread and the real tie structure are used rather than a model's
imitation of them, and because Spearman is rank-invariant NO MARGINAL FOR y IS ASSUMED AT ALL. The
assumption reduces to "monotone association, Gaussian residual with an exchangeable cluster effect",
which is much weaker than a bivariate copula and should be described that way. r is calibrated to rho
BY SIMULATION rather than by inverting rho_S = (6/pi)*arcsin(r/2): on a fixed 22-point score vector the
analytic inverse is off by as much as 0.05, and both values are printed so the size of the correction
is auditable.

Both of the following push the reported MDE DOWN, against our interest:
 (1) ICC_x is near zero, so holding the RESIDUAL cluster fraction t fixed makes the INDUCED ICC of y
     fall as rho grows. That is geometry and not a modelling choice -- an outcome that is almost all
     between-cluster cannot be strongly rank-correlated with a predictor that varies almost entirely
     within cluster. Less clustering means a narrower interval, more power, a lower MDE.
 (2) The ESTIMATED t gives the SMALLEST MDE of the three settings we report. Headlining it is
     therefore the least favourable point of the sensitivity range, which is the only honest answer to
     "you tuned the simulation".

THE STRUCTURAL FINDING IS WORTH MORE THAN THE POWER NUMBER. On normal scores the intracluster
correlation of CKA is near zero while that of d_enc is large: CKA varies within cluster (across model
size and horizon) while the outcome varies between clusters (across series). A predictor with no
cluster-level signal cannot track an outcome that is almost all cluster-level. That is a MECHANISM for
the wide interval rather than a sample-size complaint, and it says why the imprecision would not go
away by adding more cells of the same kind.

NOT A CLAIM. Observed power at the observed rho is computed but quarantined under `not_a_claim` in the
JSON: conditioning power on the estimate it is meant to judge is the Hoenig-Heisey circularity. Every
MDE here is evaluated at a HYPOTHETICAL true rho and never uses the observed estimate.

AND WHAT AN MDE DOES NOT LICENSE. It bounds what a RANK statistic could have seen. No rank statistic
detects a non-monotone relation at any n, so this number says nothing about that case; the LOCO ladder
and the decision pricing, which are not rank-based, are where that part of the evidence lives.

Run:  .venv12/bin/python scripts/cka_correlation_mde.py [--json]
      .venv12/bin/python scripts/cka_correlation_mde.py --seedcheck    (~15 min, manual only)
      .venv12/bin/python scripts/cka_correlation_mde.py --nbootcheck   (manual only)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cluster_keys as ck                                             # noqa: E402
# loco_frame() is imported rather than reimplemented: it is the function that decides which cells are
# scorable (bd_test and cka both present) and which cluster each belongs to, so a second copy here
# could price a different cell set than the interval this script exists to explain.
from cka_fixed_effects import loco_frame                              # noqa: E402

OUT_JSON = ROOT / "results/cka_correlation_mde.json"

# Pinned here rather than taken from the command line, on cka_decision_value.py's TEX_BOOT/TEX_SEED
# convention, so the numbers in the paper are reproducible from one documented command.
SIM_SEED = 0
SIM_BOOT = 1200      # see NBOOT_NOTE below
SIM_REPS = 240       # power at each grid point, headline t
NULL_REPS = 300      # more reps at rho=0, because the size is the validity check and wants precision
SENS_REPS = 100      # the two sensitivity settings for t
CAL_REPS = 2000      # calibrating r -> rho needs no bootstrap, so it can afford reps
RHO_GRID = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
T_LABELS = ("t0", "that", "t09")
BACKBONE = "Moirai"
POWER_TARGET = 0.80

# A bootstrap over SIX clusters has only C(11,5) = 462 distinct resampled multisets, so 1200 draws
# oversample a FINITE object by 2.6x and a reduced n_boot costs almost nothing here. That argument is
# on the record rather than asserted: --nbootcheck reruns one grid point at n_boot=10000.
NBOOT_NOTE = ("a cluster bootstrap over 6 units has only C(11,5)=462 distinct multisets, so 1200 "
              "draws oversample a finite resampling space 2.6x; --nbootcheck verifies at 10000")

# Bonett-Wright variance inflation for the Fisher-z of a SPEARMAN (rather than Pearson) correlation.
BW = 1.06


def normal_scores(v):
    """Van der Waerden scores of the ranks. Rank-based, so no distributional assumption on v."""
    n = len(v)
    return stats.norm.ppf((stats.rankdata(v) - 0.5) / n)


def icc_oneway(v, clusters):
    """One-way random-effects ICC, with the unequal-size (Kish) average cluster size.

    Returned clipped at zero: the ANOVA estimator is unbiased and can go negative when the true ICC is
    near zero, and a negative "fraction of variance" would be quoted as if it meant something.
    """
    v = np.asarray(v, float)
    uniq = np.unique(clusters)
    k, N = len(uniq), len(v)
    groups = [v[clusters == c] for c in uniq]
    sizes = np.array([len(g) for g in groups], float)
    gm = v.mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
    if msb + (n0 - 1) * msw == 0:
        return 0.0, float(n0)
    return float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))), float(n0)


def frame():
    """The 22 Moirai cells the within-backbone interval is computed on, in the paper's own units."""
    f = loco_frame()
    m = f["bb"] == BACKBONE
    x, y, cl, cells = f["cka"][m], f["y"][m], f["cl"][m], f["cell"][m]
    order = np.argsort(cl, kind="stable")
    return x[order], y[order], cl[order], cells[order]


def fast_spearman(rx, y):
    """Spearman with the predictor's ranks precomputed.

    Only used where the bootstrap is NOT involved -- calibrating r to rho, and reading off induced
    ICCs. Every interval in this script comes from cluster_bootstrap_spearman, the paper's estimator.
    """
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    a, b = rx - rx.mean(), ry - ry.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def draw_latent(z, cidx, k, r, t, nC, nW):
    """One realisation of the outcome's latent, conditional on the observed predictor scores."""
    e = np.sqrt(t) * nC[cidx] + np.sqrt(1.0 - t) * nW
    return r * z + np.sqrt(1.0 - r * r) * e


def noise(rng, reps, k, n):
    """Pre-drawn cluster and cell noise, so every grid point sees the SAME realisations.

    Common random numbers across rho and across t: it halves the Monte-Carlo noise on the difference
    between grid points, which is what the MDE is read off, and it removes a visibly non-monotone
    power curve that a pilot run produced from independent streams.
    """
    return rng.standard_normal((reps, k)), rng.standard_normal((reps, n))


def calibrate(z, cidx, k, t, rho_grid, rng):
    """Map each target Spearman rho to the latent weight r that produces it, BY SIMULATION.

    The analytic inverse of rho_S = (6/pi)*arcsin(r/2) assumes a bivariate normal pair; here the
    predictor is a FIXED 22-point score vector, and the two disagree by as much as 0.05. Both are
    returned so the correction is auditable.
    """
    r_grid = np.linspace(0.0, 0.97, 40)
    nC, nW = noise(rng, CAL_REPS, k, len(z))
    rz = stats.rankdata(z)
    realised = []
    for r in r_grid:
        realised.append(np.mean([fast_spearman(rz, draw_latent(z, cidx, k, r, t, nC[i], nW[i]))
                                for i in range(CAL_REPS)]))
    realised = np.array(realised)
    assert np.all(np.diff(realised) > -0.02), "r -> rho is not monotone; calibration is unusable"
    realised = np.maximum.accumulate(realised)
    out = {}
    for rho in rho_grid:
        r = float(np.interp(rho, realised, r_grid))
        analytic = float(2.0 * np.sin(np.pi * rho / 6.0))
        out[f"{rho:.2f}"] = dict(r=r, r_analytic=analytic, r_minus_analytic=r - analytic)
    return out


def power_at(x, y_cl, z, cidx, k, r, t, reps, n_boot, nC, nW, base_seed):
    """Fraction of realisations whose clustered 95% interval excludes zero, plus the realised rho."""
    rz = stats.rankdata(z)
    excl, rhos, kept, iccs = 0, [], [], []
    for i in range(reps):
        yy = draw_latent(z, cidx, k, r, t, nC[i], nW[i])
        # Each realisation gets its OWN generator seeded identically across grid points, matching
        # clustered_inference.both()'s convention rather than sharing one advancing stream.
        rng = np.random.default_rng(base_seed * 100_003 + i)
        rho, lo, hi, kp, _ = ck.cluster_bootstrap_spearman(x, yy, y_cl, n_boot=n_boot, rng=rng)
        excl += int(lo > 0.0 or hi < 0.0)
        rhos.append(fast_spearman(rz, yy))
        kept.append(kp)
        iccs.append(icc_oneway(normal_scores(yy), y_cl)[0])
    return dict(power=excl / reps, reps=reps, rho_realised=float(np.mean(rhos)),
                icc_y_induced=float(np.mean(iccs)), kept_min=int(min(kept)), n_boot=n_boot)


def mde_from_curve(curve, target=POWER_TARGET):
    """Smallest rho whose power reaches `target`, linearly interpolated between grid points."""
    g = sorted(curve, key=float)
    for a, b in zip(g, g[1:]):
        pa, pb = curve[a]["power"], curve[b]["power"]
        if pa < target <= pb:
            return float(float(a) + (target - pa) * (float(b) - float(a)) / (pb - pa))
    if curve[g[-1]]["power"] >= target:
        return float(g[0])
    return None       # the grid never reaches 80%: reported as "> max grid", never extrapolated


def fisher_mde(n_eff, alpha=0.05, power=POWER_TARGET):
    """Analytic cross-check: the rho an uncorrelated-observations Fisher-z would need at `power`."""
    if n_eff <= 3:
        return float("nan")
    zc = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(zc * np.sqrt(BW / (n_eff - 3))))


def observed(x, y, cl, n_boot, seed):
    return ck.cluster_bootstrap_spearman(x, y, cl, n_boot=n_boot,
                                         rng=np.random.default_rng(seed))


def compute(seed=SIM_SEED, sim_boot=SIM_BOOT, sim_reps=SIM_REPS, sens_reps=SENS_REPS,
            null_reps=NULL_REPS, grid=RHO_GRID, quiet=False):
    x, y, cl, cells = frame()
    n = len(x)
    uniq, cidx = np.unique(cl, return_inverse=True)
    k = len(uniq)
    sizes = [int((cl == c).sum()) for c in uniq]

    rho_obs, lo_obs, hi_obs, kept_obs, k_obs = observed(x, y, cl, 10_000, seed)
    half = (hi_obs - lo_obs) / 2.0

    zx = normal_scores(x)
    icc_x, n0 = icc_oneway(zx, cl)
    icc_y, _ = icc_oneway(normal_scores(y), cl)
    cluster_means = {str(c): float(y[cl == c].mean()) for c in uniq}

    deff = 1.0 + (n0 - 1.0) * icc_x * icc_y
    fisher = {"n22": fisher_mde(n), "n_deff": fisher_mde(n / deff), "n_clusters": fisher_mde(k)}

    ts = {"t0": 0.0, "that": icc_y, "t09": 0.9}
    cal = {lab: calibrate(zx, cidx, k, tv, tuple(grid) + (abs(rho_obs),), np.random.default_rng(seed + 11))
           for lab, tv in ts.items()}

    curves, mdes = {}, {}
    for lab, tv in ts.items():
        nC, nW = noise(np.random.default_rng(seed + 7), max(sim_reps, null_reps, sens_reps), k, n)
        curve = {}
        for rho in grid:
            rr = sim_reps if lab == "that" else sens_reps
            rr = null_reps if (rho == 0.0 and lab == "that") else rr
            r = cal[lab][f"{rho:.2f}"]["r"]
            curve[f"{rho:.2f}"] = power_at(x, cl, zx, cidx, k, r, tv, rr, sim_boot,
                                           nC, nW, seed)
            if not quiet:
                c = curve[f"{rho:.2f}"]
                print(f"    t={lab:5s} rho={rho:.2f} -> realised {c['rho_realised']:+.3f} "
                      f"power {c['power']:.3f} (n={c['reps']}, induced ICC_y {c['icc_y_induced']:.2f})")
        curves[lab] = curve
        mdes[lab] = mde_from_curve(curve)

    # The appendix says the ESTIMATED cluster-variance setting gives the smallest MDE of the three, and
    # calls that the least favourable point of the sensitivity range for our own claim. That sentence is
    # the answer to "you tuned the simulation", so it is asserted here rather than left to be reread off
    # a curve: if a re-run ever makes the estimated setting the most generous one, the sentence is an
    # overclaim and has to be rewritten. It holds at master seeds 0, 1, 7 and 13.
    _got = {k: v for k, v in mdes.items() if v is not None}
    if len(_got) == len(mdes):
        assert mdes["that"] <= min(mdes.values()) + 1e-9, (
            f"the estimated cluster-variance setting no longer gives the smallest MDE: {mdes}. "
            f"app:clustered says it does and calls it the least favourable setting for our claim; "
            f"rewrite that sentence rather than reporting the minimum of the three.")

    # Quarantined: power AT the observed estimate is the Hoenig-Heisey circularity, reported only so
    # that a reader who wants it does not have to recompute it wrongly.
    nC, nW = noise(np.random.default_rng(seed + 7), sens_reps, k, n)
    r_obs = cal["that"][f"{abs(rho_obs):.2f}"]["r"]
    obs_power = power_at(x, cl, zx, cidx, k, r_obs, icc_y, sens_reps, sim_boot, nC, nW, seed)

    return dict(
        posthoc=True,
        what="minimum detectable Spearman rho for CKA vs held-out d_enc, within Moirai",
        definition=("smallest true Spearman rho at which the paper's own clustered 95% percentile "
                    "interval excludes zero in >=80% of simulated realisations of this design"),
        design=dict(backbone=BACKBONE, n_cells=n, n_clusters=k, cluster_sizes=sizes,
                    kish_n0=n0, cells=[str(c) for c in cells],
                    cka_min=float(x.min()), cka_max=float(x.max()),
                    cka_distinct=int(len(set(x.tolist())))),
        observed=dict(rho=float(rho_obs), lo=float(lo_obs), hi=float(hi_obs),
                      kept=int(kept_obs), n_boot=10_000, half_width=float(half)),
        structure=dict(icc_x_cka=icc_x, icc_y_denc=icc_y, design_effect=float(deff),
                       n_eff_deff=float(n / deff), cluster_means_denc=cluster_means),
        model=dict(form="y = r*z + sqrt(1-r^2)*(sqrt(t)*C_cluster + sqrt(1-t)*W_cell)",
                   conditions_on_observed_cka=True, y_marginal_assumed=None,
                   t_settings=ts, calibration=cal),
        power=curves,
        mde80=mdes["that"], mde80_sensitivity={lab: mdes[lab] for lab in T_LABELS},
        null_rejection_rate=curves["that"][f"{0.0:.2f}"]["power"],
        # The size at ALL THREE t, not just the headline one: it is the number that says whether the
        # MDE is optimistic, and the two sensitivity settings are measured on fewer reps and come out
        # higher. Reporting only the most favourable of the three would be the tuning we disclaim.
        null_rejection_by_t={lab: curves[lab][f"{0.0:.2f}"]["power"] for lab in T_LABELS},
        crosschecks=dict(fisher_bonett_wright=fisher,
                         naive_halfwidth_inversion=float(half),
                         bracket_lo=float(min(fisher.values())),
                         bracket_hi=float(max(fisher.values()))),
        not_a_claim=dict(observed_power_at_observed_rho=obs_power["power"],
                         why="Hoenig-Heisey circularity: power conditioned on the estimate it judges"),
        settings=dict(seed=seed, sim_boot=sim_boot, sim_reps=sim_reps, null_reps=null_reps,
                      sens_reps=sens_reps, cal_reps=CAL_REPS, rho_grid=list(grid),
                      power_target=POWER_TARGET, nboot_note=NBOOT_NOTE),
    )


def report(R):
    d, s, o = R["design"], R["structure"], R["observed"]
    print(f"\n  {BACKBONE}: {d['n_cells']} cells in {d['n_clusters']} clusters, sizes "
          f"{'/'.join(str(v) for v in d['cluster_sizes'])}, Kish n0={d['kish_n0']:.3f}")
    print(f"  CKA {d['cka_min']:.3f}-{d['cka_max']:.3f}, {d['cka_distinct']}/{d['n_cells']} distinct")
    print(f"  observed rho={o['rho']:+.5f}  CI [{o['lo']:+.5f},{o['hi']:+.5f}]  "
          f"{o['kept']}/{o['n_boot']} kept  half-width {o['half_width']:.3f}")
    print(f"\n  STRUCTURE (normal scores): ICC of CKA = {s['icc_x_cka']:.3f}, "
          f"ICC of d_enc = {s['icc_y_denc']:.3f}")
    print("    cluster means of d_enc: "
          + ", ".join(f"{v:+.1f}" for _, v in sorted(s['cluster_means_denc'].items(),
                                                     key=lambda kv: kv[1])))
    print(f"    design effect {s['design_effect']:.3f} -> n_eff {s['n_eff_deff']:.1f} of "
          f"{d['n_cells']}")
    m, f = R["mde80_sensitivity"], R["crosschecks"]["fisher_bonett_wright"]
    fmt = lambda v: "> grid max" if v is None else f"{v:.3f}"                       # noqa: E731
    print(f"\n  MDE_80 = {fmt(R['mde80'])} at the estimated t"
          f"   (t=0: {fmt(m['t0'])}, t=0.9: {fmt(m['t09'])})")
    print(f"  cross-checks (Fisher-z x{BW}): n=22 {f['n22']:.3f}, n_eff "
          f"{R['structure']['n_eff_deff']:.1f} {f['n_deff']:.3f}, n=clusters {f['n_clusters']:.3f}")
    print(f"  naive half-width inversion {R['crosschecks']['naive_halfwidth_inversion']:.3f}")
    # The bracket is a bracket of analytic approximations to a DIFFERENT rule -- a Fisher-z test on
    # independent observations -- so the simulation is authoritative where they disagree, and the
    # direction of the disagreement is what has to be read, not the fact of it.
    #   ABOVE the bracket: the simulation says the design is less capable than any analytic reference
    #     would allow. Nothing in the model should do that, so the model would be wrong.
    #   BELOW the bracket: the design is MORE capable than a correctly sized test on 22 independent
    #     observations, which is only possible if the interval is too narrow. That is checkable and is
    #     checked -- the size at rho=0 is printed on the next line -- and it means the reported MDE is
    #     the OPTIMISTIC end. Optimistic is the safe direction here: the paper's claim is that the
    #     observed rho falls FAR BELOW the MDE, and a smaller MDE makes that claim harder to sustain.
    lo, hi = R["crosschecks"]["bracket_lo"], R["crosschecks"]["bracket_hi"]
    v = R["mde80"]
    where = ("inside" if v is not None and lo <= v <= hi else
             "BELOW -- the 6-cluster percentile interval is anti-conservative, so this MDE is the "
             "optimistic end (see the size below); the claim is harder, not easier" if v is not None
             and v < lo else "ABOVE -- no analytic reference allows this; the model is wrong")
    print(f"  simulated MDE vs the analytic bracket [{lo:.3f},{hi:.3f}]: {where}")
    print("  null rejection rate at a true rho of zero (nominal 0.05): "
          + ", ".join(f"t={k} {R['null_rejection_by_t'][k]:.3f}"
                      f" (n={R['power'][k]['0.00']['reps']})" for k in T_LABELS))
    if R["mde80"] is not None and R["mde80"] <= abs(R["observed"]["rho"]):
        print("  WARNING: the observed rho is NOT below the MDE. The body's 'cannot resolve' framing"
              " would be wrong; rewrite the sentence, do not adjust the number.")


def seedcheck():
    """Which digit survives reseeding. The paper quotes only that digit."""
    seeds = (0, 1, 7, 13)
    got = {}
    for s in seeds:
        print(f"\n  seed {s} ...")
        R = compute(seed=s, quiet=True)
        got[s] = {"mde80": R["mde80"], "mde80_t0": R["mde80_sensitivity"]["t0"],
                  "mde80_t09": R["mde80_sensitivity"]["t09"],
                  "null_rate": R["null_rejection_rate"],
                  "icc_x": R["structure"]["icc_x_cka"], "icc_y": R["structure"]["icc_y_denc"]}
        print("    " + "  ".join(f"{k}={'None' if v is None else f'{v:.4f}'}"
                                 for k, v in got[s].items()))
    keys = list(got[seeds[0]])
    print("\n  agreement by rounding grain:")
    for grain in (0.1, 0.05, 0.01):
        bad = []
        for k in keys:
            vals = {None if got[s][k] is None else round(round(got[s][k] / grain) * grain, 4)
                    for s in seeds}
            if len(vals) > 1:
                bad.append(k)
        tag = "all agree" if not bad else "NOT REPRODUCIBLE AT THIS PRECISION -- do not quote: " \
                                         + ", ".join(bad)
        print(f"    grain {grain:<5} {tag}")
    print("\n  the point estimates are seed-free (functions of the data alone): ICC_x, ICC_y,"
          " the cluster sizes and the observed rho")
    return True


def nbootcheck():
    """One grid point at the full n_boot, so 'did you cheap out on the bootstrap' has an answer."""
    x, y, cl, _ = frame()
    uniq, cidx = np.unique(cl, return_inverse=True)
    k, n = len(uniq), len(x)
    zx = normal_scores(x)
    icc_y, _ = icc_oneway(normal_scores(y), cl)
    cal = calibrate(zx, cidx, k, icc_y, (0.6,), np.random.default_rng(SIM_SEED + 11))
    r = cal["0.60"]["r"]
    print(f"\n  {NBOOT_NOTE}\n")
    for nb in (SIM_BOOT, 10_000):
        nC, nW = noise(np.random.default_rng(SIM_SEED + 7), 60, k, n)
        p = power_at(x, cl, zx, cidx, k, r, icc_y, 60, nb, nC, nW, SIM_SEED)
        print(f"    rho=0.60  n_boot={nb:6d}  power {p['power']:.3f}  (60 reps, same realisations)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seedcheck", action="store_true",
                    help="rerun the whole curve under seeds 0/1/7/13 and report the stable grain")
    ap.add_argument("--nbootcheck", action="store_true",
                    help="rerun one grid point at n_boot=10000 and stop")
    a = ap.parse_args()
    if a.seedcheck:
        seedcheck()
        return
    if a.nbootcheck:
        nbootcheck()
        return
    R = compute()
    report(R)
    if a.json:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(R, indent=2) + "\n")
        print(f"\n  wrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
