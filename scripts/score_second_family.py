#!/usr/bin/env python3
"""SECOND-FAMILY ARM (Traffic), STEP 3 of 3: score the runs exactly as
results/v60_second_family/preregistration.json fixes, and decide nothing it did not.

Chronos:  d_enc = (test_mse_B - test_mse_D) / zs_test * 100, paired on seed, D's head by the matrix
          rule (closed-form ridge if its validation loss is within 5% of the AdamW head's), zs_test
          the mean of the two runs' own held-out zero-shot, as scripts/cell_matrix.chronos_cells does.
TimesFM:  d_enc = forgetting_pct_test(B) - forgetting_pct_test(H) (primary) and -D (secondary).
Call:     two-sided 95% t-interval on the paired differences; above 0 'freezing better', below 0
          'adapting better', else 'inconclusive'.
Degradation (TimesFM only): selection gate >= 0.20 AND forg_B > 0 in every seed AND forg_frozen < 0
          in every seed.
Strict-freeze check: every Chronos D and TimesFM H run must read CKA 1.000.

Run:  python3 scripts/score_second_family.py      (writes results/v60_second_family/scored.json)
"""
import glob
import json
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v60_second_family"
REG = json.loads((BASE / "preregistration.json").read_text())
GATE = 0.20


def interval(x):
    m, se = st.mean(x), st.stdev(x) / len(x) ** 0.5
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return dict(mean=m, sem=se, lo=m - h, hi=m + h, n=len(x), n_pos=sum(v > 0 for v in x),
                call="freezing better" if m - h > 0 else
                     "adapting better" if m + h < 0 else "inconclusive")


def load(pattern):
    return {json.loads(Path(f).read_text())["seed"]: json.loads(Path(f).read_text())
            for f in glob.glob(str(BASE / pattern))}


def chronos():
    B = load("chronos/cond_B/mse_traffic/seed*/condition_B_s*.json")
    D = load("chronos/cond_D/mse_traffic/seed*/condition_D_s*.json")
    seeds = sorted(set(B) & set(D))
    assert seeds == REG["seeds"]["chronos"], seeds
    rows, d = [], []
    for s in seeds:
        b, dd = B[s], D[s]
        ridge_ok = dd["ridge_optimum"]["best_val_loss"] <= dd["best_val_loss"] * 1.05
        dt = dd["test_mse_per_element_ols"] if ridge_ok else dd["test_mse_per_element_adamw"]
        zst = (dd["zs_mse_test"] + b["zs_mse_test"]) / 2
        d.append((b["test_mse_per_element"] - dt) / zst * 100)
        rows.append(dict(seed=s, d_enc=d[-1], cka_B=b["final_cka"], cka_D=dd["final_cka"],
                         drift_B=b["weight_drift"], best_epoch_B=b["best_epoch"],
                         head_D="ridge" if ridge_ok else "adamw"))
    strict = [r["cka_D"] for r in rows]
    return dict(per_seed=rows, d_enc=interval(d), cka_B=st.mean(r["cka_B"] for r in rows),
                strict_check=dict(min_cka_D=min(strict), ok=min(strict) > 0.9995))


def timesfm():
    base = "timesfm/Traffic_h24/condition_{c}/condition_{c}_h24_s*.json"
    R = {c: load(base.format(c=c)) for c in "BDH"}
    seeds = sorted(set(R["B"]) & set(R["D"]) & set(R["H"]))
    assert seeds == REG["seeds"]["timesfm"], seeds
    f = {c: [R[c][s]["forgetting_pct_test"] for s in seeds] for c in "BDH"}
    rows = [dict(seed=s, forg_B=R["B"][s]["forgetting_pct_test"],
                 forg_D=R["D"][s]["forgetting_pct_test"], forg_H=R["H"][s]["forgetting_pct_test"],
                 cka_B=R["B"][s]["final_cka"], cka_D=R["D"][s]["final_cka"],
                 cka_H=R["H"][s]["final_cka"], drift_B=R["B"][s]["final_weight_drift"],
                 best_epoch_B=R["B"][s]["best_epoch"]) for s in seeds]
    gate = REG["screen"]["traffic_gates"]["timesfm_traffic_selection"]
    degr = {fr: bool(gate >= GATE and all(x > 0 for x in f["B"]) and all(x < 0 for x in f[fr]))
            for fr in "HD"}
    return dict(per_seed=rows,
                d_enc_H=interval([b - h for b, h in zip(f["B"], f["H"])]),
                d_enc_D=interval([b - d for b, d in zip(f["B"], f["D"])]),
                forg={c: interval(f[c]) for c in "BDH"},
                cka_B=st.mean(r["cka_B"] for r in rows),
                degradation=degr,
                strict_check=dict(min_cka_H=min(r["cka_H"] for r in rows),
                                  ok=min(r["cka_H"] for r in rows) > 0.9995))


def outcome(c, t):
    if t["degradation"]["H"] or t["degradation"]["D"]:
        return "O1"
    if "freezing better" in (c["d_enc"]["call"], t["d_enc_H"]["call"]):
        return "O2"
    return "O3"


def main():
    c, t = chronos(), timesfm()
    res = dict(registration_commit="64fefee", chronos=c, timesfm=t, outcome=outcome(c, t))
    (BASE / "scored.json").write_text(json.dumps(res, indent=1))

    def fmt(i):
        return (f"{i['mean']:+.1f} +- {i['sem']:.1f} SEM, 95% [{i['lo']:+.1f}, {i['hi']:+.1f}], "
                f"{i['n_pos']}/{i['n']} positive -> {i['call']}")
    print("CHRONOS-T5-Small / Traffic h=24  (d_enc = B - D, D strict)")
    print(f"  d_enc  {fmt(c['d_enc'])}")
    print(f"  CKA_B mean {c['cka_B']:.3f}; strict check min CKA_D {c['strict_check']['min_cka_D']:.6f}")
    for r in c["per_seed"]:
        print(f"    s{r['seed']}: d_enc {r['d_enc']:+.1f}  CKA_B {r['cka_B']:.3f}  "
              f"best_ep {r['best_epoch_B']}  head_D {r['head_D']}")
    print("TIMESFM-2.5 / Traffic h=24")
    print(f"  d_enc (B-H, primary)   {fmt(t['d_enc_H'])}")
    print(f"  d_enc (B-D, secondary) {fmt(t['d_enc_D'])}")
    for k in "BDH":  # a forgetting interval below 0 means the arm improved on zero-shot
        i = t["forg"][k]
        print(f"  forg_{k} {i['mean']:+.1f} +- {i['sem']:.1f} SEM, 95% [{i['lo']:+.1f}, {i['hi']:+.1f}], "
              f"{i['n'] - i['n_pos']}/{i['n']} seeds improve on zero-shot")
    print(f"  CKA_B mean {t['cka_B']:.3f}; strict check min CKA_H {t['strict_check']['min_cka_H']:.6f}")
    print(f"  degradation definition met: {t['degradation']}")
    for r in t["per_seed"]:
        print(f"    s{r['seed']}: B {r['forg_B']:+.1f}  D {r['forg_D']:+.1f}  H {r['forg_H']:+.1f}  "
              f"CKA_B {r['cka_B']:.3f}  best_ep {r['best_epoch_B']}")
    print(f"REGISTERED OUTCOME: {res['outcome']}")


if __name__ == "__main__":
    main()
