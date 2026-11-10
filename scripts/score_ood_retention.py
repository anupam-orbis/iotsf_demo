#!/usr/bin/env python3
"""OUT-OF-DOMAIN RETENTION, STEP 3 of 3: score the runs exactly as
results/v59_ood_retention/preregistration.json fixes, and decide nothing it did not.

Readout:   retention_pct = (MSE_finetuned - MSE_zeroshot) / MSE_zeroshot * 100 per (cell, ood dataset,
           arm, seed), read from the instrument's own retention_*.json; positive = capability lost.
Contrast:  ret_B - ret_H (primary) and ret_B - ret_D (secondary), paired on seed.
Harmful:   ret_B - ret_H >= 10 pp in EVERY seed of a (cell, ood dataset) pair.
CKA rule:  Spearman(1 - CKA on the ood inputs, ret_B) over all B (cell, ood, seed) records >= 0.5,
           AND a 95% percentile bootstrap interval that resamples whole cells excludes 0
           (n_boot 10000, seed 0).
Outcome:   O1 harmful AND CKA rule; O2 harmful AND NOT CKA rule; O3 no harmful pair.
Also:      every rerun's target-task test MSE beside the matrix record it repeats (same seed). The
           reruns were made on an A10G (CUDA); the matrix records on Apple MPS. Disagreement is
           reported, never corrected for.

Run:  python3 scripts/score_ood_retention.py      (writes results/v59_ood_retention/scored.json)
"""
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v59_ood_retention"
REG = json.loads((BASE / "preregistration.json").read_text())
HARM = REG["thresholds"]["harmful_forgetting_pp"]
RHO_MIN, N_BOOT, BOOT_SEED = 0.5, 10000, 0


def name_of(c):
    return f"{c['size']}_{c['dataset']}_h{c['horizon']}"


def load():
    """(cell, ood, arm, seed) -> retention record; every registered record must exist."""
    R, missing = {}, []
    for c in REG["cells"]:
        for arm in REG["arms"]:
            for s in REG["seeds"]:
                for o in c["ood_datasets"]:
                    f = BASE / name_of(c) / f"condition_{arm}" / f"s{s}" / f"retention_{o}.json"
                    if f.exists():
                        R[(name_of(c), o, arm, s)] = json.loads(f.read_text())
                    else:
                        missing.append(str(f.relative_to(ROOT)))
    assert not missing, f"{len(missing)} registered retention records missing, e.g. {missing[:3]}"
    return R


def interval(x):
    m = st.mean(x)
    se = st.stdev(x) / len(x) ** 0.5
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return dict(mean=m, sem=se, lo=m - h, hi=m + h, n=len(x))


def spearman_cell_boot(rows):
    """Spearman over the B records, and the percentile interval from resampling whole cells."""
    x = np.array([1 - r["cka"] for r in rows])
    y = np.array([r["ret"] for r in rows])
    rho = stats.spearmanr(x, y)[0]
    cells = sorted({r["cell"] for r in rows})
    idx = {c: [i for i, r in enumerate(rows) if r["cell"] == c] for c in cells}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        take = [i for c in rng.choice(cells, len(cells), replace=True) for i in idx[c]]
        xb, yb = x[take], y[take]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            continue
        boots.append(stats.spearmanr(xb, yb)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rho=float(rho), lo=float(lo), hi=float(hi), n=len(rows), n_cells=len(cells),
                kept=len(boots), meets=bool(rho >= RHO_MIN and (lo > 0 or hi < 0)))


def matrix_record(c, arm, seed):
    """The matrix's own record of this (cell, arm, seed), read by fields rather than by path."""
    for f in glob.glob(str(ROOT / "results" / c["matrix_dir"] / "**/*.json"), recursive=True):
        if c["matrix_dir"] == "v47_prospective" and f"/{name_of(c)}/" not in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if (isinstance(d, dict) and d.get("condition") == arm and d.get("seed") == seed
                and (d.get("model_size") or "small") == c["size"] and d.get("horizon") == c["horizon"]
                and d.get("max_train_samples") == c["n_train"] and "test_mse" in d):
            return d, Path(f).relative_to(ROOT).as_posix()
    return None, None


def reproducibility():
    out = []
    for c in REG["cells"]:
        for arm in ("B", "D"):  # H was never part of the matrix
            for s in REG["seeds"]:
                f = BASE / name_of(c) / f"condition_{arm}" / f"s{s}" / f"condition_{arm}_h{c['horizon']}_s{s}.json"
                new = json.loads(f.read_text())
                old, path = matrix_record(c, arm, s)
                out.append(dict(cell=name_of(c), arm=arm, seed=s, rerun_test_mse=new["test_mse"],
                                matrix_test_mse=old["test_mse"] if old else None, matrix_path=path,
                                rel_diff_pct=(new["test_mse"] - old["test_mse"]) / old["test_mse"] * 100
                                if old else None,
                                rerun_cka=new["final_cka"], matrix_cka=old.get("final_cka") if old else None))
    return out


def emit_tex(pairs, path=ROOT / "paper_8/tables/ood_retention.tex"):
    """One row per (cell, ood dataset): mean retention per arm, B-H with its interval, CKA on the ood inputs."""
    pretty = {"small": "S", "base": "B", "large": "L"}
    L = ["% GENERATED by scripts/score_ood_retention.py -- do not edit by hand.",
         "\\begin{tabular}{@{}llrrrrcr@{}}", "\\toprule",
         "Cell & OOD set & ret$_\\text{B}$ & ret$_\\text{D}$ & ret$_\\text{H}$ & B$-$H & 95\\% CI & "
         "CKA$_\\text{B}^\\text{ood}$ \\\\", "\\midrule"]
    prev = None
    for p in pairs:
        size, ds, h = p["cell"].split("_")
        lab = f"{pretty[size]}/{ds} {h}" if p["cell"] != prev else ""
        if prev is not None and p["cell"] != prev:
            L.append("\\addlinespace[2pt]")
        prev = p["cell"]
        L.append(f"{lab} & {p['ood'].replace('Electricity7', 'Elec.')} & "
                 f"${p['ret']['B']['mean']:+.1f}$ & ${p['ret']['D']['mean']:+.1f}$ & "
                 f"${p['ret']['H']['mean']:+.1f}$ & ${p['bh']['mean']:+.1f}$ & "
                 f"$[{p['bh']['lo']:+.1f}, {p['bh']['hi']:+.1f}]$ & ${p['cka_ood_B']:.3f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def main():
    R = load()
    pairs = []
    for c in REG["cells"]:
        for o in c["ood_datasets"]:
            ret = {a: [R[(name_of(c), o, a, s)]["retention_A_pct"] for s in REG["seeds"]] for a in REG["arms"]}
            bh = [b - h for b, h in zip(ret["B"], ret["H"])]
            bd = [b - d for b, d in zip(ret["B"], ret["D"])]
            pairs.append(dict(cell=name_of(c), ood=o,
                              ret={a: interval(v) for a, v in ret.items()},
                              ret_B_seeds=ret["B"], bh_seeds=bh,
                              bh=interval(bh), bd=interval(bd),
                              harmful=all(v >= HARM for v in bh),
                              cka_ood_B=st.mean(R[(name_of(c), o, "B", s)]["cka_on_task_a_inputs"]
                                                for s in REG["seeds"]),
                              cka_ood_H_min=min(R[(name_of(c), o, "H", s)]["cka_on_task_a_inputs"]
                                                for s in REG["seeds"])))
    b_rows = [dict(cell=k[0], ood=k[1], seed=k[3], cka=v["cka_on_task_a_inputs"], ret=v["retention_A_pct"])
              for k, v in R.items() if k[2] == "B"]
    sp = spearman_cell_boot(b_rows)
    # NOT REGISTERED, descriptive only: CKA levels differ by model size (Small 0.8-0.96, Base/Large
    # 0.4-0.7), so the pooled rho could be a size contrast. Within-size rho says whether it is.
    within = {}
    for size in ("small", "base", "large"):
        rr = [r for r in b_rows if r["cell"].startswith(size + "_")]
        if len({r["cell"] for r in rr}) >= 1 and len(rr) >= 4:
            within[size] = dict(rho=float(stats.spearmanr([1 - r["cka"] for r in rr], [r["ret"] for r in rr])[0]),
                                n=len(rr), n_cells=len({r["cell"] for r in rr}))
    harmful = [p for p in pairs if p["harmful"]]
    outcome = ("O1" if sp["meets"] else "O2") if harmful else "O3"
    repro = reproducibility()
    allret = {a: [v["retention_A_pct"] for k, v in R.items() if k[2] == a] for a in REG["arms"]}

    res = dict(registration_commit=(BASE / "REGISTRATION_COMMIT").read_text().strip()
               if (BASE / "REGISTRATION_COMMIT").exists() else None,
               outcome=outcome, n_pairs=len(pairs), n_harmful=len(harmful),
               harmful_pairs=[(p["cell"], p["ood"]) for p in harmful],
               spearman=sp, spearman_within_size_unregistered=within, pairs=pairs, reproducibility=repro,
               pooled={a: dict(mean=st.mean(v), n=len(v), n_worse=sum(x > 0 for x in v),
                               max=max(v), min=min(v)) for a, v in allret.items()})
    (BASE / "scored.json").write_text(json.dumps(res, indent=1))
    emit_tex(pairs)

    print(f"{'cell':18s} {'ood':13s} {'ret_B':>14s} {'ret_D':>7s} {'ret_H':>7s} {'B-H':>16s} {'min B-H':>8s} "
          f"{'CKA_B(ood)':>10s}")
    for p in pairs:
        print(f"{p['cell']:18s} {p['ood']:13s} {p['ret']['B']['mean']:+6.1f}+-{p['ret']['B']['sem']:4.1f}   "
              f"{p['ret']['D']['mean']:+6.1f} {p['ret']['H']['mean']:+6.1f}  "
              f"{p['bh']['mean']:+6.1f} [{p['bh']['lo']:+5.1f},{p['bh']['hi']:+5.1f}] {min(p['bh_seeds']):+7.1f}  "
              f"{p['cka_ood_B']:.3f}{'  HARMFUL' if p['harmful'] else ''}")
    for a, v in res["pooled"].items():
        print(f"pooled ret_{a}: mean {v['mean']:+.1f}%, {v['n_worse']}/{v['n']} records worse than zero-shot, "
              f"range [{v['min']:+.1f}, {v['max']:+.1f}]")
    print(f"min CKA_H on ood inputs: {min(p['cka_ood_H_min'] for p in pairs):.6f}")
    print(f"Spearman(1-CKA_ood, ret_B) {sp['rho']:+.3f}  n {sp['n']}  cells {sp['n_cells']}  "
          f"cell-bootstrap 95% [{sp['lo']:+.3f}, {sp['hi']:+.3f}]  (kept {sp['kept']}/{N_BOOT})  "
          f"rule met: {sp['meets']}")
    for k, v in within.items():
        print(f"   (unregistered) within Moirai-{k}: rho {v['rho']:+.3f}  n {v['n']}  cells {v['n_cells']}")
    print("reproducibility (target test MSE, rerun on CUDA vs matrix record on MPS):")
    for r in repro:
        d = f"{r['rel_diff_pct']:+6.1f}%" if r["rel_diff_pct"] is not None else "  (no matrix record)"
        print(f"   {r['cell']:18s} {r['arm']} s{r['seed']:<4d} rerun {r['rerun_test_mse']:.4f}  "
              f"matrix {r['matrix_test_mse'] if r['matrix_test_mse'] is None else round(r['matrix_test_mse'], 4)}  {d}")
    for arm in ("B", "D"):
        d = [abs(r["rel_diff_pct"]) for r in repro if r["arm"] == arm and r["rel_diff_pct"] is not None]
        print(f"   {arm}: median |rerun - matrix| {st.median(d):.1f}%, max {max(d):.1f}%, n {len(d)}")
    print(f"harmful pairs: {len(harmful)} of {len(pairs)}  {res['harmful_pairs']}")
    print(f"REGISTERED OUTCOME: {outcome}")


if __name__ == "__main__":
    main()
