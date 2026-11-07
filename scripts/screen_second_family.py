#!/usr/bin/env python3
"""Capability screen for the second-family arm: Chronos-T5-Small and TimesFM-2.5 on new datasets.

WHY. Every non-Moirai cell in the paper fails the fitted screen (Appendix app:gatecorrection), so the
paper's evidence on whether drift tracks the intervention is Moirai-only where it matters (u3rt Q2).
This decides, BEFORE any training, which (backbone, dataset) cells a matched B/D/H arm could
interpret: an intervention is only readable where the released checkpoint beats the fitted baseline.
A screen with no passing cell is itself the answer and is reported as such.

PROTOCOL, identical to the existing non-Moirai arm: univariate OT, lookback 96, h=24, per-window
z-scored MSE, the trainers' own load_series/build_windows, and the fitted lookback-96 ridge map
(gate_all_cells._window_linear_mse, baseline="fitted") fit on all training windows. Two window sets:
  selection: build_windows(val, ..., max_windows=200, seed=42)  -- the primary split
  held-out:  build_windows(test, ..., max_windows=200, seed=0)  -- the retrospective variant
The constant training-mean predictor is scored too: a denominator worse than it cannot support a
claim of pre-trained advantage, so the fitted map's admissibility is checked rather than assumed.
Chronos's zero-shot is the median of 20 samples, which draws from the torch RNG; it is seeded here,
and the cross-device gap this leaves (~1% on ETTh1) is stated in the output.

Run:  HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/screen_second_family.py --device cuda
Writes results/v60_second_family/screen.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from chronos_mse_finetune import MODEL_ID, build_windows, chronos_zs_mse, load_series  # noqa: E402
from gate_all_cells import GATE_THRESHOLD, _window_linear_mse  # noqa: E402

LOOKBACK, HORIZON = 96, 24
SPLITS = {"selection": ("val", 42), "held-out": ("test", 0)}


def windows(ds):
    tr, va, te = load_series(ds)
    series = {"val": va, "test": te}
    out = {name: build_windows(series[s], LOOKBACK, HORIZON, max_windows=200, seed=seed)
           for name, (s, seed) in SPLITS.items()}
    return build_windows(tr, LOOKBACK, HORIZON), out


def denominators(ds, train_w, ctx, tgt):
    return {b: _window_linear_mse(ctx, tgt, LOOKBACK, HORIZON, train=train_w, baseline=b,
                                  dataset=ds)
            for b in ("fitted", "constant")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Traffic", "Exchange", "ILI"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/v60_second_family/screen.json")
    args = ap.parse_args()

    from chronos import ChronosPipeline
    from timesfm_common import assert_no_ar, batched_point_mse, load_timesfm
    torch.manual_seed(0)
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32, device_map=args.device)
    tfm, _ = load_timesfm(max_context=LOOKBACK, max_horizon=128, device=args.device)
    assert_no_ar(tfm)

    res = {"protocol": dict(lookback=LOOKBACK, horizon=HORIZON, threshold=GATE_THRESHOLD,
                            splits={k: dict(split=v[0], seed=v[1], max_windows=200)
                                    for k, v in SPLITS.items()},
                            denominator="fitted lookback-96 ridge (gate_linear_baseline)",
                            device=args.device, torch=torch.__version__,
                            started=time.strftime("%Y-%m-%dT%H:%M:%S")),
           "cells": {}}
    for ds in args.datasets:
        train_w, ws = windows(ds)
        for name, (ctx, tgt) in ws.items():
            den = denominators(ds, train_w, ctx, tgt)
            torch.manual_seed(0)
            zs = {"chronos": chronos_zs_mse(pipe, ctx, tgt, args.device),
                  "timesfm": batched_point_mse(tfm, ctx, tgt, HORIZON, args.device)}
            for bb, z in zs.items():
                v = 1 - z / den["fitted"]
                admissible = den["fitted"] < den["constant"]
                res["cells"][f"{bb}_{ds.lower()}_{name}"] = dict(
                    backbone=bb, dataset=ds, split=name, n_windows=int(len(ctx)), zs_mse=z,
                    fitted_mse=den["fitted"], constant_mse=den["constant"], V_ridge=v,
                    fitted_admissible=bool(admissible),
                    passes=bool(admissible and v >= GATE_THRESHOLD))
                print(f"  {bb:8s} {ds:9s} {name:9s} n={len(ctx):3d}  ZS {z:.4f}  ridge "
                      f"{den['fitted']:.4f}  const {den['constant']:.4f}  V {v:+.3f}  "
                      f"{'PASS' if v >= GATE_THRESHOLD and admissible else 'fail'}", flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
