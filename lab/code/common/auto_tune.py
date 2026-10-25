"""
auto_tune.py — near-optimal detector thresholds by direct search (no grid).

Optimises the gate parameters against the ACTUAL objective:
    maximise   gtf_hit_rate  (fraction of GT frames with a matching detection)
    subject to fp_per_frame <= --max-fppf   (soft penalty when violated)

Search = coordinate descent over per-parameter value ladders with random
restarts. Each axis pass tries every ladder value with the other parameters
fixed and keeps the best; passes repeat until no axis improves. Evaluations
reuse frame_explorer's vectorised pipeline (and its components cache, so the
box/temporal-gate axes cost ~0.1 s per value; only the three pixel gates pay a
full recompute).

This finds (near-)optimal settings WITHIN the detector's axis-aligned gate
family — the same numbers you can type into frame_explorer. It will not beat
that family's ceiling; for that a learned component classifier is the next
step (see ddhf_svm_explorer.py's discussion).

Usage:
    python auto_tune.py --seqs 10 --track                 # tune on case 10
    python auto_tune.py --seqs 3,5,10 --max-fppf 0.1      # multi-case, no tracking
    python auto_tune.py --seqs 10 --track --restarts 3 --out best_10.json
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(os.path.dirname(_THIS_DIR))    # lab/ (from lab/code/common)
_FRED     = _REPO                                          # lab copy: the lab's data
_WP_DIR   = os.path.join(_FRED, "data", "working_points")   # working points, sweeps, evals
for _p in (_THIS_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import frame_explorer as FE

# Per-parameter candidate ladders. 'persist' moves (m, k) together.
LADDERS = {
    "min_events":       [2, 3, 4, 5, 6, 8, 10, 12, 16],
    "tau_sf":           [0.5, 0.55, 0.6, 0.66, 0.7, 0.75, 0.8, 0.9],
    "tau_comb":         [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
    "min_rotor_pixels": [1, 2, 3, 5, 8, 12, 20, 30, 60],
    "min_box_area":     [1, 5, 10, 20, 40, 80, 150, 300],
    "min_fill_ratio":   [0.0, 0.05, 0.1, 0.2, 0.35, 0.5],
    "min_snr_sum":      [0, 20, 50, 100, 200, 400],
    "min_snr_avg":      [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
    "alpha":            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "persist":          [(1, 1), (2, 2), (2, 3), (3, 4), (3, 5)],
    # tracking-only axes (no effect without --track; searched all the same)
    "track_coast":      [0, 1, 2, 3, 5, 8],
    "track_confirm":    [1, 2, 3],
    "track_acquire":    [1.0, 1.25, 1.5, 2.0, 3.0],
    "track_predict":    [False, True],
}
# Cheap axes first (they reuse the cached components); pixel gates last.
AXIS_ORDER = ["min_rotor_pixels", "min_box_area", "min_snr_avg", "min_snr_sum",
              "min_fill_ratio", "alpha", "persist",
              "track_coast", "track_confirm", "track_acquire", "track_predict",
              "tau_comb", "min_events", "tau_sf"]

DEFAULT_START = dict(min_events=4, tau_sf=0.7, tau_comb=4.0, min_rotor_pixels=5,
                     min_box_area=10, min_fill_ratio=0.0, min_snr_sum=0.0,
                     min_snr_avg=0.0, alpha=2.0, persist_m=1, persist_k=1,
                     track_coast=0, track_confirm=1, track_acquire=1.0,
                     track_predict=False)


def _set(params, axis, val):
    p = dict(params)
    if axis == "persist":
        p["persist_m"], p["persist_k"] = val
    else:
        p[axis] = val
    return p


def _get(params, axis):
    if axis == "persist":
        return (params["persist_m"], params["persist_k"])
    return params[axis]


def evaluate(params, Ds, mode, match, thr, track, sigma):
    """Aggregate counts over the tuning cases at these thresholds."""
    tp = fp = fn = hit = gtf = frames = 0
    for D in Ds.values():
        if track:
            pp = dict(params, tracking=True, track_sigma=sigma, match=match)
            _st, dets, _c = FE._track_pass(D, pp, mode, sigma)
        else:
            _r, dets, _c = FE.compute_all(D, mode, params)
        lo, hi = FE._frame_span(D)
        dets = FE._apply_persistence(dets, lo, hi,
                                     params["persist_m"], params["persist_k"])
        c = FE.count_matches(D, dets, match, thr)
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
        hit += c["gt_frames_hit"]; gtf += c["gt_frames"]
        frames += int(D["n_frames"])
    gtf_rate = hit / max(1, gtf)
    fppf = fp / max(1, frames)
    return dict(tp=tp, fp=fp, fn=fn, gtf_hit_rate=gtf_rate, fp_per_frame=fppf,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn))


def score_of(m, budget, penalty):
    """Objective: GT-frame hit rate, softly penalised beyond the FP budget."""
    return m["gtf_hit_rate"] - penalty * max(0.0, m["fp_per_frame"] - budget)


def descend(start, Ds, mode, match, thr, track, sigma, budget, penalty,
            max_rounds=4, log=print):
    """Coordinate descent from `start` until no axis improves."""
    params = dict(start)
    axes = [a for a in AXIS_ORDER if track or not a.startswith("track_")]
    best_m = evaluate(params, Ds, mode, match, thr, track, sigma)
    best_s = score_of(best_m, budget, penalty)
    n_eval = 1
    for rnd in range(max_rounds):
        improved = False
        for axis in axes:
            log(f"    [round {rnd + 1}] scanning {axis} …")
            cur = _get(params, axis)
            cand_best = (best_s, cur, best_m)
            # cases outer / ladder inner keeps the components cache hot
            totals = {v: None for v in LADDERS[axis] if v != cur}
            for v in totals:
                m = evaluate(_set(params, axis, v), Ds, mode, match, thr,
                             track, sigma)
                n_eval += 1
                s = score_of(m, budget, penalty)
                if s > cand_best[0] + 1e-9:
                    cand_best = (s, v, m)
            if cand_best[1] != cur:
                params = _set(params, axis, cand_best[1])
                best_s, best_m = cand_best[0], cand_best[2]
                improved = True
                log(f"    {axis} -> {cand_best[1]}   "
                    f"gtf={best_m['gtf_hit_rate']:.3f} "
                    f"fppf={best_m['fp_per_frame']:.3f} "
                    f"prec={best_m['precision']:.3f}")
        if not improved:
            break
    return params, best_m, best_s, n_eval


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("baseline", "pool", "pixelize", "components"),
                    default="baseline")
    ap.add_argument("--data-dir", default=os.path.join(_FRED, "frame_data"))
    ap.add_argument("--seqs", required=True,
                    help="Comma-separated cases to tune on (keep it a handful — "
                         "they are held in memory).")
    ap.add_argument("--match", choices=FE.MATCH_MODES, default="center")
    ap.add_argument("--match-thr", type=float, default=None)
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--track-sigma", type=float, default=20.0)
    ap.add_argument("--max-fppf", type=float, default=0.2,
                    help="FP/frame budget (soft, see --penalty). Default 0.2.")
    ap.add_argument("--penalty", type=float, default=3.0,
                    help="Score penalty per unit FP/frame over budget (default 3).")
    ap.add_argument("--restarts", type=int, default=2,
                    help="Random restarts after the default start (default 2).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None,
                    help="Seed the first descent from a point instead of the "
                         "default: either a best_point_*.json path or "
                         "'k=v,k=v,...' (e.g. from the sweep winner). The grid->"
                         "descent hybrid: coarse-sweep, then polish from its best.")
    ap.add_argument("--out", default=None,
                    help="Write the best point as JSON (default "
                         "data/working_points/best_point_<mode>[_track].json).")
    args = ap.parse_args()

    thr = args.match_thr if args.match_thr is not None else FE.MATCH_DEFAULT_THR[args.match]
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    print(f"loading {len(seqs)} case(s) …")
    Ds = {s: FE.load_D(s, args.mode, args.data_dir) for s in seqs}

    rng = random.Random(args.seed)
    start0 = dict(DEFAULT_START)
    if args.start:
        if os.path.exists(args.start):
            with open(args.start, encoding="utf-8") as fh:
                start0.update(json.load(fh).get("params", {}))
        else:
            for part in args.start.split(","):
                k, _, v = part.strip().partition("=")
                if k in start0:
                    start0[k] = type(start0[k])(float(v))
        print("seeded start:", {k: start0[k] for k in sorted(start0)})
    starts = [start0]
    for _ in range(max(0, args.restarts)):
        p = {}
        for axis in AXIS_ORDER:
            p = _set(p or dict(DEFAULT_START), axis, rng.choice(LADDERS[axis]))
        starts.append(p)

    t0 = time.perf_counter()
    best = None
    for i, st in enumerate(starts):
        print(f"[start {i + 1}/{len(starts)}]")
        params, m, s, n_eval = descend(st, Ds, args.mode, args.match, thr,
                                       args.track, args.track_sigma,
                                       args.max_fppf, args.penalty)
        print(f"  -> score={s:.4f}  gtf={m['gtf_hit_rate']:.3f}  "
              f"fppf={m['fp_per_frame']:.3f}  prec={m['precision']:.3f}  "
              f"({n_eval} evals)")
        if best is None or s > best[2]:
            best = (params, m, s)
    params, m, s = best
    dt = (time.perf_counter() - t0) / 60.0

    print(f"\nBEST after {dt:.1f} min  (cases: {', '.join(seqs)}"
          f"{', tracking s=%g' % args.track_sigma if args.track else ''})")
    print(f"  gtf_hit_rate={m['gtf_hit_rate']:.3f}  fp/frame={m['fp_per_frame']:.3f}"
          f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print("  params:", {k: params[k] for k in sorted(params)})
    flags = (f"--min-events {params['min_events']} --tau-sf {params['tau_sf']} "
             f"--tau-comb {params['tau_comb']} "
             f"--min-rotor-pixels {params['min_rotor_pixels']} "
             f"--min-box-area {params['min_box_area']} "
             f"--min-fill-ratio {params['min_fill_ratio']} "
             f"--min-snr-sum {params['min_snr_sum']} "
             f"--min-snr-avg {params['min_snr_avg']} --alpha {params['alpha']} "
             f"--persist-m {params['persist_m']} --persist-k {params['persist_k']}"
             + ((f" --track --track-sigma {args.track_sigma}"
                 f" --track-coast {params.get('track_coast', 0)}"
                 f" --track-confirm {params.get('track_confirm', 1)}"
                 f" --track-acquire {params.get('track_acquire', 1.0)}"
                 + (" --track-predict" if params.get('track_predict') else ""))
                if args.track else ""))
    print("  frame_explorer flags:\n    " + flags)

    out = args.out or os.path.join(
        _WP_DIR, f"best_point_{args.mode}{'_track' if args.track else ''}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(params=params, metrics=m, cases=seqs, mode=args.mode,
                       match=args.match, match_thr=thr, track=args.track,
                       track_sigma=args.track_sigma, max_fppf=args.max_fppf),
                  fh, indent=2)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
