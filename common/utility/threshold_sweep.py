"""
threshold_sweep.py — offline operating-point sweep over the frame_features CSVs.

Grids the detector thresholds over every case that has a
frames_<seq>_<mode>.csv.gz, scores each combo with the SAME per-instance
TP/FP/FN counting as frame_explorer (match=center by default: a detection
matches a GT when its center is within thr × the GT half-diagonal), tags each
case with its train/test split from fred_drone_map.csv, and appends one row per
(case, combo) to the output CSV — flushed after every case and RESUMABLE.

The recall-vs-FP/frame points across combos form a FROC curve: pick the combo at
the knee (max recall subject to your FP budget). --summarize aggregates the raw
rows per (combo, split) and prints the best operating points on the TRAIN split
so the TEST numbers stay honest.

No DDHF is recomputed — everything derives from the cached per-pixel features.

Usage:
    python threshold_sweep.py --mode baseline
    python threshold_sweep.py --mode baseline \\
        --min-events 4,8,16 --tau-comb 3,4,5,6,8 --min-rotor-pixels 1,3,5 \\
        --min-snr-sum 0,50,150 --persist 1:1,2:3,3:5
    python threshold_sweep.py --mode baseline --summarize --max-fppf 0.1
"""
import os
import sys
import csv
import time
import argparse
import itertools

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_THIS_DIR)
_FRED     = os.path.join(os.path.dirname(_REPO), "fred")   # default dataset + its data
_WP_DIR   = os.path.join(_FRED, "data", "working_points")   # working points, sweeps, evals
for _p in (_THIS_DIR, os.path.join(_REPO, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import frame_explorer as FE          # load_D + detection/counting helpers (no GUI)

RESULT_COLS = ["seq", "split", "combo", "min_events", "tau_sf", "tau_comb",
               "min_rotor_pixels", "min_box_area", "alpha", "min_fill_ratio",
               "min_snr_sum", "min_snr_avg", "persist_m", "persist_k",
               "frames", "gt_frames", "gt_frames_hit", "tp", "fp", "fn",
               "fp_frames", "tn_frames"]


def _floats(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def _ints(s):
    return [int(float(v)) for v in str(s).split(",") if v.strip()]


def _persists(s):
    """'1:1,2:3' -> [(1,1),(2,3)] as (m, k) pairs."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        m, _, k = part.partition(":")
        out.append((int(m), int(k)))
    return out


def _load_split_map():
    """{seq: split} from fred_drone_map.csv ('' when the file/column is absent)."""
    path = os.path.join(_FRED, "data", "fred_drone_map.csv")
    mp = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = (row.get("sequence") or "").strip()
                if seq:
                    mp[seq] = (row.get("split") or "").strip()
    return mp


def _combo_id(p):
    return (f"me{p['min_events']}_sf{p['tau_sf']}_cb{p['tau_comb']}"
            f"_mrp{p['min_rotor_pixels']}_mba{p['min_box_area']}_a{p['alpha']}"
            f"_fr{p['min_fill_ratio']}_ss{p['min_snr_sum']}_sa{p['min_snr_avg']}"
            f"_p{p['persist_m']}:{p['persist_k']}")


def _combos(args):
    grid = itertools.product(
        _ints(args.min_events), _floats(args.tau_sf), _floats(args.tau_comb),
        _ints(args.min_rotor_pixels), _ints(args.min_box_area), _floats(args.alpha),
        _floats(args.min_fill_ratio), _floats(args.min_snr_sum),
        _floats(args.min_snr_avg), _persists(args.persist))
    out = []
    for me, tsf, tcb, mrp, mba, al, fr, ss, sa, (pm, pk) in grid:
        out.append(dict(min_events=me, tau_sf=tsf, tau_comb=tcb,
                        min_rotor_pixels=mrp, min_box_area=mba, alpha=al,
                        min_fill_ratio=fr, min_snr_sum=ss, min_snr_avg=sa,
                        persist_m=pm, persist_k=pk))
    return out


def _sweep_case(D, combos, mode, match, thr, track=False, track_sigma=20.0):
    """All combos for one loaded case. Non-tracking uses the vectorised global-CC
    path (cached across combos that share the pixel gates); --track runs the
    sequential Gaussian-relaxed pass per combo (slower but still seconds)."""
    lo, hi = FE._frame_span(D)
    n_frames = int(D["n_frames"])
    rows = []
    for p in combos:
        if track:
            pp = dict(p, tracking=True, track_sigma=track_sigma, match=match)
            _stage, dets, _comps = FE._track_pass(D, pp, mode, track_sigma)
        else:
            _res, dets, _comps = FE.compute_all(D, mode, p)
        dets = FE._apply_persistence(dets, lo, hi, p["persist_m"], p["persist_k"])
        c = FE.count_matches(D, dets, match, thr)
        rows.append((p, dict(c, frames=n_frames)))
    return rows


def _append_rows(out_csv, seq, split, rows):
    """Append rows, projecting onto the file's EXISTING header when resuming an
    older sweep CSV — so schema additions never corrupt a file mid-run."""
    header = RESULT_COLS
    new = not os.path.exists(out_csv)
    if not new:
        with open(out_csv, newline="", encoding="utf-8") as fh:
            first = fh.readline().strip()
        if first:
            header = first.split(",")
    with open(out_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for p, c in rows:
            full = dict(p, seq=seq, split=split, combo=_combo_id(p), **c)
            w.writerow([full.get(h, "") for h in header])
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def summarize(out_csv, max_fppf, top=10, rank="gtf_hit_rate"):
    """Aggregate the raw rows per (combo, split); print the best combos on the
    train split (fallback: all rows) under the FP/frame budget, ranked by `rank`
    (default gtf_hit_rate = fraction of GT frames with a passing inside-GT box —
    'every frame should have a hit, with few strays')."""
    import pandas as pd
    df = pd.read_csv(out_csv)
    agg = (df.groupby(["combo", "split"], dropna=False)
             .agg(cases=("seq", "nunique"), frames=("frames", "sum"),
                  gt_frames=("gt_frames", "sum"), hit=("gt_frames_hit", "sum"),
                  tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
             .reset_index())
    agg["recall"] = agg.tp / (agg.tp + agg.fn).clip(lower=1)
    agg["precision"] = agg.tp / (agg.tp + agg.fp).clip(lower=1)
    agg["fp_per_frame"] = agg.fp / agg.frames.clip(lower=1)
    agg["gtf_hit_rate"] = agg.hit / agg.gt_frames.clip(lower=1)

    base = os.path.splitext(out_csv)[0] + "_summary.csv"
    agg.sort_values(["split", "recall"], ascending=[True, False]).to_csv(base, index=False)
    print(f"summary -> {base}")

    pick = agg[agg.split.astype(str) == "train"]
    if pick.empty:
        pick = agg
    ok = pick[pick.fp_per_frame <= max_fppf].sort_values(rank, ascending=False)
    print(f"\nTop combos by {rank} (split=train-ish, FP/frame <= {max_fppf}):")
    cols = ["combo", "split", "cases", "recall", "precision", "fp_per_frame", "gtf_hit_rate"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(ok.head(top)[cols].to_string(index=False) if not ok.empty
              else "  (none under that FP budget — raise --max-fppf)")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("baseline", "pool", "pixelize", "components"),
                    default="baseline")
    ap.add_argument("--data-dir", default=os.path.join(_FRED, "utility", "frame_data"))
    ap.add_argument("--out", default=None,
                    help="Output CSV (default utility/sweep_<mode>.csv).")
    ap.add_argument("--seqs", default=None, help="Comma-separated subset of cases.")
    ap.add_argument("--match", choices=FE.MATCH_MODES, default="center")
    ap.add_argument("--match-thr", type=float, default=None)
    # grid axes (comma-separated lists; single value = fixed)
    ap.add_argument("--min-events", default="4,8,16")
    ap.add_argument("--tau-sf", default="0.7")
    ap.add_argument("--tau-comb", default="3,4,5,6,8")
    ap.add_argument("--min-rotor-pixels", default="1,3,5,10")
    ap.add_argument("--min-box-area", default="10")
    ap.add_argument("--alpha", default="1.5")
    ap.add_argument("--min-fill-ratio", default="0")
    ap.add_argument("--min-snr-sum", default="0")
    ap.add_argument("--min-snr-avg", default="0")
    ap.add_argument("--persist", default="1:1",
                    help="m:k persistence combos, e.g. '1:1,2:3,3:5' (1:1 = off).")
    ap.add_argument("--track", action="store_true",
                    help="Sweep WITH tracking (Gaussian gate relaxation); slower "
                         "per combo but often a different frontier.")
    ap.add_argument("--track-sigma", type=float, default=20.0)
    ap.add_argument("--summarize", action="store_true",
                    help="Only aggregate an existing sweep CSV and print top combos.")
    ap.add_argument("--max-fppf", type=float, default=0.1,
                    help="FP/frame budget for the top-combos table (default 0.1).")
    ap.add_argument("--rank", choices=("gtf_hit_rate", "recall", "precision"),
                    default="gtf_hit_rate",
                    help="Ranking for the top-combos table (default gtf_hit_rate = "
                         "fraction of GT frames that got a passing inside-GT box).")
    args = ap.parse_args()

    out_csv = args.out or os.path.join(
        _WP_DIR, f"sweep_{args.mode}{'_track' if args.track else ''}.csv")
    if args.summarize:
        summarize(out_csv, args.max_fppf, rank=args.rank)
        return

    thr = args.match_thr if args.match_thr is not None else FE.MATCH_DEFAULT_THR[args.match]
    combos = _combos(args)
    split_map = _load_split_map()
    cases = ([s.strip() for s in args.seqs.split(",") if s.strip()] if args.seqs
             else FE._available_cases(args.mode, args.data_dir))

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["seq"], row["combo"]))

    print(f"sweep: mode={args.mode} match={args.match} thr={thr}  "
          f"{len(cases)} cases x {len(combos)} combos -> {out_csv}")
    t00 = time.perf_counter()
    for n, seq in enumerate(cases, 1):
        todo = [p for p in combos if (seq, _combo_id(p)) not in done]
        if not todo:
            continue
        t0 = time.perf_counter()
        try:
            D = FE.load_D(seq, args.mode, args.data_dir)
        except Exception as exc:
            print(f"  [{n}/{len(cases)}] {seq}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        rows = _sweep_case(D, todo, args.mode, args.match, thr,
                           track=args.track, track_sigma=args.track_sigma)
        _append_rows(out_csv, seq, split_map.get(seq, ""), rows)
        print(f"  [{n}/{len(cases)}] {seq}: {len(todo)} combos in "
              f"{time.perf_counter() - t0:.1f}s")
    print(f"done in {(time.perf_counter() - t00) / 60.0:.1f} min")
    summarize(out_csv, args.max_fppf, rank=args.rank)


if __name__ == "__main__":
    main()
