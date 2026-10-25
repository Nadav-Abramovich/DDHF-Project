"""
build_frame_cache.py — build ONE case's frame_data feature cache.

The single-case, self-contained counterpart of common/utility/frame_features.py
(same output files, same pruning), working on the case layouts this folder
understands: HuggingFace cases (events.hdf5 + coordinates.txt) and the older
local ones (events.raw + interpolated_coordinates.txt). A missing case is
downloaded from HuggingFace first.

Writes into --out-dir (default fred/utility/frame_data/, shared with
cache_frames.bat and bayes_working_point.bat):
    frames_<seq>_<mode>.csv.gz   per-candidate-pixel features
    gt_<seq>.csv.gz              per-frame GT boxes
    meta_<seq>_<mode>.json       n_frames, window, H, W, pruning bounds

frame_explorer.py and debug_viewer_pooled.py --wp replay working points from
these files. show_case_graphs.py runs this script automatically when a case is
opened without its cache.

Progress goes to stdout, flushed: the precompute's own "[progress] windows i/N"
lines, plus "@@ <phase> <done> <total>" lines for the download / extract /
write phases, which show_case_graphs parses for its progress bar.

Usage:
    python build_frame_cache.py 46 --mode baseline
    python build_frame_cache.py 46 --mode pixelize --workers 4
"""
import os
import sys
import csv
import gzip
import json
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODES = ("baseline", "pool", "pixelize", "components")
DRONE = "all"            # comb search over all drone bands (as frame_features)
STRIDE = 1               # every frame
_CASE_ROOTS = ("", "cases", "..")          # script/<seq>, script/cases/<seq>, fred/<seq>
_HF_DIR = os.path.join(os.path.dirname(_THIS_DIR), "data")   # hf_index.json


def _say(phase, done, total):
    print(f"@@ {phase} {int(done)} {int(total)}", flush=True)


def _case_dir(seq):
    """The case folder, downloading it from HuggingFace when absent."""
    for root in _CASE_ROOTS:
        d = os.path.join(_THIS_DIR, root, str(seq))
        if os.path.isdir(d):
            return d
    import hf_cases
    return hf_cases.ensure_case(seq, os.path.join(_THIS_DIR, "cases"),
                                data_dir=_HF_DIR, progress=_say)


def _parse_gt(path):
    """`t_sec: x1, y1, x2, y2[, ...]` lines -> {t_us: [(x1,y1,x2,y2), ...]}
    (same parser as case_statistics._parse_interpolated_gt)."""
    gt = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            t_str, sep, rest = line.partition(":")
            if not sep:
                continue
            try:
                t_us = int(round(float(t_str) * 1e6))
                vals = [float(v) for v in rest.replace(",", " ").split()[:4]]
            except ValueError:
                continue
            if len(vals) >= 4:
                gt.setdefault(t_us, []).append(tuple(vals[:4]))
    return gt


def _write_pixels(D, out_csv, min_ev_floor, sf_max, snr_min):
    """Candidate-pixel rows that could pass a realistic threshold sweep
    (identical to frame_features._write_pixels)."""
    import pandas as pd
    gc = D["gc"]; sf = D["sf"]; snr = D["snr"]
    keep = (gc >= min_ev_floor) & (sf <= sf_max) & (snr >= snr_min)
    df = pd.DataFrame({
        "frame":     D["fr"][keep].astype(np.int64),
        "x":         D["x"][keep].astype(np.int32),
        "y":         D["y"][keep].astype(np.int32),
        "n_events":  gc[keep].astype(np.int32),
        "blob_size": D["size"][keep].astype(np.int32),
        "sf":        sf[keep].astype(np.float32),
        "snr":       snr[keep].astype(np.float32),
    })
    df.sort_values("frame", kind="stable", inplace=True)
    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.6g")
    os.replace(tmp, out_csv)
    return int(len(df))


def _write_gt(case_dir, window_us, out_gt):
    """Per-frame GT keyed by DDHF frame index; interpolated_coordinates.txt when
    the case has it (older layout), else HF's coordinates.txt (same format)."""
    gt_path = os.path.join(case_dir, "interpolated_coordinates.txt")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(case_dir, "coordinates.txt")
    gt = _parse_gt(gt_path)
    tmp = out_gt + ".tmp"
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x1", "y1", "x2", "y2"])
        for t_us in sorted(gt):
            fidx = int(round(t_us / window_us))
            for (x1, y1, x2, y2) in gt[t_us]:
                w.writerow([fidx, int(round(x1)), int(round(y1)),
                            int(round(x2)), int(round(y2))])
    os.replace(tmp, out_gt)


def _meta(D):
    def g(k, default=None):
        v = D.get(k, default)
        return v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
    window_ms = g("meta_window_ms", 33.333)
    window_us = int(round(float(window_ms) * 1000.0))
    return dict(n_frames=int(g("n_frames", 0)), n_gt=int(g("n_gt", 0)),
                window_ms=float(window_ms), window_us=window_us,
                H=int(g("H", 720)), W=int(g("W", 1280)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq", help="case id, e.g. 46")
    ap.add_argument("--mode", choices=MODES, default="baseline")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(_THIS_DIR), "utility", "frame_data"))
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="precompute worker processes (default min(8, CPU-1))")
    ap.add_argument("--gpu", action="store_true", help="CuPy GPU for the NDFT")
    ap.add_argument("--min-events-floor", type=int, default=3)
    ap.add_argument("--sf-max", type=float, default=0.9)
    ap.add_argument("--snr-min", type=float, default=1.5)
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the spectral .npz precompute in script/cache/")
    args = ap.parse_args()

    seq = str(args.seq)
    os.makedirs(args.out_dir, exist_ok=True)
    px_csv = os.path.join(args.out_dir, f"frames_{seq}_{args.mode}.csv.gz")
    gt_csv = os.path.join(args.out_dir, f"gt_{seq}.csv.gz")
    meta_json = os.path.join(args.out_dir, f"meta_{seq}_{args.mode}.json")
    if os.path.exists(px_csv):
        print(f"already built: {px_csv}")
        return 0

    case_dir = _case_dir(seq)
    if case_dir is None:
        print(f"case {seq}: not available locally or on HuggingFace")
        return 2

    import ddhf_stats_explorer as EX
    # Measure only pixels at/above the write floor: sub-floor pixels are dropped
    # when writing anyway, so the output is identical and ~6x faster.
    EX.PRECOMPUTE_MIN_EVENTS = max(1, args.min_events_floor)
    seq_rel = os.path.relpath(case_dir, _THIS_DIR)   # build_dataset resolves vs here
    D = EX.build_dataset(seq_rel, args.mode, DRONE, STRIDE, {}, None,
                         args.workers, args.gpu)
    _say("write", 0, 1)
    meta = _meta(D)
    n_rows = _write_pixels(D, px_csv, args.min_events_floor,
                           args.sf_max, args.snr_min)
    _write_gt(case_dir, meta["window_us"], gt_csv)
    with open(meta_json, "w", encoding="utf-8") as fh:
        json.dump({**meta, "mode": args.mode, "drone": DRONE,
                   "min_events_floor": args.min_events_floor,
                   "sf_max": args.sf_max, "snr_min": args.snr_min,
                   "pixel_rows": n_rows}, fh, indent=2)
    if not args.keep_cache:
        try:
            os.remove(EX.features_path(seq_rel, args.mode))
        except OSError:
            pass
    _say("write", 1, 1)
    print(f"done: {n_rows} pixel rows over {meta['n_frames']} frames -> {px_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
