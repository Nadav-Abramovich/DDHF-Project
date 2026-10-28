"""
frame_features.py — Per-frame DDHF feature dump for offline IoU analysis.

For each case + detector MODE it writes the DDHF spectral features of every
candidate pixel of every frame to a CSV, plus the per-frame ground-truth boxes.
Together these let you re-derive the detector's boxes and compute IoU per frame
(or per GT frame) at ANY threshold — min_events / tau_sf / tau_comb /
min_rotor_pixels / min_box_area / alpha, and the IoU/score cutoff — WITHOUT
recomputing the (expensive) NDFT spectra. Change a threshold, re-read the CSV,
re-derive boxes, recompute IoU. No DDHF rerun.

What makes that possible: the spectral stage is cached at min_events=1 (every
pixel that has ≥1 event gets its spectral-flatness `sf` and comb-SNR `snr`
measured — see ddhf_stats_explorer.PRECOMPUTE_MIN_EVENTS). All gates
(min_events, tau_sf, tau_comb, blob size, box area, merge-alpha) are then just
filters over these cached per-pixel rows + an 8-connected components pass, which
is exactly how ddhf_stats_explorer / debug_viewer derive boxes.

Outputs (in --out-dir, default utility/frame_data/):
  frames_<seq>_<mode>.csv   one row per candidate pixel:
        frame,x,y,n_events,blob_size,sf,snr
     - frame     : DDHF frame index (window [frame*window_us, +window_us))
     - x,y       : event-image pixel (same coords as the GT boxes)
     - n_events  : events at that pixel in the window (the min_events slider)
     - blob_size : component size in px for the pixel's group (min_rotor_pixels
                   slider; pool/pixelize/components modes — else same as n_events)
     - sf        : spectral flatness  (keep pixel if sf <= tau_sf)
     - snr       : comb SNR           (keep pixel if snr >= tau_comb)
  gt_<seq>.csv              per-frame GT (from interpolated_coordinates.txt):
        frame,x1,y1,x2,y2
  meta_<seq>_<mode>.json    n_frames, window_us, H, W, mode, drone, floors.

Ground truth is interpolated_coordinates.txt (matching case_statistics.csv);
the comb search uses ALL drone bands.

Robustness / scale:
  * --mode selects the detector variant (default 'baseline' = per-pixel).
  * Runs on every FRED case (the HuggingFace case list, fred/data/hf_index.json,
    plus any case already on disk). A missing case is downloaded from HuggingFace
    (hf_cases) into fred/script/cases and, once its cache is written, deleted
    again to bound disk use (~1 GB per case); --keep-downloads keeps it. Cases
    that were already on disk are never deleted.
  * RESUMABLE: a (seq,mode) whose frames CSV already exists is skipped.
  * The spectral precompute is itself disk-cached per (seq,mode) by
    ddhf_stats_explorer, so re-runs are cheap.
  * Only "candidate" pixels are written — those that could pass a realistic gate
    sweep: n_events >= --min-events-floor (default 3), sf <= --sf-max (0.9), and
    snr >= --snr-min (1.5). This is lossless for any thresholds within those
    bounds and drops the ~99% pure-noise pixels; those bounds are the loosest
    min_events / tau_sf / tau_comb you can sweep to offline (widen them, at the
    cost of much larger CSVs, if you need more range).

Usage:
    python frame_features.py                          # all cases, mode=baseline
    python frame_features.py --mode pool --seqs 3,5
    python frame_features.py --min-events-floor 3     # smaller CSVs
    python frame_features.py --no-download            # only local cases
"""
import os
import sys
import csv
import gzip
import json
import time
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_THIS_DIR)
_FRED     = os.path.join(os.path.dirname(_REPO), "fred")   # default dataset + its data
_CODE_DIR = os.path.join(_REPO, "code")
for _p in (_CODE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import batch_ddhf_eval_pooled as B          # logging / safe_remove
import case_statistics as CS                # case discovery + interpolated GT parser
import hf_cases                             # HuggingFace case list + download

MODES = ("baseline", "pool", "pixelize", "components")   # baseline = per-pixel
DRONE = "all"                                            # comb search: all bands
STRIDE = 1                                               # every frame

PIXEL_COLUMNS = ["frame", "x", "y", "n_events", "blob_size", "sf", "snr"]


# ── case presence (local, else HuggingFace download) ────────────────────────────

def _ensure_case(name, dl, log_path):
    """(case_dir, downloaded): the local case folder, downloading it from
    HuggingFace when it is not on disk. downloaded=True marks a folder this run
    fetched (the caller deletes it after caching unless --keep-downloads)."""
    # check only this case's folder in each root (earlier roots win, as in
    # CS._local_cases) — scanning every case folder cost ~40 s per case
    for root in dl["roots"]:
        d = os.path.join(root, name)
        if CS._is_case_dir(d):
            return d, False
    if dl["no_download"]:
        raise FileNotFoundError(f"{name} not local and --no-download set.")
    case_dir = os.path.join(hf_cases.DEFAULT_CASES_DIR, name)
    if os.path.isdir(case_dir):
        B._safe_remove(case_dir)              # clear an incomplete partial
    B._log(log_path, f"  downloading case {name} from HuggingFace ...")
    got = hf_cases.ensure_case(name, hf_cases.DEFAULT_CASES_DIR)
    if got is None:
        raise FileNotFoundError(f"case {name}: HuggingFace download failed")
    return got, True


# ── writers ─────────────────────────────────────────────────────────────────────

def _write_pixels(D, out_csv, min_ev_floor, sf_max, snr_min):
    """Write the per-candidate-pixel feature rows, pruned to those that could pass
    a realistic threshold sweep: n_events >= min_ev_floor AND sf <= sf_max AND
    snr >= snr_min. This is LOSSLESS for any thresholds within those bounds (a
    pixel greener than the loosest gate is kept), and drops the ~99% pure-noise
    pixels (n_events=1, sf=1, snr=1) that can never pass.

    Uses pandas' C CSV writer: np.savetxt formats every row in Python, which took
    tens of minutes on the densest cases (33-40M rows) and made them look hung."""
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
    })                                                   # columns == PIXEL_COLUMNS
    df.sort_values("frame", kind="stable", inplace=True)  # frame-sorted, like build_dataset
    tmp = out_csv + ".tmp"
    # gzip level 1: same rows/values, ~2x faster to write than the level-9 default
    df.to_csv(tmp, index=False, compression={"method": "gzip", "compresslevel": 1},
              float_format="%.6g")
    os.replace(tmp, out_csv)
    return int(len(df))


def _write_gt(case_dir, window_us, out_gt):
    """Per-frame GT boxes from interpolated_coordinates.txt, keyed by DDHF frame
    index (t_sec*1e6 // window_us). Rewritten each run (cheap)."""
    gt_path = CS._gt_file(case_dir)                    # Drive or HF layout
    gt = CS._parse_interpolated_gt(gt_path)              # {t_us: [(x1,y1,x2,y2),...]}
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


def _first_gt_frame(gt_csv):
    """First frame index with a GT box in a written gt CSV (None if no GT)."""
    with gzip.open(gt_csv, "rt", newline="", encoding="utf-8") as fh:
        frames = [int(r["frame"]) for r in csv.DictReader(fh)]
    return min(frames) if frames else None


def _meta(D):
    """Pull the useful scalars out of a build_dataset result."""
    def g(k, default=None):
        v = D.get(k, default)
        return v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
    window_ms = g("meta_window_ms", 33.333)
    window_us = int(round(float(window_ms) * 1000.0))
    return dict(n_frames=int(g("n_frames", 0)), n_gt=int(g("n_gt", 0)),
                window_ms=float(window_ms), window_us=window_us,
                H=int(g("H", 720)), W=int(g("W", 1280)))


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Per-frame DDHF feature CSVs (per sequence, per mode) for "
                    "threshold-swept IoU without recomputing DDHF.")
    ap.add_argument("--mode", choices=MODES, default="baseline",
                    help="Detector variant: 'baseline' = per-pixel (default), "
                         "'pool' 3x3, 'pixelize' BxB, 'components' connected-comp.")
    ap.add_argument("--out-dir", default=os.path.join(_FRED, "utility", "frame_data"),
                    help="Output folder (default fred/utility/frame_data).")
    ap.add_argument("--seqs", default=None,
                    help="Comma-separated cases (default: all local + Drive).")
    ap.add_argument("--min-events-floor", type=int, default=3,
                    help="Keep only pixels with >= N events (default 3). This is the "
                         "lowest min_events you can sweep to offline; 1-2-event "
                         "pixels are pure noise. Lower it (e.g. 1) for more range at "
                         "the cost of much larger CSVs.")
    ap.add_argument("--sf-max", type=float, default=0.9,
                    help="Keep only pixels with spectral flatness <= this (default "
                         "0.9). This is the loosest tau_sf you can sweep to offline; "
                         "pixels above it (noise) are dropped.")
    ap.add_argument("--snr-min", type=float, default=1.5,
                    help="Keep only pixels with comb SNR >= this (default 1.5). This "
                         "is the loosest tau_comb you can sweep to offline; pixels "
                         "below it (noise) are dropped.")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="Precompute worker processes (default: min(8, CPU-1)). "
                         "Each loads the event stream, so raise cautiously (RAM).")
    ap.add_argument("--gpu", action="store_true", help="Use CuPy GPU for the NDFT.")
    ap.add_argument("--gpu-fused", action="store_true",
                    help="Spectral stage (NDFT, SF, comb SNR, subharmonic test) in one "
                         "fused CUDA kernel (common/code/gpu_fused.py), GPU only: fails "
                         "instead of falling back to the CPU. Same sf/snr as the CPU path "
                         "to ~1e-4 relative. Implies --gpu.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Keep each case's spectral .npz precompute (code/cache/). "
                         "Default deletes it after the CSV is written to save disk "
                         "(re-running a case then recomputes the NDFT).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames scanned per case (for testing).")
    ap.add_argument("--no-download", action="store_true",
                    help="Never download; only cases already on disk.")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="Keep cases this run downloaded (default: delete each "
                         "one once its cache is written, ~1 GB per case).")
    ap.add_argument("--cases-dir", action="append", default=None,
                    help="Dirs of extracted cases to use instead of the defaults "
                         "(fred/, fred/script/cases).")
    ap.add_argument("--skip-lead-in", action="store_true",
                    help="Compute each case only from its first GT frame on (the "
                         "noisy pre-drone lead-in is most of the work). Such a "
                         "cache only serves skip-lead-in scoring (bayes "
                         "--skip-start, the viewers' 'exclude lead-in'); its meta "
                         "JSON records lead_in_skipped + first_frame.")
    ap.add_argument("--subharmonic-reject", action="store_true",
                    help="Reject a pixel when a subharmonic of its comb fundamental, "
                         "outside the drone bands, has a stronger comb (DDHFConfig."
                         "subharmonic_reject). Changes snr: write to its own --out-dir.")
    ap.add_argument("--subharmonic-half-width", type=int, default=1,
                    help="Harmonic window half-width [bins] for that test (default 1).")
    ap.add_argument("--subharmonic-min-hz", type=float, default=60.0,
                    help="Lowest subharmonic tested [Hz] (default 60).")
    ap.add_argument("--subharmonic-ratio", type=float, default=1.0,
                    help="Reject when SNR(sub) > ratio x SNR(f0) (default 1 = stronger).")
    ap.add_argument("--subharmonic-mode", choices=("comb", "between"), default="comb",
                    help="How a subharmonic is scored: 'comb' = its comb SNR vs f0's "
                         "(default); 'between' = mean peak of its lines lying between f0's "
                         "harmonics vs the mean peak of f0's harmonics.")
    ap.add_argument("--subharmonic-min-k", type=int, default=2,
                    help="Smallest k of the subharmonics f0/k tested (default 2).")
    ap.add_argument("--comb-noise-stat", choices=("median", "mean"), default="median",
                    help="Comb SNR noise floor over the non-harmonic bins (DDHFConfig."
                         "comb_noise_stat). 'mean' changes snr: use its own --out-dir. "
                         "Default median = the usual cache.")
    ap.add_argument("--spectral-oversample", type=int, default=1,
                    help="Evaluate the NDFT n x denser (bins every 1000/(window_ms*n) Hz; "
                         "DDHFConfig.spectral_oversample). Changes sf/snr: use its own "
                         "--out-dir. Default 1 = the usual spectrum.")
    args = ap.parse_args()
    if args.gpu_fused:                    # inherited by the spawned precompute workers
        os.environ["DDHF_GPU_FUSED"] = "1"
        args.gpu = True

    import ddhf_stats_explorer as EX      # spectral precompute (shares its disk cache)
    # Precompute the NDFT only for pixels at/above our write floor. Sub-floor
    # pixels are dropped when writing anyway (their sf/snr just come out NaN), so
    # this is byte-identical output but ~6x faster — it skips the NDFT for the
    # ~99% of pixels that have only 1-2 events. This value is picked up by the
    # precompute workers via the pickled cfg overrides.
    EX.PRECOMPUTE_MIN_EVENTS = max(1, args.min_events_floor)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"frame_features_{args.mode}.log")

    # Case roots (local cases are used in place) + the HuggingFace case list.
    roots = list(args.cases_dir) if args.cases_dir else list(CS._DEFAULT_CASE_ROOTS)
    dl = dict(roots=roots, no_download=args.no_download)
    index = {} if args.no_download else hf_cases.case_index()

    if args.seqs:
        seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        seqs = sorted(set(index) | set(CS._local_cases(roots)), key=CS._seq_sort_key)

    # Optional spectral settings (off by default: the cache is then exactly the
    # usual one). They change the stored snr, so use a separate --out-dir.
    spectral_ov = {}
    if args.subharmonic_reject:
        spectral_ov = dict(subharmonic_reject=True,
                           subharmonic_half_width=args.subharmonic_half_width,
                           subharmonic_min_hz=args.subharmonic_min_hz,
                           subharmonic_ratio=args.subharmonic_ratio)
        if args.subharmonic_mode != "comb" or args.subharmonic_min_k != 2:
            spectral_ov.update(subharmonic_mode=args.subharmonic_mode,
                               subharmonic_min_k=args.subharmonic_min_k)
    if args.spectral_oversample > 1:
        spectral_ov["spectral_oversample"] = int(args.spectral_oversample)
    if args.comb_noise_stat != "median":
        spectral_ov["comb_noise_stat"] = args.comb_noise_stat
    variant = EX.feature_variant(EX.build_cfg(**spectral_ov))

    B._log(log_path, f"START frame features — mode={args.mode} bands={DRONE} "
                     f"{len(seqs)} cases -> {args.out_dir}"
                     + (f" [spectral variant {variant}]" if variant else ""))

    n_ok = n_err = n_skip = 0
    for name in seqs:
        px_csv = os.path.join(args.out_dir, f"frames_{name}_{args.mode}.csv.gz")
        gt_csv = os.path.join(args.out_dir, f"gt_{name}.csv.gz")
        meta_json = os.path.join(args.out_dir, f"meta_{name}_{args.mode}.json")
        if os.path.exists(px_csv):
            n_skip += 1
            continue
        t0 = time.perf_counter()
        B._log(log_path, f"-- Case {name} [{args.mode}] " + "-" * 24)
        downloaded, case_dir = False, None
        try:
            case_dir, downloaded = _ensure_case(name, dl, log_path)
            seq_rel = os.path.relpath(case_dir, _CODE_DIR)   # build_dataset resolves vs code/
            lead = {}
            if args.skip_lead_in:
                # first GT frame = first_gt_frame() of the written gt CSV, which
                # is exactly where skip-start scoring begins
                # same window as build_dataset (EX.build_cfg, 33.333 ms — not
                # DDHFConfig's 33.3 ms default, which can shift the frame by 1)
                _write_gt(case_dir, int(round(EX.build_cfg().window_ms * 1000.0)),
                          gt_csv)
                f0 = _first_gt_frame(gt_csv)
                EX.PRECOMPUTE_START = f0 or 1
                lead = dict(lead_in_skipped=True, first_frame=f0)
            D = EX.build_dataset(seq_rel, args.mode, DRONE, STRIDE, spectral_ov,
                                 args.max_frames, args.workers, args.gpu)
            meta = _meta(D)
            if lead.get("first_frame"):          # D counts computed windows only
                meta["n_frames"] += lead["first_frame"] - 1
            _write_gt(case_dir, meta["window_us"], gt_csv)
            if lead and _first_gt_frame(gt_csv) != lead["first_frame"]:
                raise RuntimeError("first GT frame moved after precompute")
            n_rows = _write_pixels(D, px_csv, args.min_events_floor,
                                   args.sf_max, args.snr_min)
            with open(meta_json, "w", encoding="utf-8") as fh:
                json.dump({**meta, "mode": args.mode, "drone": DRONE,
                           "min_events_floor": args.min_events_floor,
                           "sf_max": args.sf_max, "snr_min": args.snr_min,
                           "pixel_rows": n_rows, **lead, **spectral_ov}, fh, indent=2)
            if not args.keep_cache:
                B._safe_remove(EX.features_path(seq_rel, args.mode, variant))
            n_ok += 1
            dt = (time.perf_counter() - t0) / 60.0
            B._log(log_path, f"  DONE {name}: {n_rows} pixel-rows over "
                             f"{meta['n_frames']} frames ({dt:.1f} min) -> {px_csv}")
        except Exception as exc:
            n_err += 1
            B._log(log_path, f"  SKIPPED {name}: {type(exc).__name__}: {exc}")
        finally:
            if downloaded and case_dir and not args.keep_downloads:
                B._safe_remove(case_dir)              # bound disk: ~1 GB per case

    B._log(log_path, f"END: {n_ok} written, {n_skip} already-present, {n_err} errors.")


if __name__ == "__main__":
    main()
