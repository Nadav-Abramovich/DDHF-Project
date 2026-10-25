"""
ddhf_candidate_stats.py -- whole-sequence pass/fail breakdown of DDHF candidates.

For every window in a sequence this scans every DDHF *candidate* and records, in
pipeline order, the FIRST check it fails (or that it passed all of them and ended
up inside a detection box). The breakdown is split into candidates that lie
INSIDE a GT box and candidates OUTSIDE every GT box, so you can see where real
drones vs. background are being rejected.

A "candidate" is one pixel that produced >= 1 event in the window. Its fate is
the fate of the unit the detector actually tested it as part of:

  * baseline   -- the pixel itself.
  * pool       -- the pixel's 3x3 neighbourhood (matches debug_viewer_pooled).
  * pixelize   -- the BxB block (cfg.pixel_block) the pixel falls in.
  * components -- the 8-connected component the pixel belongs to.

Pipeline stages (first failing one is recorded):

  non-components (baseline / pool / pixelize):
      min_events  ->  SF (<=tau_sf)  ->  SNR (>=tau_comb)  ->  box
                  (box = green pixel survived min_rotor_pixels / min_box_area /
                   merge and ended up inside a final detection box)
  components:
      min_events  ->  min_rotor_pixels  ->  SF (<=tau_sf)  ->  SNR (>=tau_comb)
                  (a passing component IS the detection)

PASS = recognised as (part of) a drone detection. Everything else FAILed, and we
report at which check.

Speed: the scan is split across worker processes (--workers, default = CPU count),
each seeking to its own frame range in the .raw file; partial counts are summed.
The spectral stage also runs on the GPU with --gpu (baseline/pool/pixelize); when
--gpu is given the default drops to 1 worker (one process owns the GPU).

It reuses the exact detector code from debug_viewer_pooled, so the numbers match
the viewer / batch eval for the same flags.

Usage (same flags as debug_viewer_pooled; --frame is accepted but ignored -- the
whole sequence is always scanned):

    python ddhf_candidate_stats.py --seq <run folder> \\
        --mode baseline --tau-comb 10 --tau-sf 0.7 --min-events 4 --min-rotor-pixels 30
"""
import os
import argparse
import multiprocessing as mp
import numpy as np

# Importing the viewer also runs its Prophesee/Metavision DLL setup and pulls in
# the detector helpers, so we share ONE implementation of the pipeline.
import debug_viewer_pooled as dv
from ddhf_config import build_cfg, add_cfg_cli_args, cfg_overrides_from_args

# Stage codes (index into the per-region count vectors).
PASS, F_MINEV, F_SF, F_SNR, F_BOX, F_MINPX = 0, 1, 2, 3, 4, 5
N_STAGES = 6
STAGE_LABEL = {
    PASS:    "PASS  (in a detection)",
    F_MINEV: "fail min_events",
    F_SF:    "fail SF (>tau_sf)",
    F_SNR:   "fail SNR (<tau_comb)",
    F_BOX:   "fail box (min_rotor_pixels/min_box_area)",
    F_MINPX: "fail min_rotor_pixels",
}
# Which fail codes are meaningful for each mode, in pipeline order.
FAILS_PERPIXEL   = (F_MINEV, F_SF, F_SNR, F_BOX)
FAILS_COMPONENTS = (F_MINEV, F_MINPX, F_SF, F_SNR)

# Metric-value distributions. We track exact means (sum / n) plus a histogram so
# the median is correct over the whole sequence without storing every pixel.
# ev  = events per candidate (the value the min_events gate tested)
# sf  = spectral flatness  (only for candidates that reached the SF test)
# snr = comb-SNR           (same population as sf)
METRIC_NAMES = ("ev", "sf", "snr")
METRIC_EDGES = {
    "ev":  np.arange(0, 2001, dtype=np.float64),       # 0..2000 events, then overflow
    "sf":  np.linspace(0.0, 1.0, 1001),                # spectral flatness in (0, 1]
    "snr": np.linspace(0.0, 300.0, 3001),              # comb-SNR, then overflow
}
METRIC_LABEL = {"ev": "events per pixel", "sf": "spectral flatness", "snr": "comb-SNR"}
REGIONS = ("in", "out")


def _new_metric_acc():
    """Per-(region,metric) accumulator: running sum, count, histogram, overflow."""
    acc = {}
    for r in REGIONS:
        for m in METRIC_NAMES:
            edges = METRIC_EDGES[m]
            acc[f"{r}_{m}"] = dict(sum=0.0, n=0,
                                   hist=np.zeros(edges.size - 1, dtype=np.int64),
                                   over=0)
    return acc


def _add_metric(acc, region, metric, vals):
    """Fold a 1-D array of metric values into the accumulator."""
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return
    edges = METRIC_EDGES[metric]
    d = acc[f"{region}_{metric}"]
    d["sum"]  += float(vals.sum())
    d["n"]    += int(vals.size)
    h, _ = np.histogram(vals, bins=edges)
    d["hist"] += h
    d["over"] += int((vals > edges[-1]).sum())


def _metric_stats(d, edges):
    """(mean, median, n) from a metric accumulator. Median via the histogram CDF;
    if it lands in the overflow tail it is reported as the top edge."""
    n = d["n"]
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = d["sum"] / n
    cum = np.cumsum(d["hist"])
    idx = int(np.searchsorted(cum, n / 2.0, side="left"))
    if idx >= d["hist"].size:
        med = float(edges[-1])                          # in the overflow tail
    else:
        med = float(0.5 * (edges[idx] + edges[idx + 1]))
    return mean, med, n


def _rects_mask(rects, H, W):
    """Boolean (H,W) mask: True for any pixel inside any of `rects` (x1,y1,x2,y2),
    clipped to the sensor."""
    m = np.zeros((H, W), dtype=bool)
    for (x1, y1, x2, y2) in rects:
        xa = max(0, int(x1)); ya = max(0, int(y1))
        xb = min(W - 1, int(x2)); yb = min(H - 1, int(y2))
        if xb >= xa and yb >= ya:
            m[ya:yb + 1, xa:xb + 1] = True
    return m


def perpixel_stage_map(cmap, sfm, snrm, boxes, cfg, H, W):
    """Per-pixel first-failed-stage map for baseline / pool / pixelize.

    Candidate = pixel with >=1 event (cmap>0). sfm is NaN exactly when the pixel's
    group (pixel / 3x3 pool / BxB block) had < min_events, so NaN == min_events
    fail. Green pixels (passed SF & SNR) PASS iff they fall inside a final box."""
    stage = np.full((H, W), -1, dtype=np.int8)         # -1 = not a candidate
    cand   = cmap > 0
    sf_nan = np.isnan(sfm)

    fail_me = cand & sf_nan                              # < min_events
    pme     = cand & ~sf_nan                             # passed min_events
    ok_sf   = sfm <= cfg.tau_sf
    fail_sf = pme & ~ok_sf
    psf     = pme & ok_sf                                # passed SF
    ok_snr  = snrm >= cfg.tau_comb
    fail_snr = psf & ~ok_snr
    green    = psf & ok_snr                              # passed SF & SNR

    box_rects = [(b.x1, b.y1, b.x2, b.y2) for b in (boxes or [])]
    boxcov   = _rects_mask(box_rects, H, W)
    passed   = green & boxcov
    fail_box = green & ~boxcov

    stage[fail_me]  = F_MINEV
    stage[fail_sf]  = F_SF
    stage[fail_snr] = F_SNR
    stage[fail_box] = F_BOX
    stage[passed]   = PASS
    return stage


def component_stage_map(evs, H, W, cfg):
    """Per-pixel first-failed-stage map for --mode components. Mirrors
    debug_viewer_pooled.detect_components_mask, but records, per 8-connected
    component, the first gate it fails (min_events -> min_rotor_pixels -> SF ->
    SNR), then paints that onto every pixel of the component. Candidate = pixel in
    any component (== pixel with >=1 event).

    Returns (stage, ev_map, sf_map, snr_map): ev_map = the component event count
    the min_events gate tested (broadcast to its pixels); sf_map/snr_map = the
    component SF / comb-SNR where they were computed (NaN otherwise)."""
    stage   = np.full((H, W), -1, dtype=np.int8)
    ev_map  = np.zeros((H, W), dtype=np.float64)
    sf_map  = np.full((H, W), np.nan)
    snr_map = np.full((H, W), np.nan)
    if evs.size == 0:
        return stage, ev_map, sf_map, snr_map
    x = evs["x"].astype(np.int64); y = evs["y"].astype(np.int64)
    pol = dv._normalise_polarity(evs["p"])
    tn  = dv._normalise_timestamps(evs["t"].astype(np.float64),
                                   window_span=cfg.window_ms * 1000.0)
    count = np.bincount(y * W + x, minlength=H * W).reshape(H, W)
    labels, n = dv._connected_components(count > 0)
    if n == 0:
        return stage, ev_map, sf_map, snr_map
    ev_label   = labels[y, x]
    ev_counts  = np.bincount(ev_label, minlength=n + 1)
    comp_sizes = np.bincount(labels.ravel(), minlength=n + 1)
    karr = np.arange(cfg.frequency_bins, dtype=np.float64)

    comp_stage = np.full(n + 1, -1, dtype=np.int8)
    comp_sf    = np.full(n + 1, np.nan)
    comp_snr   = np.full(n + 1, np.nan)
    for comp in range(1, n + 1):
        if ev_counts[comp] < cfg.min_events:
            comp_stage[comp] = F_MINEV; continue
        if comp_sizes[comp] < cfg.min_rotor_pixels:
            comp_stage[comp] = F_MINPX; continue
        sel   = ev_label == comp
        F     = (pol[sel][None, :] * np.exp(1j * np.outer(karr, tn[sel]))).sum(axis=1)
        power = np.abs(F) ** 2
        pnd   = power[1:] + 1e-12
        sf    = float(np.exp(np.mean(np.log(pnd))) / np.mean(pnd))
        comp_sf[comp] = sf
        if sf > cfg.tau_sf:
            comp_stage[comp] = F_SF; continue
        _omega, snr = dv.best_comb(power, cfg)
        comp_snr[comp] = snr
        if snr < cfg.tau_comb:
            comp_stage[comp] = F_SNR; continue
        comp_stage[comp] = PASS

    m = labels > 0
    lab = labels[m]
    stage[m]   = comp_stage[lab]
    ev_map[m]  = ev_counts[lab]                          # group count tested vs min_events
    sf_map[m]  = comp_sf[lab]
    snr_map[m] = comp_snr[lab]
    return stage, ev_map, sf_map, snr_map


def _stage_map(evs, gts, cfg, H, W):
    """Dispatch to the right stage map for the active mode (dv globals are set by
    _apply_ctx first). Returns (stage, ev_map, sf_map, snr_map): per-pixel first-
    failed stage plus the three metric values the gates tested.

    For baseline/pool/pixelize ev_map is the pixel's OWN event count (== the
    min_events metric exactly in baseline; a lower bound under pool/pixelize, where
    the gate tested the pooled/block sum). sf_map/snr_map come straight from
    compute_pixel_stats (NaN where the group was below min_events)."""
    if dv._COMPONENTS:
        return component_stage_map(evs, H, W, cfg)
    cmap, sfm, snrm, _ = dv.compute_pixel_stats(evs, H, W, cfg)
    rmask = (~np.isnan(sfm) & (sfm <= cfg.tau_sf)
             & ~np.isnan(snrm) & (snrm >= cfg.tau_comb))
    boxes = dv._extract_boxes(rmask, np.nan_to_num(snrm), cfg)
    stage = perpixel_stage_map(cmap, sfm, snrm, boxes, cfg, H, W)
    return stage, cmap.astype(np.float64), sfm, snrm


# ── Worker-side context (rebuilt in each process; everything below is picklable) ──

def _apply_ctx(ctx):
    """Configure the shared detector globals + cfg from a picklable context dict.
    Returns (cfg, gt_rects_at)."""
    dv._USE_GPU = ctx["gpu"]
    dv._COMPONENTS, dv._PIXELIZE, dv._POOLING = dv.MODE_FLAGS[ctx["mode"]]
    cfg = build_cfg(**ctx["overrides"])
    cfg.rotor_bands_hz = dv.DRONE_BANDS[ctx["drone"]]

    gt_times = ctx["gt_times"]; gt_boxes = ctx["gt_boxes"]
    window_us = ctx["window_us"]; gt_tol_s = ctx["gt_tol_s"]

    def gt_rects_at(frame_idx):
        if gt_times is None:
            return []
        t_s  = frame_idx * window_us / 1e6
        j    = int(np.searchsorted(gt_times, t_s))
        cand = [k for k in (j - 1, j) if 0 <= k < len(gt_times)]
        if not cand:
            return []
        k = min(cand, key=lambda i: abs(gt_times[i] - t_s))
        if abs(gt_times[k] - t_s) > gt_tol_s:
            return []
        sel = np.flatnonzero(np.isclose(gt_times, gt_times[k]))
        return [tuple(int(round(v)) for v in gt_boxes[i]) for i in sel]

    return cfg, gt_rects_at


def _empty_partials():
    return dict(
        inside=np.zeros(N_STAGES, dtype=np.int64),
        outside=np.zeros(N_STAGES, dtype=np.int64),
        per_box_cand=[], per_box_pass=[], outside_per_frame=[],
        n_frames=0, n_frames_gt=0,
        metrics=_new_metric_acc(),
    )


def _process_frame(p, gts, evs, cfg, H, W):
    """Fold one window's events into the partial accumulator `p`."""
    stage, ev_map, sf_map, snr_map = _stage_map(evs, gts, cfg, H, W)

    cand     = stage >= 0
    ingt     = _rects_mask(gts, H, W)
    in_mask  = cand & ingt
    out_mask = cand & ~ingt

    p["inside"]  += np.bincount(stage[in_mask],  minlength=N_STAGES)
    p["outside"] += np.bincount(stage[out_mask], minlength=N_STAGES)
    p["outside_per_frame"].append(int(out_mask.sum()))
    # Metric-value distributions per region. ev over all candidates; sf/snr only
    # where they were actually computed (NaNs dropped by _add_metric).
    for region, rmask in (("in", in_mask), ("out", out_mask)):
        _add_metric(p["metrics"], region, "ev",  ev_map[rmask])
        _add_metric(p["metrics"], region, "sf",  sf_map[rmask])
        _add_metric(p["metrics"], region, "snr", snr_map[rmask])
    if gts:
        p["n_frames_gt"] += 1
        for rect in gts:
            bm = _rects_mask([rect], H, W)
            p["per_box_cand"].append(int((cand & bm).sum()))
            p["per_box_pass"].append(int(((stage == PASS) & bm).sum()))
    p["n_frames"] += 1


def scan_range(start, stop, ctx, stride=1, progress=False):
    """Serial scan of windows [start, stop) (1-based frame indices) in this
    process: read each window and fold it in. Used for the 1-worker path. With
    stride>1, every window is still decoded (decoding is cheap) but only every
    stride-th is processed."""
    cfg, gt_rects_at = _apply_ctx(ctx)
    H, W = ctx["H"], ctx["W"]; window_us = ctx["window_us"]
    p = _empty_partials()
    rd = dv.RawReader(ctx["raw_file"], max_events=ctx["max_events"])
    if start > 1:
        rd.seek_time(int((start - 1) * window_us))
    for n in range(start, stop):
        evs = rd.load_delta_t(window_us)                # advance the cursor (cheap)
        if (n - 1) % stride != 0:                        # skip un-sampled windows
            continue
        _process_frame(p, gt_rects_at(n), evs, cfg, H, W)
        if progress and n % 500 == 0:
            print(f"  window {n}/{stop - 1}")
    return p


def _consumer(task_q, result_q, ctx):
    """Compute worker: pull (gts, evs) windows off the bounded task queue, fold
    each into a LOCAL partial, and put that single partial on result_q when the
    queue is drained (a None sentinel). No file I/O happens here -- one reader in
    the parent feeds every worker, so there is no concurrent reading of the .raw."""
    cfg, _gt = _apply_ctx(ctx)
    H, W = ctx["H"], ctx["W"]
    p = _empty_partials()
    try:
        while True:
            item = task_q.get()
            if item is None:                            # sentinel: no more work
                break
            gts, evs = item
            _process_frame(p, gts, evs, cfg, H, W)
    except Exception as e:                              # never let the parent hang
        import traceback
        traceback.print_exc()
        p["error"] = repr(e)
    result_q.put(p)


def run_pipeline(n_scan, ctx, n_workers, gt_rects_at, stride=1):
    """Single sequential reader in this process -> bounded queue -> n_workers
    compute processes. The bounded queue gives backpressure (the reader blocks
    when workers fall behind), so events in flight stay bounded regardless of
    sequence length. With stride>1 every window is decoded but only every
    stride-th is enqueued for compute. Returns the list of per-worker partials."""
    task_q   = mp.Queue(maxsize=max(4, n_workers * 3))
    result_q = mp.Queue()
    procs = [mp.Process(target=_consumer, args=(task_q, result_q, ctx), daemon=True)
             for _ in range(n_workers)]
    for pr in procs:
        pr.start()

    rd = dv.RawReader(ctx["raw_file"], max_events=ctx["max_events"])
    window_us = ctx["window_us"]
    for n in range(1, n_scan + 1):
        evs = rd.load_delta_t(window_us)               # sequential read = fastest I/O
        if (n - 1) % stride != 0:                       # skip un-sampled windows
            continue
        task_q.put((gt_rects_at(n), evs))              # blocks if queue full
        if n % 500 == 0:
            print(f"  read window {n}/{n_scan}")
    for _ in procs:
        task_q.put(None)                               # one sentinel per worker

    partials = [result_q.get() for _ in procs]         # drain BEFORE join (avoids deadlock)
    for pr in procs:
        pr.join()
    for p in partials:
        if p.get("error"):
            print(f"  WARNING: a worker reported an error: {p['error']}")
    return partials


def _merge(partials):
    out = _empty_partials()
    for p in partials:
        out["inside"]  += p["inside"]
        out["outside"] += p["outside"]
        out["per_box_cand"].extend(p["per_box_cand"])
        out["per_box_pass"].extend(p["per_box_pass"])
        out["outside_per_frame"].extend(p["outside_per_frame"])
        out["n_frames"]    += p["n_frames"]
        out["n_frames_gt"] += p["n_frames_gt"]
        for key, d in p["metrics"].items():
            o = out["metrics"][key]
            o["sum"]  += d["sum"]
            o["n"]    += d["n"]
            o["hist"] += d["hist"]
            o["over"] += d["over"]
    return out


def _summ(vals):
    """(mean, median, n) for a list/array, guarding the empty case."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(np.median(a)), int(a.size)


def _report(p, fail_codes, mode_label, seq, cfg, stride=1):
    def region_block(title, c):
        tot = int(c.sum()); passed = int(c[PASS]); failed = tot - passed
        lines = [f"{title}: {tot} candidate pixels"]
        if tot == 0:
            lines.append("    (none)")
            return "\n".join(lines)
        lines.append(f"    {'PASS (recognised as drone)':<42} {passed:>10}  ({passed/tot*100:5.1f}%)")
        lines.append(f"    {'FAIL (not recognised)':<42} {failed:>10}  ({failed/tot*100:5.1f}%)")
        for code in fail_codes:
            v = int(c[code])
            lines.append(f"      - {STAGE_LABEL[code]:<38} {v:>10}  ({v/tot*100:5.1f}%)")
        return "\n".join(lines)

    inside = p["inside"]; outside = p["outside"]
    print()
    print("=" * 78)
    print(f"DDHF candidate pass/fail breakdown -- [{mode_label}]  seq={seq}")
    print("=" * 78)
    print(f"frames scanned: {p['n_frames']}   (with GT: {p['n_frames_gt']}, "
          f"without GT: {p['n_frames'] - p['n_frames_gt']})   "
          f"GT-box instances: {len(p['per_box_cand'])}")
    if stride > 1:
        print(f"SAMPLED: stride={stride} (every {stride}th window). Rates / means / "
              f"medians are representative; absolute counts are over the sampled frames "
              f"(x{stride} for a full-sequence estimate).")
    print("candidate = a pixel with >=1 event; first failing check is recorded.")
    print(f"  pipeline: {' -> '.join(STAGE_LABEL[c].replace('fail ', '') for c in fail_codes)} -> PASS")
    print()
    print(region_block("INSIDE GT", inside))
    print()
    print(region_block("OUTSIDE GT", outside))
    print()
    print(region_block("ALL CANDIDATES", inside + outside))

    bc_mean, bc_med, bc_n = _summ(p["per_box_cand"])
    bp_mean, bp_med, _    = _summ(p["per_box_pass"])
    of_mean, of_med, of_n = _summ(p["outside_per_frame"])
    print()
    print("-" * 78)
    print("pixel-amount summary")
    print("-" * 78)
    print(f"  candidate pixels per GT box      : mean={bc_mean:8.2f}  median={bc_med:8.1f}  "
          f"(over {bc_n} GT-box instances)")
    print(f"  PASS pixels per GT box           : mean={bp_mean:8.2f}  median={bp_med:8.1f}")
    print(f"  candidate pixels OUTSIDE GT/frame: mean={of_mean:8.2f}  median={of_med:8.1f}  "
          f"(over {of_n} frames)")
    print(f"  total candidate pixels  inside GT={int(inside.sum())}   "
          f"outside GT={int(outside.sum())}")
    if stride > 1:
        print(f"  full-sequence estimate (x{stride}): inside GT~={int(inside.sum())*stride}   "
              f"outside GT~={int(outside.sum())*stride}")

    # ── Metric-value distributions (the actual SF / SNR / event numbers) ────────
    mt = p["metrics"]
    print()
    print("-" * 78)
    print("metric values over candidate pixels  (mean / median, n = pixels measured)")
    print("ev = events/pixel tested vs min_events; sf/snr measured only where computed")
    print("-" * 78)
    print(f"  {'metric':<18}{'INSIDE GT':<37}{'OUTSIDE GT'}")
    for m in METRIC_NAMES:
        edges = METRIC_EDGES[m]
        i_mean, i_med, i_n = _metric_stats(mt[f"in_{m}"],  edges)
        o_mean, o_med, o_n = _metric_stats(mt[f"out_{m}"], edges)
        i_s = f"mean={i_mean:7.3f}  med={i_med:7.3f}  n={i_n:<9}"
        o_s = f"mean={o_mean:7.3f}  med={o_med:7.3f}  n={o_n:<9}"
        print(f"  {METRIC_LABEL[m]:<18}{i_s}{o_s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq",       type=str,   default="28")
    parser.add_argument("--frame",     type=int,   default=0,
                        help="Accepted for CLI compatibility with the viewer; IGNORED "
                             "(the whole sequence is always scanned).")
    parser.add_argument("--window-ms", type=float, default=33.333)
    add_cfg_cli_args(parser)
    parser.add_argument("--mode", choices=dv.MODES, default="pixelize",
                        help="Spatial aggregation (same meaning as debug_viewer_pooled).")
    parser.add_argument("--drone", choices=sorted(dv.DRONE_BANDS), default="all",
                        help="Restrict the comb search to one drone's bands (default 'all').")
    parser.add_argument("--gpu", action="store_true",
                        help="Run the spectral stage on the GPU (CuPy) when available. "
                             "Forces --workers 1 unless you set it explicitly.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for the scan (default: CPU count, or 1 with --gpu).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Debug: stop after this many windows.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth window (default 1 = all). N>1 subsamples "
                             "for an ~Nx speedup: pass/fail RATES and metric mean/median "
                             "stay representative; absolute pixel TOTALS are over the "
                             "sampled frames (the report also prints an x-N full-sequence "
                             "estimate).")
    args = parser.parse_args()
    stride = max(1, args.stride)

    overrides = cfg_overrides_from_args(args)
    overrides["window_ms"] = args.window_ms
    overrides["step_ms"]   = args.window_ms

    # Build cfg locally just for the banner + frame geometry (workers rebuild it).
    dv._COMPONENTS, dv._PIXELIZE, dv._POOLING = dv.MODE_FLAGS[args.mode]
    cfg = build_cfg(**overrides)
    cfg.rotor_bands_hz = dv.DRONE_BANDS[args.drone]

    DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.seq)
    import hf_cases as _hf
    RAW_FILE = _hf.event_file(DIR)          # events.raw or HF events.hdf5
    RGB_DIR  = _hf.rgb_dir(DIR)             # PADDED_RGB/ or HF RGB/
    GT_FILE  = os.path.join(DIR, "interpolated_coordinates.txt")
    if not os.path.exists(GT_FILE):          # HF case: coordinates.txt holds
        GT_FILE = os.path.join(DIR, "coordinates.txt")   # the same boxes

    window_us = int(round(args.window_ms * 1000.0))
    stream    = dv.StreamState(RAW_FILE, RGB_DIR)
    H, W      = stream.ev_height, stream.ev_width
    max_idx   = stream.max_frame_idx(window_us)
    max_events = stream.max_events
    del stream                                            # release the reader handle

    gt_times, gt_boxes = dv._load_gt_boxes(GT_FILE)
    gt_tol_s = cfg.window_ms / 1000.0

    n_scan = max_idx if args.max_frames is None else min(max_idx, args.max_frames)
    if args.workers is not None:
        n_workers = max(1, args.workers)
    else:
        n_workers = 1 if args.gpu else max(1, os.cpu_count() or 1)

    mode_label = ("connected-components" if dv._COMPONENTS else
                  (f"pixelize {cfg.pixel_block}x{cfg.pixel_block}" if dv._PIXELIZE
                   else ("3x3 pooled" if dv._POOLING else "baseline (per-pixel)")))
    fail_codes = FAILS_COMPONENTS if dv._COMPONENTS else FAILS_PERPIXEL

    print(f"[{mode_label}]  seq={args.seq}  window={args.window_ms}ms  "
          f"min_events>={cfg.min_events}  SF<={cfg.tau_sf}  SNR>={cfg.tau_comb}  "
          f"min_rotor_pixels={cfg.min_rotor_pixels}  min_box_area={cfg.min_box_area}  "
          f"alpha={cfg.alpha}")
    if gt_times is None:
        print(f"  WARNING: no GT boxes found at {GT_FILE} -- everything counts as OUTSIDE GT.")
    n_sampled = len(range(1, n_scan + 1, stride))
    stride_s  = f"  (stride {stride}: {n_sampled} sampled)" if stride > 1 else ""
    print(f"  scanning {n_scan} windows (frames 1..{n_scan}){stride_s} with {n_workers} "
          f"worker(s){' [GPU]' if args.gpu else ''} ...")

    ctx = dict(raw_file=RAW_FILE, window_us=window_us, H=H, W=W, max_events=max_events,
               mode=args.mode, drone=args.drone, gpu=args.gpu, overrides=overrides,
               gt_times=gt_times, gt_boxes=gt_boxes, gt_tol_s=gt_tol_s)

    if n_workers <= 1:
        # Serial: one process reads and computes. Simplest, and the right choice
        # for --gpu (one process owns the GPU).
        partials = [scan_range(1, n_scan + 1, ctx, stride=stride, progress=True)]
    else:
        # One sequential reader in this process feeds n_workers compute processes
        # through a bounded queue -- the .raw is read exactly once.
        _cfg_unused, gt_rects_at = _apply_ctx(ctx)
        partials = run_pipeline(n_scan, ctx, n_workers, gt_rects_at, stride=stride)
    merged = _merge(partials)

    _report(merged, fail_codes, mode_label, args.seq, cfg, stride=stride)


if __name__ == "__main__":
    main()
