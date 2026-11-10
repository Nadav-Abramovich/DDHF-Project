"""
frame_explorer.py — ddhf_stats_explorer, but driven by the frame_features CSVs
(utility/frame_data/) instead of recomputing the DDHF spectra.

It rebuilds the exact per-candidate-pixel table (`D`) that ddhf_stats_explorer
works on — frame, x, y, n_events, blob_size, sf, snr — from
frames_<seq>_<mode>.csv.gz + gt_<seq>.csv.gz, then REUSES that module's live
threshold logic and GUI. So you get the same interactive explorer (min_events /
tau_sf / tau_comb / min_rotor_pixels / min_box_area sliders, per-frame candidate-
pixel overlay coloured by gate stage, INSIDE/OUTSIDE-GT bar charts) with ZERO
DDHF recompute — change a threshold, everything updates from the CSV.

On top of the explorer it adds what the CSVs were made for: per-frame detection
boxes (8-connected components over the PASS pixels + area filter + alpha merge,
identical to utils._extract_boxes) and the IoU of those boxes against the
interpolated GT — per frame and aggregated (TP/FP/FN, mean IoU over all frames
and over GT-only frames) — again with no DDHF rerun.

Usage:
    python frame_explorer.py --seq 3                    # GUI viewer (mode baseline)
    python frame_explorer.py --seq 3 --mode pool
    python frame_explorer.py --seq 3 --report          # headless IoU/TP-FP-FN
    python frame_explorer.py --seq 3 --report \\
        --min-events 16 --tau-sf 0.5 --tau-comb 6 --min-rotor-pixels 60 --iou-thr 0.1
    python frame_explorer.py --seq 3 --report --per-frame-out iou_3.csv
"""
import os
import sys
import csv
import gzip
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(os.path.dirname(_THIS_DIR))    # lab/ (from lab/code/common)
_FRED     = _REPO                                          # lab copy: the lab's data
_CODE_DIR = _THIS_DIR                                      # lab/code is flat
for _p in (_CODE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ddhf_stats_explorer as EX             # reuse compute_stats / print_report / GUI
from utils import BoundingBox, _merge_boxes  # same box merge the detector uses
from scipy import ndimage

# numba-JIT kernels for the per-frame hot path (identical results, ~3x compute;
# see fe_fast.py). Optional: FE_FAST=0 or a missing numba falls back to scipy.
try:
    _FF = None
    if os.environ.get("FE_FAST", "1") != "0":
        import fe_fast as _FF
except Exception:
    _FF = None


def _label_px(sx, sy):
    """0-based 8-connected component id per pass pixel (ndimage numbering)."""
    if _FF is not None:
        return _FF.cc_label_px(sx.astype(np.int64), sy.astype(np.int64))
    x0, y0 = int(sx.min()), int(sy.min())
    grid = np.zeros((int(sy.max()) - y0 + 1, int(sx.max()) - x0 + 1), bool)
    grid[sy - y0, sx - x0] = True
    lbl, _n = ndimage.label(grid, structure=np.ones((3, 3)))
    return lbl[sy - y0, sx - x0] - 1

DEFAULT_DATA = os.path.join(_FRED, "frame_data")

# Small LRU cache of loaded D dicts so Prev/Next/mode-toggle back to a case you've
# already viewed is instant (each D is read-only after load, so reuse is safe).
from collections import OrderedDict
_D_CACHE = OrderedDict()
_D_CACHE_MAX = 4


# ── load the CSVs into a ddhf_stats_explorer `D` dict ───────────────────────────

def _read_gz_csv(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    return header, data


def load_D(seq, mode, data_dir, wscale=1):
    """Rebuild the explorer's per-pixel table D from frames_<seq>_<mode>.csv.gz +
    gt_<seq>.csv.gz. Reconstructs the derived columns (ig / gb / box_keys /
    box_cand / out_per_frame / gt_rects) the explorer expects. LRU-cached.

    wscale > 1 emulates a wscale× DDHF window by aggregating that many
    consecutive 33 ms windows (see _rescale_windows) — the only window change
    the cached per-window features permit."""
    wscale = max(1, int(wscale))
    key = (os.path.abspath(data_dir), str(seq), str(mode), wscale)
    hit = _D_CACHE.get(key)
    if hit is not None:
        _D_CACHE.move_to_end(key)
        return hit
    if wscale > 1:
        D = _rescale_windows(load_D(seq, mode, data_dir, 1), wscale)
        D = EX._finalize(D)
        D["_token"] = key
        _D_CACHE[key] = D
        if len(_D_CACHE) > _D_CACHE_MAX:
            _D_CACHE.popitem(last=False)
        return D
    px_path = os.path.join(data_dir, f"frames_{seq}_{mode}.csv.gz")
    npz_path = os.path.join(data_dir, f"frames_{seq}_{mode}.npz")
    gt_path = os.path.join(data_dir, f"gt_{seq}.csv.gz")
    if not os.path.exists(px_path) and not os.path.exists(npz_path):
        raise FileNotFoundError(f"no {px_path} (run frame_features.py --seq {seq} "
                                f"--mode {mode} first)")

    if os.path.exists(npz_path):
        # binary mirror (see convert_cache_npz.py): ~8x faster than CSV parsing
        z = np.load(npz_path)
        fr = z["frame"].astype(np.int64, copy=False)
        x = z["x"]; y = z["y"]; gc = z["n_events"]; size = z["blob_size"]
        sf = z["sf"]; snr = z["snr"]
    else:
        # pandas' C parser is ~6-10x faster than np.loadtxt on these million-row
        # CSVs (which is what made case/mode switching sluggish).
        import pandas as pd
        _dt = {"frame": np.int64, "x": np.int32, "y": np.int32,
               "n_events": np.int32, "blob_size": np.int32,
               "sf": np.float32, "snr": np.float32}
        try:
            df = pd.read_csv(px_path, dtype=_dt)
        except pd.errors.EmptyDataError:
            df = None
        if df is None or df.empty:
            fr = np.zeros(0, np.int64); x = y = np.zeros(0, np.int32)
            gc = size = np.zeros(0, np.int32); sf = snr = np.zeros(0, np.float32)
        else:
            fr = df["frame"].to_numpy(); x = df["x"].to_numpy(); y = df["y"].to_numpy()
            gc = df["n_events"].to_numpy(); size = df["blob_size"].to_numpy()
            sf = df["sf"].to_numpy(); snr = df["snr"].to_numpy()
    order = np.argsort(fr, kind="stable")           # frame-sorted, like build_dataset
    fr, x, y, gc, size, sf, snr = (v[order] for v in (fr, x, y, gc, size, sf, snr))

    # GT rects as (frame, x1, y1, x2, y2); each row is one global GT box.
    gt_rows = []
    if os.path.exists(gt_path):
        with gzip.open(gt_path, "rt", newline="", encoding="utf-8") as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                if len(row) >= 5:
                    gt_rows.append([int(float(v)) for v in row[:5]])
    gt_rects = np.array(gt_rows, np.int32) if gt_rows else np.zeros((0, 5), np.int32)

    H, W = _dims(seq, mode, data_dir, x, y, gt_rects)

    n_frames = _meta_n_frames(seq, mode, data_dir, _frame_index(fr)[0].size)
    D = _assemble_D(fr, x, y, gc, size, sf, snr, gt_rects, H, W, n_frames)
    D = EX._finalize(D)                              # adds uniq_frames/frame_starts/counts
    D["_token"] = key                                # identity for _CC_CACHE keys
    _D_CACHE[key] = D
    if len(_D_CACHE) > _D_CACHE_MAX:
        _D_CACHE.popitem(last=False)
    return D


def _assemble_D(fr, x, y, gc, size, sf, snr, gt_rects, H, W, n_frames):
    """Derived per-pixel GT columns (ig/gb) + summary columns, shared by the
    normal load path and the window-rescaled one."""
    gb = np.full(fr.size, -1, np.int64)
    if gt_rects.size:
        gt_by_frame = {}
        for gid, (gf, x1, y1, x2, y2) in enumerate(gt_rects):
            gt_by_frame.setdefault(int(gf), []).append((gid, x1, y1, x2, y2))
        uf, st, ct = _frame_index(fr)
        for f, s, c in zip(uf, st, ct):
            boxes = gt_by_frame.get(int(f))
            if not boxes:
                continue
            sx = x[s:s + c]; sy = y[s:s + c]
            for gid, x1, y1, x2, y2 in boxes:
                inside = (sx >= x1) & (sx <= x2) & (sy >= y1) & (sy <= y2)
                gb[s:s + c][inside] = gid
    ig = (gb >= 0).astype(np.uint8)

    n_boxes = gt_rects.shape[0]
    box_keys = np.arange(n_boxes, dtype=np.int32)
    box_cand = np.bincount(gb[gb >= 0], minlength=n_boxes).astype(np.int32) \
        if n_boxes else np.zeros(0, np.int32)
    uf, st, ct = _frame_index(fr)
    out_per_frame = np.array([int((gb[s:s + c] < 0).sum()) for s, c in zip(st, ct)], np.int32)
    return dict(fr=fr, x=x, y=y, gc=gc, size=size, sf=sf, snr=snr, ig=ig, gb=gb,
                box_keys=box_keys, box_cand=box_cand, out_per_frame=out_per_frame,
                gt_rects=gt_rects, n_frames=int(n_frames),
                n_gt=int(np.unique(gt_rects[:, 0]).size if gt_rects.size else 0),
                H=int(H), W=int(W))


def _rescale_windows(base, k):
    """Emulate a k× DDHF window from the cached per-33 ms-window features.

    A true k-window NDFT can't be rebuilt from per-window aggregates, so this
    merges k consecutive windows per pixel the only faithful way the data
    allows: n_events are SUMMED (exact) and the BEST sub-window spectral
    response is kept (min sf, max snr, max blob size) — a pixel passes the
    gates in the merged window iff some sub-window would pass it. GT boxes for
    a merged frame come from its middle sub-frame (keeping all k would inflate
    the GT-instance counts k×)."""
    fr2 = base["fr"] // k
    H2 = int(base["H"]) + 1; W2 = int(base["W"]) + 1
    pid = (fr2.astype(np.int64) * H2 + base["y"].astype(np.int64)) * W2 \
        + base["x"].astype(np.int64)
    o = np.argsort(pid, kind="stable")
    ps = pid[o]
    st = np.r_[0, np.flatnonzero(np.diff(ps)) + 1] if ps.size else np.zeros(0, np.int64)
    if ps.size:
        take = o[st]
        fr = fr2[o][st]
        x = base["x"][o][st]; y = base["y"][o][st]
        gc = np.add.reduceat(base["gc"][o].astype(np.int64), st).astype(np.int32)
        size = np.maximum.reduceat(base["size"][o], st)
        sf = np.minimum.reduceat(base["sf"][o], st)
        snr = np.maximum.reduceat(base["snr"][o], st)
    else:
        fr = np.zeros(0, np.int64); x = y = np.zeros(0, np.int32)
        gc = size = np.zeros(0, np.int32); sf = snr = np.zeros(0, np.float32)

    gr = base["gt_rects"]
    if gr.size:
        gf = gr[:, 0].astype(np.int64); gf2 = gf // k
        rows = []
        for m in np.unique(gf2):
            sub = gr[gf2 == m]
            subf = sub[:, 0].astype(np.int64)
            mid = subf[np.argmin(np.abs(subf - (m * k + k // 2)))]
            for row in sub[subf == mid]:
                rows.append([int(m), int(row[1]), int(row[2]), int(row[3]), int(row[4])])
        gt_rects = np.array(rows, np.int32)
    else:
        gt_rects = np.zeros((0, 5), np.int32)

    n_frames = (int(base["n_frames"]) + k - 1) // k
    return _assemble_D(fr, x, y, gc, size, sf, snr, gt_rects,
                       base["H"], base["W"], n_frames)


def _frame_index(fr):
    if fr.size == 0:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)
    uf, st, ct = np.unique(fr, return_index=True, return_counts=True)
    return uf, st, ct


def _meta_n_frames(seq, mode, data_dir, fallback):
    import json
    p = os.path.join(data_dir, f"meta_{seq}_{mode}.json")
    if os.path.exists(p):
        try:
            return int(json.load(open(p)).get("n_frames", fallback))
        except Exception:
            pass
    return fallback


def _dims(seq, mode, data_dir, x, y, gt_rects):
    import json
    p = os.path.join(data_dir, f"meta_{seq}_{mode}.json")
    if os.path.exists(p):
        try:
            m = json.load(open(p)); return int(m.get("H", 720)), int(m.get("W", 1280))
        except Exception:
            pass
    H = int(max(y.max() + 1 if y.size else 0, gt_rects[:, 4].max() + 1 if gt_rects.size else 0, 720))
    W = int(max(x.max() + 1 if x.size else 0, gt_rects[:, 3].max() + 1 if gt_rects.size else 0, 1280))
    return H, W


# ── per-frame detection boxes + IoU (no DDHF rerun) ─────────────────────────────

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter)


def _boxes_from_pass(px, py, min_box_area, alpha, min_fill_ratio=0.0,
                     snr=None, min_snr_sum=0.0, min_rotor_pixels=1,
                     min_snr_avg=0.0):
    """8-connected components over PASS pixels -> gated, alpha-merged boxes
    (same rule as utils._extract_boxes). Component gates, all off by default:
      min_rotor_pixels : component pixel count
      min_box_area     : component bbox area
      min_fill_ratio   : pixel count / bbox area (density; ratio analogue of
                         min_rotor_pixels)
      min_snr_sum      : sum of the component's pixel comb-SNRs — a 3-px blob at
                         SNR 25 each is far more drone-like than 3 px at 6, which
                         pixel-count gates can't express."""
    if px.size == 0:
        return []
    pix_lab = _label_px(px, py) + 1                  # component id per input pixel
    n = int(pix_lab.max())
    boxes = []
    for lid in range(1, n + 1):
        m = pix_lab == lid
        npx = int(m.sum())
        if npx < min_rotor_pixels:
            continue
        b = BoundingBox(int(px[m].min()), int(py[m].min()),
                        int(px[m].max()), int(py[m].max()), 1.0)
        if b.area < min_box_area:
            continue
        if min_fill_ratio > 0.0 and npx < min_fill_ratio * max(1, b.area):
            continue
        if snr is not None and min_snr_sum > 0.0 and float(snr[m].sum()) < min_snr_sum:
            continue
        if snr is not None and min_snr_avg > 0.0 \
                and float(snr[m].sum()) < min_snr_avg * npx:
            continue
        boxes.append(b)
    return _merge_boxes(boxes, alpha)


def _frame_span(D):
    """(lowest, highest) frame index across candidates and GT."""
    los, his = [], []
    if D["uniq_frames"].size:
        los.append(int(D["uniq_frames"][0])); his.append(int(D["uniq_frames"][-1]))
    if D["gt_rects"].size:
        los.append(int(D["gt_rects"][:, 0].min())); his.append(int(D["gt_rects"][:, 0].max()))
    return (min(los), max(his)) if los else (0, 0)


def _dets_from_mask(D, mask, p, comp_gate=True):
    """{frame: [det rects]} from a per-row boolean mask of gate-passing pixels:
    per-frame 8-connected components + the component gates + alpha merge.

    Fast path: one np.add.reduceat gives every frame's green-pixel count, and
    frames that cannot reach min_rotor_pixels are skipped wholesale — this is
    what keeps slider updates and grid sweeps quick at tight thresholds.
    comp_gate=False (components mode) skips the per-component pixel-count gate,
    which that mode already applied on the pooled blob size."""
    dets = {}
    starts = D["frame_starts"]
    if starts.size == 0 or not mask.any():
        return dets
    gcnt = np.add.reduceat(mask.astype(np.int32), starts)
    need = max(1, int(p.get("min_rotor_pixels", 1))) if comp_gate else 1
    X = D["x"]; Y = D["y"]; S = D["snr"]
    uf = D["uniq_frames"]; cnts = D["frame_counts"]
    for i in np.flatnonzero(gcnt >= need):
        s = int(starts[i]); c = int(cnts[i])
        m = mask[s:s + c]
        bs = _boxes_from_pass(X[s:s + c][m], Y[s:s + c][m], p["min_box_area"],
                              p.get("alpha", 1.5), p.get("min_fill_ratio", 0.0),
                              snr=S[s:s + c][m], min_snr_sum=p.get("min_snr_sum", 0.0),
                              min_rotor_pixels=need,
                              min_snr_avg=p.get("min_snr_avg", 0.0))
        if bs:
            dets[int(uf[i])] = [(b.x1, b.y1, b.x2, b.y2) for b in bs]
    return dets


def _apply_persistence(dets, f_lo, f_hi, m, k):
    """m-of-k temporal persistence on raw per-frame detections: report a detection
    only if in at least m of the last k frames (current one included) SOME raw
    detection center lay within max(12 px, 1.5×max(w,h)) of it. History uses the
    RAW detections, so one filtered frame doesn't cascade. m<=1 or k<=1 = off —
    a drone's rotors recur in place across consecutive frames; noise doesn't."""
    m = int(m); k = int(k)
    if k <= 1 or m <= 1:
        return dets
    from collections import deque
    hist = deque(maxlen=k - 1)
    out = {}
    for f in range(f_lo, f_hi + 1):
        cur = dets.get(f, [])
        kept = []
        for r in cur:
            cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
            rad = max(12.0, 1.5 * max(r[2] - r[0], r[3] - r[1]))
            support = 1
            for centers in hist:
                if any((cx - a) ** 2 + (cy - b) ** 2 <= rad * rad for (a, b) in centers):
                    support += 1
            if support >= m:
                kept.append(r)
        hist.append([((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in cur])
        if kept:
            out[f] = kept
    return out


def _score_caches(D, dets, match):
    """Best match score per GT instance / per detection / per GT frame — arrays a
    threshold can be applied to instantly (the match_thr slider re-splits live).
    Vectorised: GT×detection pairs are built with searchsorted/repeat and scored
    in one shot instead of a per-frame Python loop."""
    gr = D["gt_rects"]
    n_gt = int(gr.shape[0])
    flat_f, flat_r = [], []
    for f, rs in dets.items():
        for r in rs:
            flat_f.append(f); flat_r.append(r)
    df = np.asarray(flat_f, np.int64)
    dr = np.asarray(flat_r, float).reshape(-1, 4)
    n_det = df.size
    worst = _worst(match)
    gt_best = np.full(n_gt, worst, float)
    det_best = np.full(n_det, worst, float)

    if n_gt and n_det:
        gf = gr[:, 0].astype(np.int64)
        od = np.argsort(df, kind="stable")
        dfs = df[od]; drs = dr[od]
        lo = np.searchsorted(dfs, gf, "left"); hi = np.searchsorted(dfs, gf, "right")
        cnt = hi - lo
        gi = np.repeat(np.arange(n_gt), cnt)                  # pair -> GT index
        if gi.size:
            dpos = (np.arange(int(cnt.sum()))
                    - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    + np.repeat(lo, cnt))                     # pair -> det (sorted)
            G4 = gr[gi, 1:5].astype(float); D4 = drs[dpos]
            if match == "center":
                gcx = (G4[:, 0] + G4[:, 2]) / 2; gcy = (G4[:, 1] + G4[:, 3]) / 2
                dcx = (D4[:, 0] + D4[:, 2]) / 2; dcy = (D4[:, 1] + D4[:, 3]) / 2
                hd = 0.5 * np.hypot(G4[:, 2] - G4[:, 0], G4[:, 3] - G4[:, 1])
                sc = np.hypot(dcx - gcx, dcy - gcy) / np.maximum(hd, 1e-9)
                np.minimum.at(gt_best, gi, sc)
                np.minimum.at(det_best, dpos, sc)
            else:
                ix1 = np.maximum(G4[:, 0], D4[:, 0]); iy1 = np.maximum(G4[:, 1], D4[:, 1])
                ix2 = np.minimum(G4[:, 2], D4[:, 2]); iy2 = np.minimum(G4[:, 3], D4[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                da = np.clip(D4[:, 2] - D4[:, 0], 0, None) * np.clip(D4[:, 3] - D4[:, 1], 0, None)
                if match == "iou":
                    ga = np.clip(G4[:, 2] - G4[:, 0], 0, None) * np.clip(G4[:, 3] - G4[:, 1], 0, None)
                    sc = inter / np.maximum(ga + da - inter, 1e-9)
                else:                                          # iod
                    sc = np.where(da > 0, inter / np.maximum(da, 1e-9), 0.0)
                np.maximum.at(gt_best, gi, sc)
                np.maximum.at(det_best, dpos, sc)

    if n_gt:                                                   # per-GT-frame best
        gf = gr[:, 0].astype(np.int64)
        og = np.argsort(gf, kind="stable")
        st = np.r_[0, np.flatnonzero(np.diff(gf[og])) + 1]
        red = np.minimum if match == "center" else np.maximum
        gtf_best = red.reduceat(gt_best[og], st)
    else:
        gtf_best = np.zeros(0, float)
    return gt_best, det_best, gtf_best


def _hists(D, stage):
    """Stage array + INSIDE/OUTSIDE-GT per-stage histograms for the bar charts."""
    ig = D["ig"].astype(bool)
    return {"stage": stage,
            "in":  (np.bincount(stage[ig], minlength=6).astype(np.int64)
                    if ig.any() else np.zeros(6, np.int64)),
            "out": (np.bincount(stage[~ig], minlength=6).astype(np.int64)
                    if (~ig).any() else np.zeros(6, np.int64))}


def _components_global(D, mask):
    """8-connected components of the masked pixels over EVERY frame AT ONCE.

    Pixels get a unique integer key (frame, y, x) on a virtually padded grid (so
    neighbour arithmetic can't wrap across rows/frames); the 4 forward-neighbour
    offsets are resolved with searchsorted, and one scipy.sparse.csgraph
    connected_components call labels everything. Replaces ~3400 per-frame
    ndimage.label calls with a handful of vector ops — the core speedup."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    fr = D["fr"][idx].astype(np.int64)
    x = D["x"][idx].astype(np.int64); y = D["y"][idx].astype(np.int64)
    snr = D["snr"][idx].astype(np.float64)
    W2 = int(D["W"]) + 1; H2 = int(D["H"]) + 1
    key = (fr * H2 + y) * W2 + x
    srt = np.argsort(key)
    ks = key[srt]
    ei, ej = [], []
    for off in (1, W2 - 1, W2, W2 + 1):              # E, SW, S, SE neighbours
        nk = ks + off
        pos = np.searchsorted(ks, nk)
        ok = pos < ks.size
        pos = np.clip(pos, 0, ks.size - 1)
        hit = ok & (ks[pos] == nk)
        ei.append(np.flatnonzero(hit)); ej.append(pos[hit])
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = ks.size
    ei = np.concatenate(ei); ej = np.concatenate(ej)
    graph = coo_matrix((np.ones(ei.size, np.int8), (ei, ej)), shape=(n, n))
    ncomp, lab = connected_components(graph, directed=False)
    xs = x[srt]; ys = y[srt]; ss = snr[srt]; fs = fr[srt]
    o2 = np.argsort(lab, kind="stable")
    labs = lab[o2]
    starts = np.r_[0, np.flatnonzero(np.diff(labs)) + 1]
    comp = dict(
        frame=fs[o2][starts],
        x1=np.minimum.reduceat(xs[o2], starts), x2=np.maximum.reduceat(xs[o2], starts),
        y1=np.minimum.reduceat(ys[o2], starts), y2=np.maximum.reduceat(ys[o2], starts),
        npx=np.diff(np.r_[starts, labs.size]).astype(np.int64),
        snr_sum=np.add.reduceat(ss[o2], starts),
    )
    return dict(idx=idx, srt=srt, lab=lab, ncomp=int(ncomp), comp=comp)


def _merge_rects_fast(x1, y1, x2, y2, alpha):
    """Vectorised utils._merge_boxes (identical rule): boxes whose centroid
    distance < alpha × max(w, h) of either box (w/h inclusive) merge transitively;
    each group becomes its envelope. numpy pairwise + one small csgraph CC call
    instead of the O(k²) Python loop — a clutter frame with ~800 boxes went from
    seconds to milliseconds."""
    n = x1.size
    if n == 1:
        return [(int(x1[0]), int(y1[0]), int(x2[0]), int(y2[0]))]
    if n > 2000:                                     # degenerate settings: the n²
        return list(zip(x1.astype(int).tolist(),     # pairwise matrix would blow
                        y1.astype(int).tolist(),     # memory — return unmerged
                        x2.astype(int).tolist(), y2.astype(int).tolist()))
    if _FF is not None:
        return _FF.merge_rects(x1, y1, x2, y2, alpha)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    s = np.maximum(x2 - x1 + 1, y2 - y1 + 1).astype(float)   # max(w, h), inclusive
    dx = cx[:, None] - cx[None, :]; dy = cy[:, None] - cy[None, :]
    thr = alpha * np.maximum(s[:, None], s[None, :])
    adj = (dx * dx + dy * dy) < thr * thr
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ncc, lab = connected_components(coo_matrix(adj), directed=False)
    o = np.argsort(lab, kind="stable")
    st = np.r_[0, np.flatnonzero(np.diff(lab[o])) + 1]
    return [(int(np.min(x1[g])), int(np.min(y1[g])),
             int(np.max(x2[g])), int(np.max(y2[g])))
            for g in np.split(o, st[1:])]


# The pixel gates + global CC only depend on (case, mode, min_events, tau_sf,
# tau_comb) [+ min_rotor_pixels in components mode] — cached so that tuning any
# component/temporal gate (box area, fill, SNR-sum, alpha, persistence, and
# min_rotor_pixels in the per-pixel modes) reuses them and costs milliseconds.
_CC_CACHE = {"key": None, "stage0": None, "cg": None}


def compute_all(D, mode, params):
    """One vectorised pass -> (stage+histograms, RAW per-frame detections).

    Pixel gates are plain vector ops; the component stage uses _components_global
    (one global CC call) with vectorised component gates (min_rotor_pixels,
    min_box_area, fill ratio, SNR-sum); only the final per-frame alpha merge
    loops, and only over frames that actually have surviving components. The
    gates+CC half is cached (see _CC_CACHE) so most slider changes skip it."""
    ck = (D.get("_token"), mode, params["min_events"], params["tau_sf"],
          params["tau_comb"],
          params["min_rotor_pixels"] if mode == "components" else None)
    if _CC_CACHE["key"] == ck:
        stage = _CC_CACHE["stage0"].copy()
        cg = _CC_CACHE["cg"]
    else:
        gc = D["gc"]; sf = D["sf"]; snr = D["snr"]
        pass_me = gc >= params["min_events"]
        ok_sf = pass_me & (sf <= params["tau_sf"])
        ok_snr = ok_sf & (snr >= params["tau_comb"])
        stage = np.empty(gc.size, np.int8)
        stage[~pass_me] = EX.F_MINEV
        stage[pass_me & ~ok_sf] = EX.F_SF
        stage[ok_sf & ~ok_snr] = EX.F_SNR
        if mode == "components":
            ok_size = ok_snr & (D["size"] >= params["min_rotor_pixels"])
            stage[ok_snr & ~ok_size] = EX.F_MINPX
            stage[ok_size] = EX.PASS
            cg = _components_global(D, ok_size)
        else:
            cg = _components_global(D, ok_snr)      # green px stages filled below
        _CC_CACHE.update(key=ck, stage0=stage.copy(), cg=cg)
    comp_min_px = 1 if mode == "components" else int(params["min_rotor_pixels"])

    dets = {}
    if cg is not None:
        c = cg["comp"]
        area = (c["x2"] - c["x1"] + 1) * (c["y2"] - c["y1"] + 1)   # BoundingBox.area rule
        keep = (c["npx"] >= comp_min_px) & (area >= params["min_box_area"])
        mfr = float(params.get("min_fill_ratio", 0.0))
        if mfr > 0:
            keep &= c["npx"] >= mfr * np.maximum(1, area)
        mss = float(params.get("min_snr_sum", 0.0))
        if mss > 0:
            keep &= c["snr_sum"] >= mss
        mavg = float(params.get("min_snr_avg", 0.0))
        if mavg > 0:                                  # mean pixel SNR (size-free)
            keep &= c["snr_sum"] >= mavg * np.maximum(1, c["npx"])
        if mode != "components":                      # stage: PASS vs fail-box
            px_keep = keep[cg["lab"]]
            gidx = cg["idx"][cg["srt"]]
            stage[gidx[px_keep]] = EX.PASS
            stage[gidx[~px_keep]] = EX.F_BOX
        ki = np.flatnonzero(keep)
        if ki.size:
            kf = c["frame"][ki]
            alpha = float(params.get("alpha", 1.5))
            o = np.argsort(kf, kind="stable")
            for seg in np.split(o, np.flatnonzero(np.diff(kf[o])) + 1):
                comps = ki[seg]
                if comps.size > _DET_FRAME_CAP:      # degenerate settings guard
                    comps = comps[np.argsort(c["snr_sum"][comps])[::-1][:_DET_FRAME_CAP]]
                f = int(c["frame"][comps[0]])
                dets[f] = _merge_rects_fast(c["x1"][comps], c["y1"][comps],
                                            c["x2"][comps], c["y2"][comps], alpha)
    if cg is None:
        comps_out = None
    else:                                            # pre-gate comps (boxes view);
        c0 = cg["comp"]                              # no tracking -> no relaxation
        comps_out = dict(c0, relax=np.zeros(c0["frame"].size, np.float32))
    return _hists(D, stage), dets, comps_out


def first_gt_frame(D):
    """First frame with a GT box, or None when the case has no GT."""
    return int(D["gt_rects"][:, 0].min()) if D["gt_rects"].size else None


# case folders searched for RGB backdrops: the exported lab runs
_RGB_CASE_ROOTS = (os.path.join(_FRED, "runs", "dataset"),)


def _rgb_paths(seq):
    """The case's RGB frames (RGB/ in a HuggingFace case, PADDED_RGB/ in an
    older one) in capture order, or [] when the case folder is absent."""
    import glob
    import hf_cases
    for root in _RGB_CASE_ROOTS:
        try:
            d = hf_cases.rgb_dir(os.path.join(root, str(seq)))
        except FileNotFoundError:
            continue
        # names carry HH_MM_SS.ffffff capture times, so name order = time order
        return sorted(glob.glob(os.path.join(d, "*.jpg"))
                      + glob.glob(os.path.join(d, "*.png")))
    return []


def _rgb_frame(seq, f, H, W, _cache={}):
    """Dimmed RGB frame for event window `f` as an (H, W, 3) uint8 backdrop, or
    None when the case folder / PIL is absent. The RGB and event streams share a
    time origin and the RGB runs on the same 30 fps grid as the 33.3 ms event
    windows, so window f shows RGB frame f. One decoded frame is cached."""
    if _cache.get("seq") != str(seq):
        _cache.update(seq=str(seq), paths=_rgb_paths(seq), idx=None, img=None)
    if not _cache["paths"]:
        return None
    idx = max(0, min(int(f), len(_cache["paths"]) - 1))
    if _cache["idx"] != idx:
        try:
            from PIL import Image
            im = Image.open(_cache["paths"][idx]).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H))
            _cache["img"] = (np.asarray(im) * 0.45).astype(np.uint8)
            _cache["idx"] = idx
        except Exception:
            return None
    return _cache["img"]


# ── post-detection low-frequency veto ───────────────────────────────────────────
# Off unless params["veto_peak"] > 0. For every FINAL detection box the raw events
# inside it (that window) are pooled into one spectrum on a 4x finer frequency grid
# and divided by its median: f = the tallest bin in 60-250 Hz, h1 its height,
# hh = the tallest bin within +-2 bins of 2f or 3f. The box is deleted when
# h1 >= veto_peak AND hh >= veto_harm: a strong periodic source BELOW the rotor
# bands (a fan, a mains-driven light) whose harmonics the comb took for a rotor.
# Boxes with < 16 events are kept. It needs the case's raw events; the (h1, hh)
# of every measured box are cached next to the feature tables (veto_<seq>.npz), so
# replays and threshold changes do not read the events again.
VETO_BAND_HZ = (60.0, 250.0)
VETO_MAX_HZ = 6300.0
VETO_OS = 4
VETO_MIN_EVENTS = 16
_VETO_MEM = {}


def _event_path(seq):
    """The case's raw event file, or None when the case is not on disk."""
    import hf_cases
    for root in _RGB_CASE_ROOTS:
        try:
            return hf_cases.event_file(os.path.join(root, str(seq)))
        except FileNotFoundError:
            continue
    return None


def _veto_window_us(data_dir, seq, mode):
    import json
    try:
        with open(os.path.join(data_dir, f"meta_{seq}_{mode}.json"),
                  encoding="utf-8") as fh:
            return int(json.load(fh).get("window_us", 33333))
    except (OSError, ValueError):
        return 33333


def _veto_measure(t_rel, pol, window_us):
    """(h1, hh) of one box. t_rel = event times [us] from the window start, pol = +-1.
    Timestamps are whole microseconds, so the spectrum on the 1 / (VETO_OS x window)
    grid is exactly the FFT of the per-microsecond signed event counts."""
    h = np.bincount(t_rel, weights=pol, minlength=window_us)[:window_us]
    k = int(round(VETO_MAX_HZ * VETO_OS * window_us / 1e6))
    p = np.abs(np.fft.rfft(h, n=VETO_OS * window_us)[:k]) ** 2
    pn = p / max(float(np.median(p[1:])), 1e-12)
    hz = 1e6 / (VETO_OS * window_us)
    lo = int(np.ceil(VETO_BAND_HZ[0] / hz)); hi = int(np.floor(VETO_BAND_HZ[1] / hz))
    i1 = lo + int(np.argmax(pn[lo:hi + 1]))
    hh = max(float(pn[max(1, i1 * m - 2): i1 * m + 3].max()) for m in (2, 3))
    return float(pn[i1]), hh


def _veto_scores(D, dets):
    """{(frame, x1, y1, x2, y2): (h1, hh)} covering every box of `dets`: read from
    veto_<seq>.npz, boxes not in it are measured from the raw events and added.
    Without the raw events the unmeasured boxes stay unknown (and are kept)."""
    data_dir, seq, mode, wscale = D["_token"]
    path = os.path.join(data_dir, f"veto_{seq}.npz")
    sc = _VETO_MEM.get(path)
    if sc is None:
        sc = {}
        if os.path.exists(path):
            try:
                with np.load(path) as z:             # closed, so the file can be replaced
                    sc = {tuple(int(v) for v in b): (float(a[0]), float(a[1]))
                          for b, a in zip(z["box"], z["val"])}
            except Exception:
                sc = {}
        _VETO_MEM.clear(); _VETO_MEM[path] = sc      # one case kept in memory
    need = {}
    for f, rects in dets.items():
        for r in rects:
            key = (int(f), int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            if key not in sc:
                need.setdefault(int(f), set()).add(key)
    if not need or wscale != 1:
        return sc
    ev_path = _event_path(seq)
    if ev_path is None:
        print(f"[veto] case {seq}: raw events not on disk - "
              f"{sum(len(v) for v in need.values())} boxes not measured (kept)")
        return sc
    from utils import _normalise_polarity
    w_us = _veto_window_us(data_dir, seq, mode)
    rd = EX.dv.RawReader(ev_path, max_events=20_000_000)
    cur = 0
    for f in sorted(need):
        t0 = (f - 1) * w_us                          # table frame f = window f-1
        if f < 1:
            sc.update((key, (0.0, 0.0)) for key in need[f]); continue
        if t0 != cur:
            rd.seek_time(t0)
        evs = rd.load_delta_t(w_us); cur = t0 + w_us
        # a seek lands on the file index at or before t0, so the read can start
        # a little early: keep exactly this window
        evs = evs[(evs["t"] >= t0) & (evs["t"] < t0 + w_us)]
        x = evs["x"]; y = evs["y"]
        for key in need[f]:
            m = (x >= key[1]) & (x <= key[3]) & (y >= key[2]) & (y <= key[4])
            if int(m.sum()) < VETO_MIN_EVENTS:
                sc[key] = (0.0, 0.0)
            else:
                sc[key] = _veto_measure((evs["t"][m] - t0).astype(np.int64),
                                        _normalise_polarity(evs["p"][m]), w_us)
    del rd
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, box=np.array(list(sc.keys()), np.int32).reshape(-1, 5),
             val=np.array(list(sc.values()), np.float32).reshape(-1, 2))
    os.replace(tmp, path)
    return sc


def _apply_veto(D, dets, params):
    """(kept, vetoed), both {frame: [rects]}: the boxes the veto deletes at
    params['veto_peak'] / params['veto_harm'] are moved to `vetoed`."""
    peak = float(params["veto_peak"]); harm = float(params.get("veto_harm", 20.0))
    sc = _veto_scores(D, dets)
    kept, vetoed = {}, {}
    for f, rects in dets.items():
        keep = []
        for r in rects:
            h1, hh = sc.get((int(f), int(r[0]), int(r[1]), int(r[2]), int(r[3])),
                            (0.0, 0.0))
            if h1 >= peak and hh >= harm:
                vetoed.setdefault(f, []).append(r)
            else:
                keep.append(r)
        kept[f] = keep
    return kept, vetoed


def count_matches(D, dets, match, thr, f_lo=None):
    """Per-instance TP/FP/FN of {frame: [rects]} vs GT: every matched GT = 1 TP,
    every unmatched GT = 1 FN, every detection matching no GT = 1 FP (so one good
    detection + two strays in a frame = 1 TP + 2 FP).

    Also frame-level counts: fp_frames = frames with >=1 unmatched detection
    (two wrong boxes in one frame count once), tn_frames = frames with neither
    GT nor any detection.

    f_lo (optional): ignore every frame BEFORE it — detections there are neither
    TPs nor FPs and the frame doesn't count at all (pass first_gt_frame(D) to
    score only from the first sighting on; the noisy pre-drone lead-in of many
    recordings otherwise dominates the FP counts). The returned
    'frames_counted' is the number of frames actually scored — use it instead
    of D['n_frames'] for per-frame rates."""
    if f_lo is not None:
        dets = {f: r for f, r in dets.items() if f >= f_lo}
    frames_counted = (int(D["n_frames"]) if f_lo is None
                      else max(1, int(D["n_frames"]) - (int(f_lo) - 1)))
    gb, db, fb = _score_caches(D, dets, match)
    gp = (gb <= thr) if match == "center" else (gb >= thr)
    dp = (db <= thr) if match == "center" else (db >= thr)
    fh = (fb <= thr) if match == "center" else (fb >= thr)

    gbf = {}
    for row in D["gt_rects"]:
        gbf.setdefault(int(row[0]), []).append(tuple(int(v) for v in row[1:5]))
    fp_frames = 0
    det_frames = set()
    for f, rs in dets.items():
        if not rs:
            continue
        det_frames.add(f)
        gts = gbf.get(f, [])
        if not gts:
            fp_frames += 1
            continue
        for d in rs:
            sc = _best_of((_match_score(g, d, match) for g in gts), match)
            if not _match_ok(sc, thr, match):
                fp_frames += 1
                break
    tn_frames = frames_counted - len(set(gbf) | det_frames)
    return dict(tp=int(gp.sum()), fp=int((~dp).sum()), fn=int(gb.size - gp.sum()),
                gt_frames=int(fb.size), gt_frames_hit=int(fh.sum()),
                fp_frames=int(fp_frames), tn_frames=int(tn_frames),
                frames_counted=int(frames_counted))


def _compute_job(D, mode, params, abort_cb=None):
    """One metrics job: stage+histograms and detections at `params` (the Gaussian-
    relaxed tracking pass when params['tracking'], else the vectorised
    compute_all) + persistence + match-score caches. Returns (res, dets, caches).
    Shared by the GUI's worker process, its in-process fallback, metrics_report
    and the threshold sweep. abort_cb (tracking only): raise _Superseded when a
    newer request is waiting, so a long pass never blocks fresher settings."""
    if params.get("tracking"):
        stage, dets, comps = _track_pass(D, params, mode,
                                         params.get("track_sigma", 20.0),
                                         abort_cb=abort_cb)
        res = _hists(D, stage)
    else:
        res, dets, comps = compute_all(D, mode, params)
    lo, hi = _frame_span(D)
    dets = _apply_persistence(dets, lo, hi, params.get("persist_m", 1),
                              params.get("persist_k", 1))
    if params.get("skip_start"):                     # ignore the pre-GT lead-in
        f0 = first_gt_frame(D)
        if f0 is not None:
            dets = {f: r for f, r in dets.items() if f >= f0}
    if float(params.get("veto_peak", 0) or 0) > 0:   # post-detection veto
        dets, res["vetoed"] = _apply_veto(D, dets, params)
    caches = _score_caches(D, dets, params.get("match", "center"))
    return res, dets, caches, comps


def _metrics_worker(conn):
    """Persistent worker PROCESS: keeps its own load_D cache, computes each job with
    _compute_job and sends back the (small) results. Out-of-process on purpose —
    the job is dominated by pure-Python loops that hold the GIL, so a worker
    *thread* starves the GUI; a process leaves it fully responsive."""
    while True:
        try:
            msg = conn.recv()
            while conn.poll():                       # drain to the newest request
                msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        try:
            D = load_D(msg["seq"], msg["mode"], msg["data_dir"],
                       msg.get("wscale", 1))
            res, dets, caches, comps = _compute_job(D, msg["mode"], msg["params"],
                                                    abort_cb=conn.poll)
            conn.send((msg["gen"], res, dets, caches, comps))
        except _Superseded:
            continue                                 # newer request is waiting
        except Exception as exc:
            conn.send((msg["gen"], "error", repr(exc), None, None))


def _iod(g, d):
    """Intersection-over-Detection: |g ∩ d| / area(d). 1.0 = the detection lies
    fully inside the GT box — the right overlap measure when the detector finds
    only the rotors of a much larger GT box."""
    ix1 = max(g[0], d[0]); iy1 = max(g[1], d[1])
    ix2 = min(g[2], d[2]); iy2 = min(g[3], d[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    da = max(0, d[2] - d[0]) * max(0, d[3] - d[1])
    return inter / da if da > 0 else 0.0


MATCH_MODES = ("center", "iou", "iod")
MATCH_DEFAULT_THR = {"center": 1.0, "iou": 0.1, "iod": 0.5}


def _match_score(g, d, match):
    """Score of detection d against GT g. iou/iod: HIGHER is better, matched when
    score >= thr. center: distance between centers in units of the GT box
    half-diagonal (the codebase's --match center rule); LOWER is better, matched
    when score <= thr (thr = center-dist-scale, 1.0 by default)."""
    if match == "iou":
        return _iou(g, d)
    if match == "iod":
        return _iod(g, d)
    gcx, gcy = (g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0
    dcx, dcy = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
    hd = 0.5 * ((g[2] - g[0]) ** 2 + (g[3] - g[1]) ** 2) ** 0.5
    return ((dcx - gcx) ** 2 + (dcy - gcy) ** 2) ** 0.5 / max(hd, 1e-9)


def _match_ok(score, thr, match):
    return score <= thr if match == "center" else score >= thr


def _worst(match):
    return float("inf") if match == "center" else 0.0


def _best_of(scores, match):
    scores = list(scores)
    if not scores:
        return _worst(match)
    return min(scores) if match == "center" else max(scores)


_TRACK_MAX_CENTERS = 64      # relaxation anchors per frame (largest boxes first)
_DET_FRAME_CAP = 2000        # keep at most this many components per frame (by SNR);
                             # only degenerate near-zero gates ever reach this


class _Superseded(Exception):
    """Raised inside a worker job when a newer request has already arrived."""


_COAST_DECAY = 0.7           # anchor weight per missed frame while coasting
_TRACK_LIST_CAP = 256        # max simultaneous tracks (degenerate-settings guard)


def _track_pass(D, p, mode, sigma, abort_cb=None, trace=None):
    """Sequential per-frame gating with Gaussian threshold relaxation around a
    small TRACK TABLE (track-before-detect style).

    Base rule: for a pixel at distance d from the nearest track anchor,
    g = w·exp(-d²/2σ²) relaxes the gates toward fully open at the anchor:
        min_events -> max(1, min_events·(1-g))     tau_sf   -> tau_sf + (1-tau_sf)·g
        tau_comb   -> tau_comb·(1-g)               min_rotor_pixels / min_box_area /
                                                   snr gates -> ·(1-g at centroid)

    Optional mechanisms (all off at their defaults, restoring the plain
    previous-frame behaviour):
      track_coast N   (0=off)  anchors survive N missed frames, weight decaying
                               by _COAST_DECAY per miss — bridges 1..N-frame
                               dropouts instead of dying on the first one.
      track_confirm M (1=off)  a track must be seen in M frames before it grants
                               relaxation — stops one-frame FPs from
                               self-reinforcing.
      track_acquire A (1=off)  far from any track the gates are TIGHTENED ×A
                               (tau_sf ÷A), relaxing to fully open on-track:
                               strict acquisition, loose sustain. Multiplier is
                               A·(1-g).
      track_predict   (off)    anchors extrapolate with the track's velocity —
                               keeps fast movers inside the σ radius.

    Guards: vectorised per-component stats, _DET_FRAME_CAP strongest components
    per frame, _TRACK_MAX_CENTERS anchors, _TRACK_LIST_CAP tracks, and
    `abort_cb` aborts when a newer request supersedes this one. Returns
    (stage, {frame: [det rects]}, comps)."""
    gc = D["gc"]; size = D["size"]; sf = D["sf"]; snr = D["snr"]
    X = D["x"].astype(np.int64); Y = D["y"].astype(np.int64)
    stage = np.full(D["fr"].size, EX.F_MINEV, dtype=np.int8)
    dets = {}
    acc = {k_: [] for k_ in ("frame", "x1", "y1", "x2", "y2", "npx", "snr_sum",
                             "relax")}
    fill = float(p.get("min_fill_ratio", 0.0))
    mss = float(p.get("min_snr_sum", 0.0))
    mavg = float(p.get("min_snr_avg", 0.0))
    alpha = float(p.get("alpha", 1.5))
    A = max(1.0, float(p.get("track_acquire", 1.0)))
    coast = max(0, int(p.get("track_coast", 0)))
    confirm = max(1, int(p.get("track_confirm", 1)))
    predict = bool(p.get("track_predict", False))
    # NOTE: tau_sf is NOT acquisition-tightened — SF is a bounded [0,1] flatness
    # measure whose useful range sits at 0.5-0.9; dividing it by A rejects nearly
    # everything. A applies to the count/SNR-like gates only.
    tau_sf0 = p["tau_sf"]
    two_sig2 = 2.0 * float(sigma) ** 2
    match_r2 = (2.0 * float(sigma)) ** 2             # det-track association radius²
    tracks = []                                      # dicts: x,y,vx,vy,hits,miss,area

    def _anchors():
        """(positions (k,2), weights (k,)) of confirmed, alive tracks."""
        alive = [t for t in tracks if t["hits"] >= confirm]
        if not alive:
            return None, None
        if len(alive) > _TRACK_MAX_CENTERS:
            alive = sorted(alive, key=lambda t: -t["area"])[:_TRACK_MAX_CENTERS]
        steps = np.array([t["miss"] + 1 for t in alive], float)
        ax = np.array([t["x"] for t in alive], float)
        ay = np.array([t["y"] for t in alive], float)
        if predict:
            ax = ax + np.array([t["vx"] for t in alive]) * steps
            ay = ay + np.array([t["vy"] for t in alive]) * steps
        w = _COAST_DECAY ** np.array([t["miss"] for t in alive], float)
        return np.column_stack([ax, ay]), w

    for i in range(D["uniq_frames"].size):
        if abort_cb is not None and (i & 63) == 0 and abort_cb():
            raise _Superseded()
        f = int(D["uniq_frames"][i])
        s = int(D["frame_starts"][i]); e = s + int(D["frame_counts"][i])
        xs = X[s:e]; ys = Y[s:e]
        anc, anc_w = _anchors()
        if trace is not None:                        # per-frame prior snapshot
            trace[f] = (anc, anc_w, len(tracks))     # (viewer overlay only)
        if anc is not None:
            d2 = ((xs[:, None] - anc[:, 0]) ** 2 + (ys[:, None] - anc[:, 1]) ** 2)
            g = (anc_w[None, :] * np.exp(-d2 / two_sig2)).max(axis=1)
        else:
            g = np.zeros(e - s)
        relax = A * (1.0 - g)                        # A× far -> 0 on-track
        me  = np.maximum(1.0, p["min_events"] * relax)
        tsf = tau_sf0 + (1.0 - tau_sf0) * g
        tcb = p["tau_comb"] * relax
        st = np.empty(e - s, dtype=np.int8)
        pass_me = gc[s:e] >= me
        ok_sf  = pass_me & (sf[s:e] <= tsf)
        ok_snr = ok_sf   & (snr[s:e] >= tcb)
        st[~pass_me] = EX.F_MINEV
        st[pass_me & ~ok_sf] = EX.F_SF
        st[ok_sf & ~ok_snr] = EX.F_SNR
        rects = []
        if mode == "components":
            ok_size = ok_snr & (size[s:e] >= p["min_rotor_pixels"] * relax)
            st[ok_snr & ~ok_size] = EX.F_MINPX
            st[ok_size] = EX.PASS
            if ok_size.any():
                rects = [(b.x1, b.y1, b.x2, b.y2) for b in _boxes_from_pass(
                    xs[ok_size], ys[ok_size], p["min_box_area"], alpha, fill,
                    snr=snr[s:e][ok_size], min_snr_sum=mss, min_snr_avg=mavg)]
        else:
            gi = np.flatnonzero(ok_snr)
            if gi.size:
                sx = xs[gi]; sy = ys[gi]
                s_snr = snr[s:e][gi].astype(np.float64)
                pix_lab = _label_px(sx, sy)                  # 0-based component id
                o = np.argsort(pix_lab, kind="stable")
                pl = pix_lab[o]
                starts = np.r_[0, np.flatnonzero(np.diff(pl)) + 1]
                npx = np.diff(np.r_[starts, pl.size])
                bx1 = np.minimum.reduceat(sx[o], starts)
                bx2 = np.maximum.reduceat(sx[o], starts)
                by1 = np.minimum.reduceat(sy[o], starts)
                by2 = np.maximum.reduceat(sy[o], starts)
                csnr = np.add.reduceat(s_snr[o], starts)
                acc["frame"].append(np.full(npx.size, f, np.int64))
                acc["x1"].append(bx1); acc["y1"].append(by1)
                acc["x2"].append(bx2); acc["y2"].append(by2)
                acc["npx"].append(npx); acc["snr_sum"].append(csnr)
                if anc is not None:                          # centroid relaxation
                    mcx = np.add.reduceat(sx[o].astype(np.float64), starts) / npx
                    mcy = np.add.reduceat(sy[o].astype(np.float64), starts) / npx
                    dc = ((mcx[:, None] - anc[:, 0]) ** 2
                          + (mcy[:, None] - anc[:, 1]) ** 2)
                    g_c = (anc_w[None, :] * np.exp(-dc / two_sig2)).max(axis=1)
                else:
                    g_c = np.zeros(npx.size)
                relax_c = A * (1.0 - g_c)
                # boxes view expects relf = 1-rel == the detector's multiplier
                acc["relax"].append((1.0 - relax_c).astype(np.float32))
                area = (bx2 - bx1 + 1) * (by2 - by1 + 1)     # BoundingBox.area rule
                keep = ((npx >= p["min_rotor_pixels"] * relax_c)
                        & (area >= p["min_box_area"] * relax_c))
                if fill > 0.0:
                    keep &= npx >= fill * np.maximum(1, area)
                if mss > 0.0:
                    keep &= csnr >= mss * relax_c
                if mavg > 0.0:
                    keep &= csnr >= mavg * npx * relax_c
                ki = np.flatnonzero(keep)
                if ki.size > _DET_FRAME_CAP:                 # degenerate settings
                    ki = ki[np.argsort(csnr[ki])[::-1][:_DET_FRAME_CAP]]
                    keep = np.zeros(npx.size, bool); keep[ki] = True
                keep_px = keep[pix_lab]
                st[gi[keep_px]] = EX.PASS
                st[gi[~keep_px]] = EX.F_BOX
                if ki.size:
                    rects = _merge_rects_fast(bx1[ki], by1[ki], bx2[ki], by2[ki], alpha)
        dets[f] = rects

        # ── update the track table with this frame's detections ────────────────
        if rects:
            R = np.asarray(rects, float)
            dcx = (R[:, 0] + R[:, 2]) / 2.0; dcy = (R[:, 1] + R[:, 3]) / 2.0
            dar = (R[:, 2] - R[:, 0]) * (R[:, 3] - R[:, 1])
            if dcx.size > 300:                       # association guard
                selb = np.argsort(dar)[::-1][:300]
                dcx = dcx[selb]; dcy = dcy[selb]; dar = dar[selb]
            used = np.zeros(dcx.size, bool)
            for t in tracks:                         # greedy nearest-det per track
                steps = t["miss"] + 1
                px_ = t["x"] + (t["vx"] * steps if predict else 0.0)
                py_ = t["y"] + (t["vy"] * steps if predict else 0.0)
                d2t = (dcx - px_) ** 2 + (dcy - py_) ** 2
                d2t[used] = np.inf
                j = int(np.argmin(d2t)) if d2t.size else -1
                if j >= 0 and d2t[j] <= match_r2:
                    used[j] = True
                    t["vx"] = (dcx[j] - t["x"]) / steps
                    t["vy"] = (dcy[j] - t["y"]) / steps
                    t["x"] = float(dcx[j]); t["y"] = float(dcy[j])
                    t["area"] = float(dar[j])
                    t["hits"] += 1; t["miss"] = 0
                else:
                    t["miss"] += 1
            tracks = [t for t in tracks if t["miss"] <= coast]
            for j in np.argsort(dar)[::-1]:          # new tracks, largest first
                if used[j] or len(tracks) >= _TRACK_LIST_CAP:
                    continue
                tracks.append(dict(x=float(dcx[j]), y=float(dcy[j]), vx=0.0, vy=0.0,
                                   hits=1, miss=0, area=float(dar[j])))
        else:
            for t in tracks:
                t["miss"] += 1
            tracks = [t for t in tracks if t["miss"] <= coast]
        stage[s:e] = st
    comps = ({k_: np.concatenate(v) for k_, v in acc.items()}
             if acc["frame"] else None)
    return stage, dets, comps


def metrics_report(D, p, mode, match="center", thr=None, per_frame_out=None,
                   min_fill_ratio=0.0, tracking=False, track_sigma=20.0):
    """Per-frame detections vs GT under the chosen MATCH rule -> aggregate
    TP/FP/FN (+ optional per-frame CSV). Counting is PER INSTANCE: every GT
    matched by some detection is one TP, every unmatched GT one FN, and every
    detection matching no GT one FP — so one good detection plus two strays in
    the same frame counts as 1 TP and 2 FP."""
    thr = MATCH_DEFAULT_THR[match] if thr is None else float(thr)
    pp = dict(p); pp.setdefault("min_fill_ratio", min_fill_ratio)
    pp["tracking"] = tracking; pp["track_sigma"] = track_sigma; pp["match"] = match
    _res, dets, _caches, _comps = _compute_job(D, mode, pp)

    gt_by_frame = {}
    for row in D["gt_rects"]:
        gt_by_frame.setdefault(int(row[0]), []).append(tuple(int(v) for v in row[1:5]))
    frames = sorted(set(int(f) for f in D["uniq_frames"]) | set(gt_by_frame))

    if per_frame_out:
        with open(per_frame_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "n_gt", "n_det", "tp", "fp", "fn", "best_score"])
            for f in frames:
                gts = gt_by_frame.get(f, []); dr = dets.get(f, [])
                gsc = [_best_of((_match_score(g, d, match) for d in dr), match)
                       for g in gts]
                dsc = [_best_of((_match_score(g, d, match) for g in gts), match)
                       for d in dr]
                f_tp = sum(_match_ok(v, thr, match) for v in gsc)
                f_fp = sum(not _match_ok(v, thr, match) for v in dsc)
                best = _best_of(gsc if gts else dsc, match)
                w.writerow([f, len(gts), len(dr), f_tp, f_fp, len(gts) - f_tp,
                            round(best, 4) if np.isfinite(best) else ""])

    f_lo = first_gt_frame(D) if pp.get("skip_start") else None
    c = count_matches(D, dets, match, thr, f_lo=f_lo)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(match=match, thr=thr, frames=len(frames),
                gt_frames=c["gt_frames"], gt_frames_hit=c["gt_frames_hit"],
                tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                fp_per_frame=fp / max(1, c["frames_counted"]))


# ── interactive GUI (self-contained; adds case switching + detection/IoU) ───────

def _available_cases(mode, data_dir):
    """Case names with a cached feature table for `mode`, numeric-sorted.
    Either format counts: the binary frames_<seq>_<mode>.npz that load_D prefers
    (convert_cache_npz.py) or the original frames_<seq>_<mode>.csv.gz."""
    import glob
    pre = os.path.join(data_dir, "frames_")
    names = set()
    for suf in (f"_{mode}.npz", f"_{mode}.csv.gz"):
        names.update(os.path.basename(f)[len("frames_"):-len(suf)]
                     for f in glob.glob(f"{pre}*{suf}"))
    return sorted(names, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def _available_modes(seq, data_dir):
    """Detector modes that actually have a CSV for this case — the only ones the
    GUI lets you switch to for it."""
    import glob
    return [m for m in ("baseline", "pool", "pixelize", "components")
            if glob.glob(os.path.join(data_dir, f"frames_{seq}_{m}.csv.gz"))]


def launch_gui(seq, mode, data_dir, p0, match="center", match_thr=None,
               tracking=False, track_sigma=20.0):
    """Explorer GUI driven by the CSVs: per-frame candidate pixels coloured by gate
    stage, GT boxes (cyan) + derived detection boxes (lime), threshold + frame
    sliders, Prev/Next case + mode switching, a match-mode button (center/iou/iod
    with per-instance TP/FP/FN counting), and a tracking toggle that Gaussian-
    relaxes the gates around the previous frame's detections (sigma slider)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, TextBox
    from matplotlib.patches import Rectangle

    modes = _available_modes(seq, data_dir) or [mode]
    if mode not in modes:
        mode = modes[0]
    cases = _available_cases(mode, data_dir) or [seq]
    G = {"cases": cases, "idx": cases.index(seq) if seq in cases else 0,
         "D": load_D(seq, mode, data_dir), "seq": seq, "mode": mode, "modes": modes}
    state = dict(p0); state["frame_pos"] = 0.0
    state.setdefault("min_fill_ratio", 0.0)
    state.setdefault("min_snr_sum", 0.0)               # component SNR-mass gate
    state.setdefault("persist_m", 1)                   # m-of-k temporal persistence
    state.setdefault("persist_k", 1)
    state["match"] = match if match in MATCH_MODES else "center"
    state["match_thr"] = (float(match_thr) if match_thr is not None
                          else MATCH_DEFAULT_THR[state["match"]])
    state["tracking"] = bool(tracking)
    state["track_sigma"] = float(track_sigma)
    _thr_memory = dict(MATCH_DEFAULT_THR)              # last-used thr per match mode
    _thr_memory[state["match"]] = state["match_thr"]
    G["dets"] = None                                   # {frame: [rects]} at current gates
    G["gt_best"] = G["det_best"] = G["gtf_best"] = None  # metric score caches

    # TWO windows: `fig` holds the heavy plots (bar charts + the 1280x720 overlay
    # image); `figc` holds every widget + the info text. A slider drag triggers
    # matplotlib's own draw_idle on ITS figure only — so pressing/dragging redraws
    # just the light controls window, never the heavy plots (that full-figure
    # redraw was the remaining press stall).
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f"frame explorer [{mode}] — view")
    figc = plt.figure(figsize=(10, 6))
    figc.canvas.manager.set_window_title("frame explorer — controls")

    def _swallow_resize_bug(exc):
        # matplotlib < 3.7.2 dispatches a ResizeEvent (no .inaxes) into widget mouse
        # handlers on window resize -> benign AttributeError; ignore just that one.
        if isinstance(exc, AttributeError) and "inaxes" in str(exc):
            return
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    for _f in (fig, figc):
        try:
            _f.canvas.callbacks.exception_handler = _swallow_resize_bug
        except Exception:
            pass
    ax_in  = fig.add_axes([0.06, 0.56, 0.40, 0.38])   # per-stage bars, INSIDE GT
    ax_out = fig.add_axes([0.56, 0.56, 0.40, 0.38])   # per-stage bars, OUTSIDE GT
    ax_ov  = fig.add_axes([0.28, 0.03, 0.46, 0.45]); ax_ov.set_xticks([]); ax_ov.set_yticks([])
    ax_txt = figc.add_axes([0.04, 0.72, 0.92, 0.26]); ax_txt.axis("off")
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)

    def draw_region(ax, c, title):
        """Vertical per-gate-stage bars: how many candidate pixels failed each check
        (min_events / SF / SNR / box) vs PASS, as a % of the region's candidates."""
        ax.clear()
        ORDER = EX.stage_order(G["mode"])             # stages differ per mode
        tot = int(c.sum())
        xs = np.arange(len(ORDER))
        vals = [c[code] / tot * 100 if tot else 0 for code in ORDER]
        ax.bar(xs, vals, color=[EX.STAGE_HEX[code] for code in ORDER])
        for xi, code, v in zip(xs, ORDER, vals):
            ax.text(xi, v + 1, f"{int(c[code])}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([EX.STAGE_LABEL[code].replace("fail ", "✗") for code in ORDER],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 105); ax.set_ylabel("% of candidates", fontsize=8)
        ax.set_title(f"{title}  ({tot} px)", fontsize=10)

    # ── background worker: the slow parts (tracking pass / det derivation / score
    # caches) run in a separate PROCESS so the GUI is never GIL-starved (the job is
    # Python-loop heavy; a thread would freeze the sliders). A poll timer applies
    # finished results; a generation counter discards superseded jobs. Falls back
    # to a thread if the process can't start.
    import threading
    import multiprocessing as _mp
    _job = {"gen": 0, "out": None}
    _worker = {"proc": None, "conn": None}

    def _ensure_worker():
        if _worker["proc"] is not None and _worker["proc"].is_alive():
            return _worker["conn"]
        try:
            parent, child = _mp.Pipe()
            pr = _mp.Process(target=_metrics_worker, args=(child,), daemon=True)
            pr.start()
            _worker["proc"], _worker["conn"] = pr, parent
            return parent
        except Exception:
            return None

    def _start_metrics_job():
        _job["gen"] += 1
        gen = _job["gen"]
        params = dict(state)
        conn = _ensure_worker()
        if conn is not None:
            try:
                conn.send(dict(gen=gen, seq=G["seq"], mode=G["mode"],
                               data_dir=data_dir, params=params))
                return
            except Exception:
                _worker["proc"] = _worker["conn"] = None   # broken pipe -> fallback
        D = G["D"]; mode_ = G["mode"]                  # in-process fallback (thread)

        def work():
            try:
                _job["out"] = (gen,) + _compute_job(D, mode_, params)
            except Exception as exc:
                _job["out"] = (gen, "error", repr(exc), None)

        threading.Thread(target=work, daemon=True).start()

    def _metrics_lines():
        gb = G.get("gt_best")
        if gb is None:
            return "metrics: (computing …)"
        mtch = state["match"]; thr = state["match_thr"]
        db = G["det_best"]; fb = G["gtf_best"]
        gp = (gb <= thr) if mtch == "center" else (gb >= thr)
        dp = (db <= thr) if mtch == "center" else (db >= thr)
        fh = (fb <= thr) if mtch == "center" else (fb >= thr)
        tp = int(gp.sum()); fn = gb.size - tp; fp = int((~dp).sum())
        nf = max(1, G["D"]["n_frames"])
        return (f"match={mtch}  thr={thr:.2f}:  TP={tp}  FP={fp}  FN={fn}  "
                f"FP/frame={fp / nf:.3f}\n"
                f"GT frames hit {int(fh.sum())} / missed {fb.size - int(fh.sum())}  "
                f"(n={fb.size})"
                + (f"   [tracking σ={state['track_sigma']:.0f}]" if state["tracking"] else ""))

    def _default_frame(D):
        if D["uniq_frames"].size == 0:
            return 0.0
        ig = D["ig"].astype(bool)
        if ig.any():
            pos = int(np.searchsorted(D["uniq_frames"], int(np.argmax(np.bincount(D["fr"][ig])))))
            return min(pos, D["uniq_frames"].size - 1) / max(1, D["uniq_frames"].size - 1)
        return 0.5
    state["frame_pos"] = _default_frame(G["D"])

    def _draw_text():
        D = G["D"]
        txt.set_text(
            f"case {G['seq']}  ({G['idx']+1}/{len(G['cases'])})\n"
            f"mode {G['mode']}   [avail: {', '.join(G['modes'])}]\n\n"
            f"px  : ev>={state['min_events']}  sf<={state['tau_sf']:.3f}  "
            f"snr>={state['tau_comb']:.2f}\n"
            f"comp: px>={state['min_rotor_pixels']}  fill>={state['min_fill_ratio']:.2f}  "
            f"area>={state['min_box_area']}  snr_sum>={state['min_snr_sum']:.0f}\n"
            f"merge a={state['alpha']:.2f}   persist {state['persist_m']}/{state['persist_k']}"
            f"   tracking {'ON s=%.0f' % state['track_sigma'] if state['tracking'] else 'off'}\n\n"
            f"frames={D['n_frames']}  GT frames={D['n_gt']}\n"
            f"total candidate px={D['gc'].size}\n\n" + _metrics_lines())

    def _draw_overlay():
        """Redraw ONLY the current-frame overlay from the cached stage array — cheap,
        so frame-scrubbing stays instant. Uses G['res'] (last compute_stats)."""
        D = G["D"]; ax_ov.clear(); ax_ov.set_xticks([]); ax_ov.set_yticks([])
        r = G.get("res")
        if r is None or D["uniq_frames"].size == 0:
            ax_ov.set_title("no candidate pixels"); return
        stage = r["stage"]
        i = int(round(state["frame_pos"] * (D["uniq_frames"].size - 1)))
        i = max(0, min(i, D["uniq_frames"].size - 1))
        f = int(D["uniq_frames"][i]); s = int(D["frame_starts"][i]); e = s + int(D["frame_counts"][i])
        canvas = np.full((D["H"], D["W"], 3), 18, np.uint8)
        xs = D["x"][s:e].astype(np.int64); ys = D["y"][s:e].astype(np.int64); st = stage[s:e]
        for code in (EX.F_MINEV, EX.F_SF, EX.F_SNR, EX.F_MINPX, EX.F_BOX, EX.PASS):
            m = st == code
            if m.any():
                canvas[ys[m], xs[m]] = EX.STAGE_RGB[code]
        ax_ov.imshow(canvas, origin="upper", interpolation="nearest")
        gr = D["gt_rects"]
        gts = [tuple(int(v) for v in row[1:5]) for row in gr[gr[:, 0] == f]] if gr.size else []
        for (x1, y1, x2, y2) in gts:
            ax_ov.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", lw=1.5))
        if G["dets"] is not None:                     # tracking pass / metrics cache
            det_rects = list(G["dets"].get(f, []))
        else:
            pm = st == EX.PASS
            det_rects = ([(b.x1, b.y1, b.x2, b.y2) for b in _boxes_from_pass(
                xs[pm], ys[pm], state["min_box_area"], state["alpha"],
                state["min_fill_ratio"], snr=D["snr"][s:e][pm],
                min_snr_sum=state["min_snr_sum"])] if pm.any() else [])
        for (bx1, by1, bx2, by2) in det_rects:
            ax_ov.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                      fill=False, edgecolor="lime", lw=1.5))
        vet = list((r.get("vetoed") or {}).get(f, []))   # removed by the veto
        for (bx1, by1, bx2, by2) in vet:
            ax_ov.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False,
                                      edgecolor="orange", lw=1.5, ls="--"))
        vlab = f", {len(vet)} vetoed (orange)" if vet else ""
        mtch = state["match"]
        best = _best_of((_match_score(g, d, mtch) for g in gts for d in det_rects), mtch)
        blab = (f"best {mtch}={best:.2f}" if (gts and det_rects and np.isfinite(best))
                else "no match")
        ax_ov.set_title(f"{G['seq']}  frame {f}: {len(gts)} GT (cyan), "
                        f"{len(det_rects)} det (lime){vlab}, {blab}", fontsize=10)

    def refresh(_=None):
        """UI-only: mark the caches stale and hand the WHOLE computation (stage,
        detections, metrics) to the worker process. Nothing heavy ever runs on
        the GUI thread, so pressing/dragging can't stall."""
        G["gt_best"] = None                           # metrics line -> "computing…"
        _draw_text(); figc.canvas.draw_idle()
        _start_metrics_job()

    def _apply_job(_=None):
        """Poll-timer callback: publish a finished worker result to the GUI."""
        out = None
        conn = _worker["conn"]
        if conn is not None:
            try:
                while conn.poll():                    # keep only the newest result
                    out = conn.recv()
            except (EOFError, OSError):
                _worker["proc"] = _worker["conn"] = None
        if out is None:
            out = _job.get("out")
            _job["out"] = None
        if out is None:
            return
        gen, res, dets, caches = out
        if gen != _job["gen"]:                        # superseded while running
            return
        if isinstance(res, str) and res == "error":
            txt.set_text(f"metrics failed:\n{dets!r}")
            figc.canvas.draw_idle()
            return
        G["res"] = res
        draw_region(ax_in, res["in"], "INSIDE GT")
        draw_region(ax_out, res["out"], "OUTSIDE GT")
        G["dets"] = dets
        G["gt_best"], G["det_best"], G["gtf_best"] = caches
        _draw_overlay(); _draw_text()
        fig.canvas.draw_idle(); figc.canvas.draw_idle()

    # Debounce: sliders schedule the heavy refresh ~200 ms after the last change —
    # but NEVER while the mouse button is held (a refresh landing mid-press froze
    # the start of every drag). While pressed we just re-arm; the release kicks it.
    _timer = figc.canvas.new_timer(interval=200)
    try:
        _timer.single_shot = True
    except Exception:
        pass
    _pending = {"fn": None}
    _mouse = {"down": False}

    def _fire_pending():
        try:
            _timer.stop()                             # stop even if single_shot is ignored
        except Exception:
            pass
        if _mouse["down"]:                            # mid-drag: wait for release
            return
        fn = _pending["fn"]; _pending["fn"] = None
        if fn:
            fn()

    _timer.add_callback(_fire_pending)

    def _on_press(_e):
        _mouse["down"] = True

    def _on_release(_e):
        _mouse["down"] = False
        if _pending["fn"] is not None:                # run the deferred refresh now
            try:
                _timer.stop(); _timer.start()
            except Exception:
                _fire_pending()

    figc.canvas.mpl_connect("button_press_event", _on_press)
    figc.canvas.mpl_connect("button_release_event", _on_release)

    # Repeating poll timer that publishes finished worker results to the GUI.
    _poll_timer = figc.canvas.new_timer(interval=120)
    _poll_timer.add_callback(_apply_job)
    try:
        _poll_timer.start()
    except Exception:
        pass

    def _debounce(fn):
        _pending["fn"] = fn
        try:
            _timer.stop()
        except Exception:
            pass
        _timer.start()

    # Throttled controls repaint: sliders have drawon=False (no draw per motion);
    # anything that touches the controls window just sets a dirty flag and this
    # 150 ms timer repaints it — so a press/drag costs ZERO synchronous draws.
    _dirty = {"c": False}

    def _mark_dirty():
        _dirty["c"] = True

    _throttle = figc.canvas.new_timer(interval=150)

    def _throttle_tick(_=None):
        if _dirty["c"]:
            _dirty["c"] = False
            figc.canvas.draw_idle()

    _throttle.add_callback(_throttle_tick)
    try:
        _throttle.start()
    except Exception:
        pass

    def _light_overlay():
        _draw_overlay(); fig.canvas.draw_idle()      # heavy window: overlay only
        _draw_text(); _mark_dirty()

    def _light_text():
        _draw_text(); _mark_dirty()                  # repainted by the throttle timer

    # Each threshold gets a slider AND an editable value box: type an exact value
    # (Enter to apply) — including one BEYOND the slider's range, which auto-extends
    # the slider. The box always mirrors the slider's current value.
    INT_PARAMS = ("min_events", "min_rotor_pixels", "min_box_area",
                  "persist_m", "persist_k")
    sliders, boxes = {}, {}

    def _fmt(name, v):
        return str(int(round(v))) if name in INT_PARAMS else f"{float(v):.4g}"

    def _sync_box(name, v):
        boxes[name].text_disp.set_text(_fmt(name, v))   # .text reflects the display
        _mark_dirty()                                   # repainted by the throttle

    def _add_bar(name, lo, hi, step, y, apply_fn, x=0.14, w=0.55):
        axs = figc.add_axes([x, y, w, 0.035])
        sl = Slider(axs, name, lo, hi, valinit=state.get(name, lo), valstep=step)
        sl.valtext.set_visible(False)                 # the value box shows it instead
        sl.drawon = False                             # no draw per press/motion —
        axb = figc.add_axes([x + w + 0.02, y, 0.09, 0.04])   # the throttle repaints
        tb = TextBox(axb, "", initial=_fmt(name, sl.val))
        sliders[name] = sl; boxes[name] = tb

        def _on_slider(v):
            val = int(round(v)) if name in INT_PARAMS else float(v)
            _sync_box(name, val); apply_fn(val)

        def _on_box(text):
            try:
                val = int(round(float(text))) if name in INT_PARAMS else float(text)
            except ValueError:
                _sync_box(name, sl.val); return
            if val < sl.valmin:
                sl.valmin = val; sl.ax.set_xlim(val, sl.valmax)
            if val > sl.valmax:
                sl.valmax = val; sl.ax.set_xlim(sl.valmin, val)
            sl.eventson = False; sl.set_val(val); sl.eventson = True   # no double apply
            _sync_box(name, val); apply_fn(val)

        sl.on_changed(_on_slider); tb.on_submit(_on_box)

    def _apply_det(nm):
        def f(val):
            state[nm] = val
            G["gt_best"] = None         # metrics stale until the debounced recompute
            _light_text()               # thresholds update live; heavy recompute debounced
            _debounce(refresh)
        return f

    # Detection gates: two columns of slider+box pairs (pixel gates left; component
    # / temporal gates right). Every change debounces one out-of-process recompute.
    y0, dy = 0.44, 0.065
    colA, colB, w2 = 0.10, 0.60, 0.24
    me_hi = max(2, int(G["D"]["gc"].max()) if G["D"]["gc"].size else 30)
    _add_bar("min_events", 1, me_hi, 1, y0 - 0 * dy, _apply_det("min_events"), colA, w2)
    _add_bar("tau_sf", 0.0, 1.0, None, y0 - 1 * dy, _apply_det("tau_sf"), colA, w2)
    _add_bar("tau_comb", 0.0, 30.0, None, y0 - 2 * dy, _apply_det("tau_comb"), colA, w2)
    _add_bar("min_rotor_pixels", 1, 200, 1, y0 - 3 * dy,
             _apply_det("min_rotor_pixels"), colA, w2)
    _add_bar("min_fill_ratio", 0.0, 1.0, None, y0 - 4 * dy,
             _apply_det("min_fill_ratio"), colA, w2)
    _add_bar("min_box_area", 1, 2000, 1, y0 - 0 * dy, _apply_det("min_box_area"), colB, w2)
    _add_bar("alpha", 0.0, 5.0, None, y0 - 1 * dy, _apply_det("alpha"), colB, w2)
    _add_bar("min_snr_sum", 0.0, 500.0, None, y0 - 2 * dy,
             _apply_det("min_snr_sum"), colB, w2)
    _add_bar("persist_m", 1, 10, 1, y0 - 3 * dy, _apply_det("persist_m"), colB, w2)
    _add_bar("persist_k", 1, 10, 1, y0 - 4 * dy, _apply_det("persist_k"), colB, w2)

    # Match threshold: only re-splits the cached match scores — cheap, instant.
    # (center: dist in GT half-diagonals, matched if <=; iou/iod: matched if >=.)
    def _apply_match_thr(val):
        state["match_thr"] = float(val)
        _thr_memory[state["match"]] = float(val)
        _light_text()
    _add_bar("match_thr", 0.0, 3.0, None, 0.575, _apply_match_thr)

    # Tracking sigma [px]: reach of the Gaussian gate relaxation around the previous
    # frame's detections. Only triggers the heavy pass while tracking is ON.
    def _apply_sigma(val):
        state["track_sigma"] = float(val)
        if state["tracking"]:
            _debounce(refresh)
        else:
            _light_text()
    _add_bar("track_sigma", 1.0, 100.0, None, 0.64, _apply_sigma)

    # frame position slider + a frame-INDEX value box.
    ax_fr = figc.add_axes([0.14, 0.51, 0.55, 0.035])
    s_frame = Slider(ax_fr, "frame", 0.0, 1.0, valinit=state["frame_pos"])
    s_frame.valtext.set_visible(False)
    s_frame.drawon = False                            # throttle repaints the knob
    ax_fb = figc.add_axes([0.71, 0.51, 0.09, 0.04])

    def _cur_frame_idx():
        D = G["D"]
        if D["uniq_frames"].size == 0:
            return 0
        i = int(round(state["frame_pos"] * (D["uniq_frames"].size - 1)))
        return int(D["uniq_frames"][max(0, min(i, D["uniq_frames"].size - 1))])

    tb_frame = TextBox(ax_fb, "", initial=str(_cur_frame_idx()))

    def _frame_slider(v):
        state["frame_pos"] = float(v)
        tb_frame.text_disp.set_text(str(_cur_frame_idx()))
        _light_overlay()                              # cheap: reuse cached stats

    def _frame_box(text):
        D = G["D"]
        try:
            fidx = int(float(text))
        except ValueError:
            return
        if D["uniq_frames"].size == 0:
            return
        pos = int(np.searchsorted(D["uniq_frames"], fidx))
        pos = max(0, min(pos, D["uniq_frames"].size - 1))
        state["frame_pos"] = pos / max(1, D["uniq_frames"].size - 1)
        s_frame.eventson = False; s_frame.set_val(state["frame_pos"]); s_frame.eventson = True
        _light_overlay()

    s_frame.on_changed(_frame_slider); tb_frame.on_submit(_frame_box)

    # ── case + mode switching ────────────────────────────────────────────────────
    def _load(name, want_mode=None):
        """Load (case, mode) from the CSVs. Only modes that exist for the case are
        offered; if the requested/current mode isn't available for `name`, fall
        back to that case's first available mode."""
        avail = _available_modes(name, data_dir) or G["modes"]
        m = want_mode if want_mode in avail else (G["mode"] if G["mode"] in avail else avail[0])
        try:
            G["D"] = load_D(name, m, data_dir)
        except Exception as exc:
            txt.set_text(f"could not load {name} / {m}:\n{exc}"); figc.canvas.draw_idle(); return
        G["seq"] = name; G["mode"] = m; G["modes"] = avail
        G["cases"] = _available_cases(m, data_dir) or [name]
        G["idx"] = G["cases"].index(name) if name in G["cases"] else 0
        b_mode.label.set_text(f"mode: {m}")
        mx = max(2, int(G["D"]["gc"].max()) if G["D"]["gc"].size else 30)
        sliders["min_events"].valmax = mx; sliders["min_events"].ax.set_xlim(1, mx)
        state["frame_pos"] = _default_frame(G["D"])
        s_frame.eventson = False; s_frame.set_val(state["frame_pos"]); s_frame.eventson = True
        tb_frame.text_disp.set_text(str(_cur_frame_idx()))
        refresh()                                     # heavy: full recompute for the new case/mode

    def step_case(d):
        if G["cases"]:
            G["idx"] = (G["idx"] + d) % len(G["cases"]); _load(G["cases"][G["idx"]])

    def cycle_mode():
        ms = G["modes"]
        if len(ms) > 1:                               # only the modes this case has
            _load(G["seq"], ms[(ms.index(G["mode"]) + 1) % len(ms)])

    # tracking toggle: Gaussian gate relaxation around the previous frame's dets.
    ax_track = figc.add_axes([0.62, 0.035, 0.16, 0.06])
    b_track = Button(ax_track, "tracking: ON" if state["tracking"] else "tracking: OFF")

    def _toggle_track(_e=None):
        state["tracking"] = not state["tracking"]
        b_track.label.set_text("tracking: ON" if state["tracking"] else "tracking: OFF")
        refresh()
    b_track.on_clicked(_toggle_track)

    # match-mode cycle (center / iou / iod); remembers the thr you used per mode.
    ax_match = figc.add_axes([0.80, 0.035, 0.16, 0.06])
    b_match = Button(ax_match, f"match: {state['match']}")

    def _cycle_match(_e=None):
        _thr_memory[state["match"]] = state["match_thr"]
        nxt = MATCH_MODES[(MATCH_MODES.index(state["match"]) + 1) % len(MATCH_MODES)]
        state["match"] = nxt
        state["match_thr"] = _thr_memory.get(nxt, MATCH_DEFAULT_THR[nxt])
        b_match.label.set_text(f"match: {nxt}")
        sl = sliders["match_thr"]
        sl.eventson = False; sl.set_val(state["match_thr"]); sl.eventson = True
        _sync_box("match_thr", state["match_thr"])
        if G["dets"] is not None:                     # detections unchanged: rescore only
            G["gt_best"], G["det_best"], G["gtf_best"] = _score_caches(
                G["D"], G["dets"], nxt)
        _light_text()
    b_match.on_clicked(_cycle_match)

    ax_mode = figc.add_axes([0.04, 0.035, 0.13, 0.06]); b_mode = Button(ax_mode, f"mode: {G['mode']}")
    b_mode.on_clicked(lambda _e: cycle_mode())
    ax_prev = figc.add_axes([0.19, 0.035, 0.09, 0.06]); b_prev = Button(ax_prev, "◀ Prev")
    ax_next = figc.add_axes([0.29, 0.035, 0.09, 0.06]); b_next = Button(ax_next, "Next ▶")
    b_prev.on_clicked(lambda _e: step_case(-1)); b_next.on_clicked(lambda _e: step_case(+1))
    ax_jump = figc.add_axes([0.46, 0.035, 0.13, 0.06])
    tb = TextBox(ax_jump, "case: ", initial=str(seq))
    tb.on_submit(lambda t: _load(t.strip()) if t.strip() else None)

    # test/debug hooks: poll the worker + fire the debounce without an event loop
    G["_poll"] = _apply_job
    G["_fire"] = _fire_pending
    G["_state"] = state
    G["_fig"] = fig; G["_figc"] = figc

    refresh()                                         # initial compute (worker follows)
    plt.show()
    return G


def launch_gui_tk(seq, mode, data_dir, p0, match="center", match_thr=None,
                  tracking=False, track_sigma=20.0):
    """Single-window GUI with NATIVE Tk controls.

    matplotlib widgets redraw their whole figure on every press/motion — that is
    what kept freezing the sliders no matter how light the callbacks became.
    Here the sliders/entries/buttons are native Tk widgets (interacting with them
    costs nothing), and the one embedded matplotlib canvas (stage bars + overlay)
    is redrawn ONLY when a worker-process result arrives or the frame is
    scrubbed — never by pressing a control."""
    import tkinter as tk
    from tkinter import ttk
    import threading
    import multiprocessing as _mp
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Rectangle

    modes = _available_modes(seq, data_dir) or [mode]
    if mode not in modes:
        mode = modes[0]
    G = {"seq": seq, "mode": mode, "modes": modes,
         "cases": _available_cases(mode, data_dir) or [seq],
         "D": load_D(seq, mode, data_dir),
         "res": None, "dets": None, "comps": None,
         "view": "frame", "wscale": 1,
         "gt_best": None, "det_best": None, "gtf_best": None}
    G["idx"] = G["cases"].index(seq) if seq in G["cases"] else 0

    state = dict(p0)
    state.setdefault("min_fill_ratio", 0.0)
    state.setdefault("min_snr_sum", 0.0)
    state.setdefault("min_snr_avg", 0.0)
    state.setdefault("persist_m", 1); state.setdefault("persist_k", 1)
    state["match"] = match if match in MATCH_MODES else "center"
    state["match_thr"] = (float(match_thr) if match_thr is not None
                          else MATCH_DEFAULT_THR[state["match"]])
    state["tracking"] = bool(tracking); state["track_sigma"] = float(track_sigma)
    state["frame_pos"] = 0.0
    state.setdefault("window_x", 1)
    state.setdefault("track_coast", 0)               # 0 = off (prev-frame only)
    state.setdefault("track_confirm", 1)             # 1 = off (no confirmation)
    state.setdefault("track_acquire", 1.0)           # 1 = off (no acquire tighten)
    state.setdefault("track_predict", False)         # velocity extrapolation
    thr_memory = dict(MATCH_DEFAULT_THR); thr_memory[state["match"]] = state["match_thr"]
    INT_P = ("min_events", "min_rotor_pixels", "min_box_area", "persist_m",
             "persist_k", "window_x", "track_coast", "track_confirm")

    def _default_frame(D):
        if D["uniq_frames"].size == 0:
            return 0.0
        ig = D["ig"].astype(bool)
        if ig.any():
            pos = int(np.searchsorted(D["uniq_frames"],
                                      int(np.argmax(np.bincount(D["fr"][ig])))))
            return min(pos, D["uniq_frames"].size - 1) / max(1, D["uniq_frames"].size - 1)
        return 0.5
    state["frame_pos"] = _default_frame(G["D"])

    root = tk.Tk()
    root.title("frame explorer")
    hidden = bool(os.environ.get("FE_HIDDEN"))
    if hidden:
        root.withdraw()

    # ── embedded plots (redrawn only on results / frame scrub) ────────────────
    fig = Figure(figsize=(9.8, 7.2), dpi=100)
    ax_in = fig.add_axes([0.07, 0.58, 0.40, 0.36])
    ax_out = fig.add_axes([0.57, 0.58, 0.40, 0.36])
    ax_ov = fig.add_axes([0.22, 0.04, 0.56, 0.48])
    ax_ov.set_xticks([]); ax_ov.set_yticks([])
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ctrl = ttk.Frame(root, padding=(10, 8)); ctrl.pack(side=tk.RIGHT, fill=tk.Y)

    # LOADING banner over the plots (a native Tk label — showing/hiding it costs
    # nothing) while a change's result is still being computed/rendered.
    _loading = tk.Label(canvas.get_tk_widget(), text="  LOADING …  ",
                        font=("Segoe UI", 11, "bold"), fg="white", bg="#d97706")

    def _show_loading():
        _loading.place(relx=0.5, rely=0.02, anchor="n")
        _loading.lift()

    def _hide_loading():
        _loading.place_forget()

    # Info/metrics text sits at the BOTTOM (packed side=bottom before everything
    # else), so when its length changes it grows into the slack space instead of
    # shifting the sliders above it.
    info_var = tk.StringVar(value="computing …")
    tk.Label(ctrl, textvariable=info_var, font=("Consolas", 9), justify="left",
             anchor="sw").pack(side="bottom", fill="x", pady=(8, 0))

    def draw_region(ax, c, title):
        ax.clear()
        ORDER = EX.stage_order(G["mode"])
        tot = int(c.sum())
        xs = np.arange(len(ORDER))
        vals = [c[code] / tot * 100 if tot else 0 for code in ORDER]
        ax.bar(xs, vals, color=[EX.STAGE_HEX[code] for code in ORDER])
        for xi, code, v in zip(xs, ORDER, vals):
            ax.text(xi, v + 1, f"{int(c[code])}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([EX.STAGE_LABEL[code].replace("fail ", "x") for code in ORDER],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 105); ax.set_title(f"{title}  ({tot} px)", fontsize=10)

    def _comp_inside(C):
        """Per pre-gate component: does it match ANY GT under the current match
        rule + threshold? Vectorised searchsorted pairing per frame."""
        m = state["match"]; thr = state["match_thr"]
        n = int(C["frame"].size)
        best = np.full(n, _worst(m), float)
        gr = G["D"]["gt_rects"]
        if gr.size and n:
            of = np.argsort(C["frame"], kind="stable")
            cf = C["frame"][of]
            R = np.column_stack([C["x1"], C["y1"], C["x2"], C["y2"]]).astype(float)[of]
            gf = gr[:, 0].astype(np.int64)
            lo = np.searchsorted(cf, gf, "left"); hi = np.searchsorted(cf, gf, "right")
            cnt = hi - lo
            gi = np.repeat(np.arange(gf.size), cnt)
            if gi.size:
                dpos = (np.arange(int(cnt.sum()))
                        - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt))
                G4 = gr[gi, 1:5].astype(float); D4 = R[dpos]
                if m == "center":
                    gcx = (G4[:, 0] + G4[:, 2]) / 2; gcy = (G4[:, 1] + G4[:, 3]) / 2
                    dcx = (D4[:, 0] + D4[:, 2]) / 2; dcy = (D4[:, 1] + D4[:, 3]) / 2
                    hd = 0.5 * np.hypot(G4[:, 2] - G4[:, 0], G4[:, 3] - G4[:, 1])
                    sc = np.hypot(dcx - gcx, dcy - gcy) / np.maximum(hd, 1e-9)
                    bs = np.full(n, _worst(m), float)
                    np.minimum.at(bs, dpos, sc)
                else:
                    ix1 = np.maximum(G4[:, 0], D4[:, 0]); iy1 = np.maximum(G4[:, 1], D4[:, 1])
                    ix2 = np.minimum(G4[:, 2], D4[:, 2]); iy2 = np.minimum(G4[:, 3], D4[:, 3])
                    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                    da = np.clip(D4[:, 2] - D4[:, 0], 0, None) * np.clip(D4[:, 3] - D4[:, 1], 0, None)
                    if m == "iou":
                        ga = np.clip(G4[:, 2] - G4[:, 0], 0, None) * np.clip(G4[:, 3] - G4[:, 1], 0, None)
                        sc = inter / np.maximum(ga + da - inter, 1e-9)
                    else:
                        sc = np.where(da > 0, inter / np.maximum(da, 1e-9), 0.0)
                    bs = np.full(n, _worst(m), float)
                    np.maximum.at(bs, dpos, sc)
                best[of] = bs
        return (best <= thr) if m == "center" else (best >= thr)

    def _draw_boxes_bars():
        """Component-level funnel: every candidate component (pre box-gates),
        split INSIDE/OUTSIDE GT by the match setting, counted by the first box
        gate that would reject it under the current thresholds."""
        ax_ov.clear()
        ax_ov.set_aspect("auto")                      # undo imshow's equal aspect
        C = G.get("comps")
        if C is None or C["frame"].size == 0:
            ax_ov.set_xticks([]); ax_ov.set_yticks([])
            ax_ov.set_title("boxes view: no components", fontsize=10); return
        npx = C["npx"].astype(np.int64)
        area = (C["x2"] - C["x1"] + 1) * (C["y2"] - C["y1"] + 1)
        fillr = npx / np.maximum(1, area)
        snrs = C["snr_sum"]
        rel = C.get("relax")                          # tracking relaxation per comp
        relf = 1.0 - rel if rel is not None else 1.0  # same scaling the detector uses
        stagec = np.full(npx.size, 5, np.int8)        # PASS
        ok = np.ones(npx.size, bool)
        m0 = ok & (npx < state["min_rotor_pixels"] * relf);  stagec[m0] = 0; ok &= ~m0
        m1 = ok & (area < state["min_box_area"] * relf);     stagec[m1] = 1; ok &= ~m1
        m2 = ok & (fillr < state["min_fill_ratio"]);         stagec[m2] = 2; ok &= ~m2
        m3 = ok & (snrs < state["min_snr_sum"] * relf);      stagec[m3] = 3; ok &= ~m3
        m4 = ok & (snrs < state["min_snr_avg"] * np.maximum(1, npx) * relf)
        stagec[m4] = 4
        inside = _comp_inside(C)
        labels = ["x min_rotor_px", "x min_box_area", "x min_fill", "x snr_sum",
                  "x snr_avg", "PASS"]
        xs = np.arange(6)
        for off, mask, color, name in ((-0.19, inside, "#00a0a0", "inside GT"),
                                       (+0.19, ~inside, "0.55", "outside GT")):
            cnt = np.bincount(stagec[mask], minlength=6)
            tot = max(1, int(cnt.sum()))
            ax_ov.bar(xs + off, cnt / tot * 100, width=0.36, color=color,
                      label=f"{name} ({int(cnt.sum())})")
            for xi, v, cn in zip(xs + off, cnt / tot * 100, cnt):
                ax_ov.text(xi, v + 1, str(int(cn)), ha="center", fontsize=7)
        ax_ov.set_xticks(xs)
        ax_ov.set_xticklabels(labels, fontsize=8)
        ax_ov.set_yticks([0, 25, 50, 75, 100])
        ax_ov.set_ylim(0, 108)
        ax_ov.legend(fontsize=8, loc="upper center")
        trk = "  (track-relaxed gates)" if state["tracking"] else ""
        ax_ov.set_title(f"components by box gate, split by match={state['match']}"
                        f"{trk}", fontsize=10)

    def _draw_overlay():
        if G.get("view") == "boxes":
            _draw_boxes_bars(); return
        D = G["D"]; ax_ov.clear(); ax_ov.set_xticks([]); ax_ov.set_yticks([])
        r = G.get("res")
        if r is None or D["uniq_frames"].size == 0:
            ax_ov.set_title("computing …", fontsize=10); return
        stage = r["stage"]
        i = int(round(state["frame_pos"] * (D["uniq_frames"].size - 1)))
        i = max(0, min(i, D["uniq_frames"].size - 1))
        f = int(D["uniq_frames"][i])
        s = int(D["frame_starts"][i]); e = s + int(D["frame_counts"][i])
        bg = _rgb_frame(G["seq"], f, D["H"], D["W"])
        cnv = (bg.copy() if bg is not None
               else np.full((D["H"], D["W"], 3), 18, np.uint8))
        xs = D["x"][s:e].astype(np.int64); ys = D["y"][s:e].astype(np.int64)
        st = stage[s:e]
        for code in (EX.F_MINEV, EX.F_SF, EX.F_SNR, EX.F_MINPX, EX.F_BOX, EX.PASS):
            m = st == code
            if m.any():
                cnv[ys[m], xs[m]] = EX.STAGE_RGB[code]
        ax_ov.imshow(cnv, origin="upper", interpolation="nearest")
        gr = D["gt_rects"]
        gts = [tuple(int(v) for v in row[1:5]) for row in gr[gr[:, 0] == f]] if gr.size else []
        for (x1, y1, x2, y2) in gts:
            ax_ov.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="cyan", lw=1.5))
        det_rects = list((G["dets"] or {}).get(f, []))
        extra = max(0, len(det_rects) - 300)         # patch cap: huge det counts
        for (x1, y1, x2, y2) in det_rects[:300]:     # would freeze the canvas draw
            ax_ov.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="lime", lw=1.5))
        vet = list((r.get("vetoed") or {}).get(f, []))   # removed by the veto
        for (x1, y1, x2, y2) in vet[:300]:
            ax_ov.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="orange", lw=1.5, ls="--"))
        vlab = f", {len(vet)} vetoed (orange)" if vet else ""
        mtch = state["match"]
        best = _best_of((_match_score(g, d, mtch) for g in gts for d in det_rects), mtch)
        blab = (f"best {mtch}={best:.2f}" if (gts and det_rects and np.isfinite(best))
                else "no match")
        more = f" (+{extra} not drawn)" if extra else ""
        ax_ov.set_title(f"{G['seq']}  frame {f}: {len(gts)} GT (cyan), "
                        f"{len(det_rects)} det (lime){more}{vlab}, {blab}", fontsize=10)

    def _metrics_text():
        gb = G.get("gt_best")
        if gb is None:
            return "metrics: computing …"
        thr = state["match_thr"]; mtch = state["match"]
        db = G["det_best"]; fb = G["gtf_best"]
        gp = (gb <= thr) if mtch == "center" else (gb >= thr)
        dp = (db <= thr) if mtch == "center" else (db >= thr)
        fh = (fb <= thr) if mtch == "center" else (fb >= thr)
        tp = int(gp.sum()); fn = gb.size - tp; fp = int((~dp).sum())
        nf = max(1, G["D"]["n_frames"])
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        return (f"match={mtch} thr={thr:.2f}\n"
                f"TP={tp}  FP={fp}  FN={fn}  FP/frame={fp / nf:.3f}\n"
                f"precision={prec:.3f}  recall={rec:.3f}\n"
                f"GT frames hit {int(fh.sum())} / {fb.size}"
                f"  ({int(fh.sum()) / max(1, fb.size):.1%})")

    def _update_info():
        D = G["D"]
        info_var.set(
            f"case {G['seq']} ({G['idx'] + 1}/{len(G['cases'])})  mode {G['mode']}\n"
            f"[avail: {', '.join(G['modes'])}]\n"
            f"frames={D['n_frames']}  GT frames={D['n_gt']}\n"
            f"candidate px={D['gc'].size}\n"
            f"tracking {'ON s=%.0f coast=%d conf=%d acq=%.1f%s' % (state['track_sigma'], state['track_coast'], state['track_confirm'], state['track_acquire'], ' pred' if state['track_predict'] else '') if state['tracking'] else 'off'}"
            f"   persist {state['persist_m']}/{state['persist_k']}"
            f"   window x{G.get('wscale', 1)}\n\n"
            + _metrics_text())

    # ── worker process (same protocol as before) ──────────────────────────────
    _job = {"gen": 0, "out": None}
    _worker = {"proc": None, "conn": None}

    def _ensure_worker():
        if _worker["proc"] is not None and _worker["proc"].is_alive():
            return _worker["conn"]
        try:
            parent, child = _mp.Pipe()
            pr = _mp.Process(target=_metrics_worker, args=(child,), daemon=True)
            pr.start()
            _worker["proc"], _worker["conn"] = pr, parent
            return parent
        except Exception:
            return None

    def _start_job():
        _job["gen"] += 1
        gen = _job["gen"]; params = dict(state)
        conn = _ensure_worker()
        if conn is not None:
            try:
                conn.send(dict(gen=gen, seq=G["seq"], mode=G["mode"],
                               data_dir=data_dir, params=params,
                               wscale=G.get("wscale", 1)))
                return
            except Exception:
                _worker["proc"] = _worker["conn"] = None
        D = G["D"]; mode_ = G["mode"]

        def work():
            try:
                _job["out"] = (gen,) + _compute_job(D, mode_, params)
            except Exception as exc:
                _job["out"] = (gen, "error", repr(exc), None, None)
        threading.Thread(target=work, daemon=True).start()

    def refresh():
        """Fire the job immediately (no debounce): sending is ~1 ms, the worker
        drains its queue to the newest request, and results land via _poll — so
        every slider tick updates the bars/stats as fast as they can compute.
        Until the new result arrives the previous numbers stay, with a LOADING
        banner over the plots to say they're outdated."""
        _show_loading()
        _start_job()

    _quiet = {"on": False}

    def _apply(out):
        _gen, res, dets, caches, comps = out
        if isinstance(res, str):
            info_var.set(f"metrics failed: {dets}")
            _hide_loading()
            return
        G["res"] = res; G["dets"] = dets; G["comps"] = comps
        G["gt_best"], G["det_best"], G["gtf_best"] = caches
        draw_region(ax_in, res["in"], "INSIDE GT")
        draw_region(ax_out, res["out"], "OUTSIDE GT")
        _draw_overlay()
        canvas.draw_idle()
        _update_info()
        _hide_loading()

    def _poll(once=False):
        out = None
        conn = _worker["conn"]
        if conn is not None:
            try:
                while conn.poll():
                    out = conn.recv()
            except (EOFError, OSError):
                _worker["proc"] = _worker["conn"] = None
        if out is None:
            out = _job.get("out"); _job["out"] = None
        if out is not None and out[0] == _job["gen"]:
            _apply(out)                               # apply at once, even mid-drag
        if not once:
            root.after(30, lambda: _poll())

    # ── native controls ────────────────────────────────────────────────────────
    scales, entries = {}, {}

    def _fmt(name, v):
        return str(int(round(v))) if name in INT_P else f"{float(v):.4g}"

    def _sync_entry(name):
        e = entries[name]
        e.delete(0, "end"); e.insert(0, _fmt(name, state[name]))

    _ov_redraw = {"after": None}

    def _sched_overlay():
        if _ov_redraw["after"] is None:               # coalesce scrub redraws
            _ov_redraw["after"] = root.after(
                60, lambda: (_ov_redraw.__setitem__("after", None),
                             _draw_overlay(), canvas.draw_idle()))

    def _on_change(name):
        if _quiet["on"]:
            return
        v = scales[name].get()
        v = int(v) if name in INT_P else float(v)
        if name == "frame":
            n = max(1, G["D"]["uniq_frames"].size)
            state["frame_pos"] = v / max(1, n - 1)
            entries["frame"].delete(0, "end")
            i = max(0, min(int(v), n - 1))
            entries["frame"].insert(0, str(int(G["D"]["uniq_frames"][i]) if n else 0))
            _sched_overlay()
            return
        if state.get(name) == v:
            return
        state[name] = v
        _sync_entry(name)
        if name == "window_x":                        # emulated k× DDHF window
            k = int(v)
            if k == G.get("wscale", 1):
                return
            _show_loading(); root.update_idletasks()  # aggregation blocks briefly
            try:
                G["D"] = load_D(G["seq"], G["mode"], data_dir, k)
            except Exception as exc:
                info_var.set(f"window x{k} failed: {exc}")
                _hide_loading()
                return
            G["wscale"] = k
            G["res"] = None; G["dets"] = None; G["comps"] = None
            n = max(1, G["D"]["uniq_frames"].size)
            scales["frame"].config(to=n - 1)
            state["frame_pos"] = _default_frame(G["D"])
            _quiet["on"] = True
            scales["frame"].set(int(round(state["frame_pos"] * (n - 1))))
            _quiet["on"] = False
            refresh()
        elif name == "match_thr":
            thr_memory[state["match"]] = v
            _update_info()                            # instant re-split, no job
            if G.get("view") == "boxes":              # re-split the boxes view too
                _sched_overlay()
        elif name in ("track_sigma", "track_coast", "track_confirm",
                      "track_acquire"):
            if state["tracking"]:
                refresh()
        else:                                         # detection gate: immediate
            refresh()

    def _add_scale(name, lo, hi, res):
        row = ttk.Frame(ctrl); row.pack(fill="x", pady=1)
        ttk.Label(row, text=name, width=15).pack(side="left")
        sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal",
                      showvalue=0, length=170,
                      command=lambda _v, n=name: _on_change(n))
        sc.set(state.get(name, lo))
        sc.pack(side="left", fill="x", expand=True)
        e = ttk.Entry(row, width=7); e.pack(side="left", padx=(4, 0))
        scales[name], entries[name] = sc, e

        def _submit(_e=None, n=name):
            try:
                val = float(entries[n].get())
            except ValueError:
                if n == "frame":
                    _on_change("frame")               # rewrites the entry from the scale
                else:
                    _sync_entry(n)
                return
            if n == "frame":
                # the entry holds a FRAME ID — map it to the nearest frame that
                # actually has data (the scale itself is an index)
                uf = G["D"]["uniq_frames"]
                if uf.size:
                    idx = int(np.searchsorted(uf, int(round(val))))
                    sc.set(max(0, min(idx, uf.size - 1)))
                    _on_change("frame")
                return
            val = int(round(val)) if n in INT_P else val
            if val < sc.cget("from"):
                sc.config(from_=val)
            if val > sc.cget("to"):
                sc.config(to=val)
            sc.set(val)
            _on_change(n)                             # explicit: programmatic set may
        e.bind("<Return>", _submit)                   # not invoke the Tk command

    me_hi = max(2, int(G["D"]["gc"].max()) if G["D"]["gc"].size else 30)
    n0 = max(1, G["D"]["uniq_frames"].size)
    _add_scale("frame", 0, n0 - 1, 1)
    _add_scale("match_thr", 0.0, 3.0, 0.01)
    _add_scale("track_sigma", 1, 100, 1)
    _add_scale("track_coast", 0, 10, 1)              # anchor survives N misses (0=off)
    _add_scale("track_confirm", 1, 5, 1)             # hits before relaxing (1=off)
    _add_scale("track_acquire", 1.0, 5.0, 0.1)       # far-gate tightening ×A (1=off)
    _predict_var = tk.BooleanVar(value=bool(state["track_predict"]))

    def _toggle_predict():
        state["track_predict"] = bool(_predict_var.get())
        if state["tracking"]:
            refresh()
    ttk.Checkbutton(ctrl, text="velocity predict (tracking)",
                    variable=_predict_var,
                    command=_toggle_predict).pack(anchor="w", pady=(0, 2))
    _add_scale("window_x", 1, 6, 1)                  # emulated k× DDHF window
    ttk.Separator(ctrl).pack(fill="x", pady=4)
    _add_scale("min_events", 1, me_hi, 1)
    _add_scale("tau_sf", 0.0, 1.0, 0.01)
    _add_scale("tau_comb", 0.0, 30.0, 0.1)
    _add_scale("min_rotor_pixels", 1, 200, 1)
    _add_scale("min_fill_ratio", 0.0, 1.0, 0.01)
    _add_scale("min_box_area", 1, 2000, 1)
    _add_scale("alpha", 0.0, 5.0, 0.05)
    _add_scale("min_snr_sum", 0, 500, 1)
    _add_scale("min_snr_avg", 0.0, 30.0, 0.1)
    _add_scale("persist_m", 1, 10, 1)
    _add_scale("persist_k", 1, 10, 1)
    for nm in scales:
        if nm != "frame":
            _sync_entry(nm)

    # ── case / mode / tracking / match buttons ────────────────────────────────
    def _load_case(name, want_mode=None):
        avail = _available_modes(name, data_dir) or G["modes"]
        m = want_mode if want_mode in avail else (G["mode"] if G["mode"] in avail
                                                  else avail[0])
        try:
            _show_loading(); root.update_idletasks()
            G["D"] = load_D(name, m, data_dir, G.get("wscale", 1))
        except Exception as exc:
            info_var.set(f"could not load {name}/{m}:\n{exc}")
            _hide_loading()
            return
        G["seq"] = name; G["mode"] = m; G["modes"] = avail
        G["cases"] = _available_cases(m, data_dir) or [name]
        G["idx"] = G["cases"].index(name) if name in G["cases"] else 0
        b_mode.config(text=f"mode: {m}")
        case_var.set(name)
        me = max(2, int(G["D"]["gc"].max()) if G["D"]["gc"].size else 30)
        scales["min_events"].config(to=me)
        n = max(1, G["D"]["uniq_frames"].size)
        scales["frame"].config(to=n - 1)
        state["frame_pos"] = _default_frame(G["D"])
        _quiet["on"] = True
        scales["frame"].set(int(round(state["frame_pos"] * (n - 1))))
        _quiet["on"] = False
        G["res"] = None; G["dets"] = None
        refresh()

    def _step(d):
        if G["cases"]:
            G["idx"] = (G["idx"] + d) % len(G["cases"])
            _load_case(G["cases"][G["idx"]])

    def _cycle_mode():
        ms = G["modes"]
        if len(ms) > 1:
            _load_case(G["seq"], ms[(ms.index(G["mode"]) + 1) % len(ms)])

    def _toggle_track():
        state["tracking"] = not state["tracking"]
        b_track.config(text="tracking: ON" if state["tracking"] else "tracking: OFF")
        refresh()

    def _cycle_match():
        thr_memory[state["match"]] = state["match_thr"]
        nxt = MATCH_MODES[(MATCH_MODES.index(state["match"]) + 1) % len(MATCH_MODES)]
        state["match"] = nxt
        state["match_thr"] = thr_memory.get(nxt, MATCH_DEFAULT_THR[nxt])
        b_match.config(text=f"match: {nxt}")
        _quiet["on"] = True
        scales["match_thr"].set(state["match_thr"])
        _quiet["on"] = False
        _sync_entry("match_thr")
        if G["dets"] is not None:                     # rescore only, no recompute
            G["gt_best"], G["det_best"], G["gtf_best"] = _score_caches(
                G["D"], G["dets"], nxt)
        _update_info()
        if G.get("view") == "boxes":                  # re-split the boxes view
            _sched_overlay()

    def _toggle_view():
        G["view"] = "boxes" if G.get("view") == "frame" else "frame"
        b_view.config(text=f"view: {G['view']}")
        _draw_overlay(); canvas.draw_idle()

    ttk.Separator(ctrl).pack(fill="x", pady=4)
    row1 = ttk.Frame(ctrl); row1.pack(fill="x", pady=2)
    b_mode = ttk.Button(row1, text=f"mode: {G['mode']}", command=_cycle_mode)
    b_mode.pack(side="left", expand=True, fill="x")
    ttk.Button(row1, text="< Prev", width=7,
               command=lambda: _step(-1)).pack(side="left", padx=2)
    ttk.Button(row1, text="Next >", width=7,
               command=lambda: _step(+1)).pack(side="left")
    b_view = ttk.Button(row1, text="view: frame", width=11, command=_toggle_view)
    b_view.pack(side="left", padx=2)
    row2 = ttk.Frame(ctrl); row2.pack(fill="x", pady=2)
    ttk.Label(row2, text="case:").pack(side="left")
    case_var = tk.StringVar(value=str(seq))
    e_case = ttk.Entry(row2, textvariable=case_var, width=8)
    e_case.pack(side="left", padx=4)
    e_case.bind("<Return>", lambda _e: _load_case(case_var.get().strip()))
    b_track = ttk.Button(row2, text="tracking: ON" if state["tracking"] else "tracking: OFF",
                         command=_toggle_track)
    b_track.pack(side="left", padx=4, expand=True, fill="x")
    b_match = ttk.Button(row2, text=f"match: {state['match']}", command=_cycle_match)
    b_match.pack(side="left", expand=True, fill="x")

    # frame entry shows the actual frame index
    n = max(1, G["D"]["uniq_frames"].size)
    i0 = int(round(state["frame_pos"] * (n - 1)))
    entries["frame"].insert(0, str(int(G["D"]["uniq_frames"][i0]) if n else 0))
    _quiet["on"] = True
    scales["frame"].set(i0)
    _quiet["on"] = False

    _update_info()
    refresh()
    _poll()
    G["_poll"] = lambda: _poll(once=True)
    G["_root"] = root
    G["_state"] = state
    G["_fig"] = fig
    G["_set"] = lambda n, v: (scales[n].set(v), _on_change(n))   # test/debug hook
    if not hidden:
        root.mainloop()
    return G


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="Case name, e.g. 3.")
    ap.add_argument("--mode", choices=("baseline", "pool", "pixelize", "components"),
                    default="baseline")
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="Folder with frames_/gt_ CSVs (default lab/frame_data).")
    ap.add_argument("--report", action="store_true",
                    help="Headless: print IoU / TP-FP-FN at the given thresholds "
                         "(no GUI).")
    ap.add_argument("--per-frame-out", default=None,
                    help="--report: also write a per-frame IoU CSV to this path.")
    # threshold knobs (used by --report; the GUI has sliders)
    ap.add_argument("--min-events", type=int, default=16)
    ap.add_argument("--tau-sf", type=float, default=0.5)
    ap.add_argument("--tau-comb", type=float, default=6.0)
    ap.add_argument("--min-rotor-pixels", type=int, default=60)
    ap.add_argument("--min-fill-ratio", type=float, default=0.0,
                    help="Ratio analogue of min_rotor_pixels: keep a component only "
                         "if PASS-pixels / bbox-area >= this (0 = off).")
    ap.add_argument("--min-box-area", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--min-snr-sum", type=float, default=0.0,
                    help="Component gate: sum of the component's pixel comb-SNRs "
                         "must reach this (0 = off). Separates a few STRONG pixels "
                         "from a few noise pixels, which counts can't.")
    ap.add_argument("--min-snr-avg", type=float, default=0.0,
                    help="Component gate: MEAN pixel comb-SNR (sum/npx) must reach "
                         "this (0 = off) — size-independent strength, unlike "
                         "--min-snr-sum which favours large components.")
    ap.add_argument("--persist-m", type=int, default=1,
                    help="Temporal persistence: report a detection only if a raw "
                         "detection co-located in >= m of the last k frames "
                         "(m=1 = off).")
    ap.add_argument("--persist-k", type=int, default=1,
                    help="Temporal persistence window k (frames).")
    ap.add_argument("--match", choices=MATCH_MODES, default="center",
                    help="Detection-vs-GT matching: 'center' = det center within "
                         "thr × GT half-diagonal of the GT center (the codebase's "
                         "--match center rule; right for rotor-only detections), "
                         "'iou' = IoU >= thr, 'iod' = |det∩GT|/|det| >= thr. "
                         "Counting is per instance: 1 good det + 2 strays in one "
                         "frame = 1 TP + 2 FP. Default center.")
    ap.add_argument("--match-thr", type=float, default=None,
                    help="Match threshold (default: 1.0 center / 0.1 iou / 0.5 iod).")
    ap.add_argument("--iou-thr", type=float, default=0.1,
                    help="Legacy alias: used as the threshold when --match iou and "
                         "--match-thr is not given.")
    ap.add_argument("--track", action="store_true",
                    help="Tracking: Gaussian-relax the gates around the previous "
                         "frame's detections (see --track-sigma).")
    ap.add_argument("--track-sigma", type=float, default=20.0,
                    help="Sigma [px] of the tracking relaxation (default 20).")
    ap.add_argument("--track-coast", type=int, default=0,
                    help="Tracking: anchors survive N missed frames with decaying "
                         "weight (0 = off, previous-frame only).")
    ap.add_argument("--track-confirm", type=int, default=1,
                    help="Tracking: a track must be seen in N frames before it "
                         "relaxes the gates (1 = off).")
    ap.add_argument("--track-acquire", type=float, default=1.0,
                    help="Tracking: far from any track the gates are tightened xA "
                         "(strict acquisition, loose sustain; 1 = off).")
    ap.add_argument("--track-predict", action="store_true",
                    help="Tracking: extrapolate anchors with track velocity.")
    ap.add_argument("--veto-peak", type=float, default=0.0,
                    help="Post-detection veto: delete a box whose pooled raw-event "
                         "spectrum has a 60-250 Hz line at least this many times "
                         "its median, with a 2nd / 3rd harmonic >= --veto-harm "
                         "(0 = off; needs the case's raw events).")
    ap.add_argument("--veto-harm", type=float, default=20.0,
                    help="Post-detection veto: required height of the 2f / 3f "
                         "line (times the median).")
    ap.add_argument("--skip-start", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Ignore each case's frames BEFORE the first GT "
                         "sighting (noisy lead-in; DEFAULT ON — pass "
                         "--no-skip-start to count the whole recording).")
    args = ap.parse_args()

    thr = args.match_thr
    if thr is None:
        thr = args.iou_thr if args.match == "iou" else MATCH_DEFAULT_THR[args.match]

    p = dict(min_events=args.min_events, tau_sf=args.tau_sf, tau_comb=args.tau_comb,
             min_rotor_pixels=args.min_rotor_pixels, min_box_area=args.min_box_area,
             alpha=args.alpha, min_fill_ratio=args.min_fill_ratio,
             skip_start=args.skip_start,
             min_snr_sum=args.min_snr_sum, min_snr_avg=args.min_snr_avg,
             persist_m=args.persist_m, persist_k=args.persist_k,
             track_coast=args.track_coast, track_confirm=args.track_confirm,
             track_acquire=args.track_acquire, track_predict=args.track_predict,
             veto_peak=args.veto_peak, veto_harm=args.veto_harm)

    if args.report:
        D = load_D(args.seq, args.mode, args.data_dir)
        EX.print_report(D, p, args.mode)             # gate-stage breakdown (reused)
        r = metrics_report(D, p, args.mode, args.match, thr, args.per_frame_out,
                           args.min_fill_ratio, tracking=args.track,
                           track_sigma=args.track_sigma)
        tr = f"  [tracking sigma={args.track_sigma:.0f}]" if args.track else ""
        print(f"\n[match={r['match']} thr={r['thr']}]{tr}  "
              f"frames={r['frames']} gt_frames={r['gt_frames']}")
        print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']}  "
              f"precision={r['precision']:.3f} recall={r['recall']:.3f}  "
              f"FP/frame={r['fp_per_frame']:.3f}")
        print(f"  GT frames hit {r['gt_frames_hit']} / {r['gt_frames']}")
        if args.per_frame_out:
            print(f"  per-frame results -> {args.per_frame_out}")
        return

    # Interactive: single-window Tk GUI (native controls — instant sliders; the
    # matplotlib canvas redraws only when worker results arrive).
    launch_gui_tk(args.seq, args.mode, args.data_dir, p, match=args.match,
                  match_thr=thr, tracking=args.track, track_sigma=args.track_sigma)


if __name__ == "__main__":
    main()
