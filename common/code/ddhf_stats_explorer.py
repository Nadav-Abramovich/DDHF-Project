"""
ddhf_stats_explorer.py -- precompute a sequence's DDHF spectral stage ONCE, then
move the threshold parameters with sliders and watch the pass/fail stats update
live (inside-GT vs outside-GT), with no recomputation. Includes a per-frame
overlay (where the PASS/fail pixels land) and an in-GUI sequence picker.

Why this works
--------------
The expensive part of DDHF (grouping events + the NDFT -> spectral flatness +
comb-SNR) does NOT depend on the threshold parameters: min_events only gates
which groups are *reported*, and tau_sf / tau_comb / min_rotor_pixels /
min_box_area are applied *after* the spectrum. So we run the spectral stage once
(measuring every candidate) and store, per candidate pixel: group event-count,
component size (components mode), SF, comb-SNR, inside-GT flag, GT-box id, and
(frame, x, y). Every threshold then becomes an instant filter on those arrays.

Two parameter classes
----------------------
* STRUCTURAL (chosen at launch; define the precompute + its on-disk cache):
    --seq --mode --window-ms --pixel-block --drone and the comb/harmonic flags,
    plus --stride. Change any of these -> a new precompute (cached).
* INTERACTIVE (sliders; instant):
    baseline/pool/pixelize : min_events, tau_sf, tau_comb, min_rotor_pixels, min_box_area
    components             : min_events, min_rotor_pixels (comp size), tau_sf, tau_comb

Modes: baseline / pool / pixelize / components.

Usage:
    ..\\..\\common\\.venv312\\Scripts\\python.exe ddhf_stats_explorer.py \\
        --seq ../3 --mode baseline --stride 4
    # then drag the sliders / frame slider, or type a new seq in the box.

    # GPU spectral stage (per-pixel modes) for a faster precompute:
    ... --gpu

    # headless logic check (no GUI):
    ... --selftest
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
from scipy.ndimage import label as _label, find_objects as _find_objects, convolve as _convolve

import debug_viewer_pooled as dv
import ddhf_candidate_stats as cs
from ddhf_config import build_cfg, add_cfg_cli_args, cfg_overrides_from_args

# Stage codes (pipeline order; a candidate is recorded at the FIRST it fails).
PASS, F_MINEV, F_SF, F_SNR, F_BOX, F_MINPX = 0, 1, 2, 3, 4, 5
STAGE_LABEL = {
    PASS:    "PASS",
    F_MINEV: "fail min_events",
    F_SF:    "fail SF",
    F_SNR:   "fail SNR",
    F_BOX:   "fail box",
    F_MINPX: "fail min_rotor_px",
}
STAGE_HEX = {
    F_MINEV: "#9e9e9e", F_SF: "#ffb74d", F_SNR: "#e57373",
    F_BOX: "#ba68c8", F_MINPX: "#ba68c8", PASS: "#4caf50",
}
def _hex_rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
STAGE_RGB = {k: _hex_rgb(v) for k, v in STAGE_HEX.items()}

def stage_order(mode):
    # min_rotor_pixels (blob-size) shown as the FINAL gate before PASS. In the real
    # components detector it is a prefilter, but every gate is AND'd so the PASS set
    # is identical -- only the failure attribution order changes.
    return ([F_MINEV, F_SF, F_SNR, F_MINPX, PASS] if mode == "components"
            else [F_MINEV, F_SF, F_SNR, F_BOX, PASS])

BOX_CAP = 400_000          # if more than this many green px, skip the box gate (too loose)
PRECOMPUTE_MIN_EVENTS = 1  # spectral stage measures every candidate; min_events is a slider
PRECOMPUTE_START = 1       # first window computed; earlier ones are read past, not computed
_BOXES_PER_FRAME = 64      # gt-box id packing: key = frame * 64 + local_index
MODES = ("baseline", "pool", "pixelize", "components")


# ── per-frame spectral stage -> per-candidate-pixel arrays ───────────────────────

def _group_count_map(cmap, mode, pixel_block):
    """Per-pixel event count of the group the detector tested the pixel in:
    baseline = own count; pool = 3x3 neighbourhood sum; pixelize = BxB block sum."""
    if mode == "baseline":
        return cmap.astype(np.int32)
    if mode == "pool":
        return _convolve(cmap.astype(np.int32), np.ones((3, 3), np.int32), mode="constant")
    B = max(1, int(pixel_block))
    H, W = cmap.shape
    Hb = (H + B - 1) // B; Wb = (W + B - 1) // B
    pad = np.zeros((Hb * B, Wb * B), np.int32); pad[:H, :W] = cmap
    blocks = pad.reshape(Hb, B, Wb, B).sum(axis=(1, 3))
    return np.repeat(np.repeat(blocks, B, 0), B, 1)[:H, :W]


def _batch_spectral(pol_k, tn_k, starts, cfg):
    """Per-group (SF, comb-SNR) via the batch NDFT, on GPU when dv._USE_GPU and a
    GPU is available, else CPU. Same spectral path compute_pixel_stats uses."""
    K = cfg.frequency_bins
    if dv._USE_GPU and dv._gpu_available() and not dv._low_cut_bins(cfg):   # low cut: CPU only
        try:
            power_g = dv._ndft_power_batch_gpu(pol_k, tn_k, starts, K, k_step=1.0 / dv._os(cfg))
            sf_g = dv._spectral_flatness_batch_gpu(power_g[1:])
            snr_g, _ = dv._comb_snr_batch_gpu(power_g, cfg, return_omega=True)
            return dv._cp.asnumpy(sf_g), dv._cp.asnumpy(snr_g)
        except Exception:
            try: dv._cp.get_default_memory_pool().free_all_blocks()
            except Exception: pass
    power = dv._apply_low_cut(dv._ndft_power_batch(None, pol_k, tn_k, starts, K,
                                                   k_step=1.0 / dv._os(cfg)), cfg)
    sf = dv._spectral_flatness_batch(power[dv._sf_start(cfg):])
    snr, _ = dv._comb_snr_batch(power, cfg, return_omega=True)
    return sf, snr


def _gt_maps(gts, cmap, n, H, W):
    """(ig, gb, box_counts): inside-any-GT mask values, first-GT-box id per pixel,
    and per-GT-box candidate counts."""
    ig = cs._rects_mask(gts, H, W)
    idm = np.full((H, W), -1, dtype=np.int32)
    box_counts = []
    for li, rect in enumerate(gts):
        bm = cs._rects_mask([rect], H, W)
        key = n * _BOXES_PER_FRAME + li
        idm[bm & (idm < 0)] = key
        box_counts.append((key, int((bm & (cmap > 0)).sum())))
    return ig, idm, box_counts


def _centers(rects):
    return [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in rects]


def _dist_prev(xs, ys, prev_centers):
    """Per-candidate distance (px) to the NEAREST previous-frame GT centre (teacher-
    forcing). Large sentinel where there was no previous GT (untracked -> g~0)."""
    if not prev_centers:
        return np.full(xs.size, 1e6, dtype=np.float32)
    xf = xs.astype(np.float64); yf = ys.astype(np.float64)
    d = None
    for (cx, cy) in prev_centers:
        dd = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
        d = dd if d is None else np.minimum(d, dd)
    return d.astype(np.float32)


def _frame_candidates(n, gts, evs, cfg, mode, H, W, prev_centers):
    """Per-candidate-pixel arrays for one window.
    cols = (fr,x,y,gc,size,sf,snr,ig,gb,dprev)."""
    if mode == "components":
        return _frame_candidates_components(n, gts, evs, cfg, H, W, prev_centers)
    cmap, sfm, snrm, _ = dv.compute_pixel_stats(evs, H, W, cfg)   # cfg.min_events == 1
    if not cmap.any():
        return None
    gcm = _group_count_map(cmap, mode, cfg.pixel_block)
    ys, xs = np.nonzero(cmap > 0)
    ig, idm, box_counts = _gt_maps(gts, cmap, n, H, W)
    return _pack(n, xs, ys, gcm[ys, xs], np.zeros(xs.size, np.int32),
                 sfm[ys, xs], snrm[ys, xs], ig[ys, xs], idm[ys, xs],
                 _dist_prev(xs, ys, prev_centers), box_counts,
                 int(((cmap > 0) & ~ig).sum()), bool(gts))


def _frame_candidates_components(n, gts, evs, cfg, H, W, prev_centers):
    if evs.size == 0:
        return None
    x = evs["x"].astype(np.int64); y = evs["y"].astype(np.int64)
    pol = dv._normalise_polarity(evs["p"])
    tn  = dv._normalise_timestamps(evs["t"].astype(np.float64),
                                   window_span=cfg.window_ms * 1000.0)
    count = np.bincount(y * W + x, minlength=H * W).reshape(H, W)
    labels, ncomp = dv._connected_components(count > 0)
    if ncomp == 0:
        return None
    ev_label = labels[y, x]
    order = np.argsort(ev_label, kind="stable")
    lab_s = ev_label[order]; pol_s = pol[order]; tn_s = tn[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(lab_s)) + 1))
    sf_grp, snr_grp = _batch_spectral(pol_s, tn_s, starts, cfg)
    comp_ids = lab_s[starts]
    ev_grp = np.diff(np.append(starts, lab_s.size)).astype(np.int32)
    comp_sizes = np.bincount(labels.ravel(), minlength=ncomp + 1).astype(np.int32)
    sfc = np.full(ncomp + 1, np.nan, np.float32); snrc = np.full(ncomp + 1, np.nan, np.float32)
    evc = np.zeros(ncomp + 1, np.int32)
    sfc[comp_ids] = sf_grp; snrc[comp_ids] = snr_grp; evc[comp_ids] = ev_grp

    ys, xs = np.nonzero(labels > 0)
    lab_px = labels[ys, xs]
    cmap = count
    ig, idm, box_counts = _gt_maps(gts, cmap, n, H, W)
    return _pack(n, xs, ys, evc[lab_px], comp_sizes[lab_px], sfc[lab_px], snrc[lab_px],
                 ig[ys, xs], idm[ys, xs], _dist_prev(xs, ys, prev_centers),
                 box_counts, int(((cmap > 0) & ~ig).sum()), bool(gts))


def _pack(n, xs, ys, gc, size, sf, snr, ig, gb, dprev, box_counts, out_per_frame, has_gt):
    return dict(
        cols=(np.full(xs.size, n, np.int32), xs.astype(np.int16), ys.astype(np.int16),
              gc.astype(np.int32), size.astype(np.int32), sf.astype(np.float32),
              snr.astype(np.float32), ig.astype(np.uint8), gb.astype(np.int32),
              dprev.astype(np.float32)),
        box_counts=box_counts, out_per_frame=out_per_frame, has_gt=has_gt)


# ── precompute (parallel: one reader -> bounded queue -> compute workers) ─────────

_N_COLS = 10
_COL_DT = (np.int32, np.int16, np.int16, np.int32, np.int32, np.float32, np.float32,
           np.uint8, np.int32, np.float32)


def _empty_cols():
    return [np.array([], dt) for dt in _COL_DT]


def _pre_consumer(task_q, result_q, ctx, mode):
    """Stateless worker: computes one window at a time and streams the result
    straight back. The parent does all accumulation (and checkpointing), so a
    worker's memory stays flat no matter how dense the case is."""
    cfg, gt_rects_at = cs._apply_ctx(ctx)
    H, W = ctx["H"], ctx["W"]
    try:
        while True:
            item = task_q.get()
            if item is None:
                break
            n, gts, evs = item
            prev_centers = _centers(gt_rects_at(n - 1))   # teacher-forced track origin
            fc = _frame_candidates(n, gts, evs, cfg, mode, H, W, prev_centers)
            result_q.put(("ok", n, gts, fc))
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            result_q.put(("error", repr(e), None, None))
        except Exception:
            pass                                     # parent still sees our death


_CKPT_FLUSH = 500                                    # windows per partial file


class _Ckpt:
    """Window-level checkpointing + progress for the precompute. The parent
    feeds every computed window through add(); each _CKPT_FLUSH windows the
    accumulated part is written to <base>.partNNN.npz, so a killed build
    (crash, watchdog, Ctrl-C) resumes from the last flush instead of window 1.
    Partials carry the structural cache key (stale ones are discarded) and are
    deleted once the final features file is saved."""

    def __init__(self, base, key, n_todo):
        self.base, self.key, self.n_todo = base, key, n_todo
        self.done, self.parts, self.next_i = set(), [], 0
        self._reset_cur()
        if not base:
            return
        import glob as _g
        for f in sorted(_g.glob(base + ".part*.npz")):
            try:
                z = np.load(f, allow_pickle=False)
                if str(z["key"]) != str(key):
                    raise ValueError("stale checkpoint key")
                part = dict(
                    cols=[z[f"c{i}"] for i in range(_N_COLS)],
                    box_counts=list(zip(z["box_keys"].tolist(),
                                        z["box_cand"].tolist())),
                    out_pf=z["out_pf"].tolist(),
                    gt_rects=[tuple(int(v) for v in r) for r in z["gt_rects"]],
                    n_frames=int(z["n_frames"]), n_gt=int(z["n_gt"]))
                wins = [int(w) for w in z["wins"]]
            except Exception:
                try:
                    os.remove(f)
                except OSError:
                    pass
                continue
            self.parts.append(part)
            self.done.update(wins)
            self.next_i = max(self.next_i,
                              int(f.rsplit(".part", 1)[1].split(".")[0]) + 1)
        if self.done:
            print(f"  [resume] {len(self.done)}/{self.n_todo} windows already "
                  f"computed ({len(self.parts)} checkpoint file(s))", flush=True)

    def _reset_cur(self):
        self.cur = dict(chunks=[], box_counts=[], out_pf=[], gt_rects=[],
                        n_frames=0, n_gt=0, wins=[])

    def add(self, n, gts, fc):
        c = self.cur
        c["wins"].append(n)
        c["n_frames"] += 1
        c["n_gt"] += int(bool(gts))
        for li, (x1, y1, x2, y2) in enumerate(gts):
            c["gt_rects"].append((n, x1, y1, x2, y2))
        if fc is None:
            c["out_pf"].append(0)
            c["box_counts"].extend((n * _BOXES_PER_FRAME + li, 0)
                                   for li in range(len(gts)))
        else:
            c["chunks"].append(fc["cols"])
            c["box_counts"].extend(fc["box_counts"])
            c["out_pf"].append(fc["out_per_frame"])
        self.done.add(n)
        if len(self.done) % 200 == 0:
            print(f"  [progress] windows {len(self.done)}/{self.n_todo} "
                  f"({100.0 * len(self.done) / max(1, self.n_todo):.0f}%)",
                  flush=True)
        if len(c["wins"]) >= _CKPT_FLUSH:
            self.flush()

    def _cur_part(self):
        c = self.cur
        cols = ([np.concatenate([ch[i] for ch in c["chunks"]])
                 for i in range(_N_COLS)] if c["chunks"] else _empty_cols())
        return dict(cols=cols, box_counts=c["box_counts"], out_pf=c["out_pf"],
                    gt_rects=c["gt_rects"], n_frames=c["n_frames"],
                    n_gt=c["n_gt"])

    def flush(self):
        if not self.cur["wins"]:
            return
        part = self._cur_part()
        if self.base:
            f = f"{self.base}.part{self.next_i:03d}.npz"
            np.savez(f, key=self.key,
                     wins=np.array(self.cur["wins"], np.int32),
                     box_keys=np.array([k for k, _ in part["box_counts"]], np.int32),
                     box_cand=np.array([cd for _, cd in part["box_counts"]], np.int32),
                     out_pf=np.array(part["out_pf"], np.int32),
                     gt_rects=(np.array(part["gt_rects"], np.int32)
                               if part["gt_rects"] else np.zeros((0, 5), np.int32)),
                     n_frames=part["n_frames"], n_gt=part["n_gt"],
                     **{f"c{i}": part["cols"][i] for i in range(_N_COLS)})
            self.next_i += 1
            print(f"  [checkpoint] windows {len(self.done)}/{self.n_todo} "
                  f"-> {os.path.basename(f)}", flush=True)
        self.parts.append(part)
        self._reset_cur()

    def finish(self):
        self.flush()
        return self.parts or [self._cur_part()]      # empty case: one empty part

    def cleanup(self):
        if not self.base:
            return
        import glob as _g
        for f in _g.glob(self.base + ".part*.npz"):
            try:
                os.remove(f)
            except OSError:
                pass


def _precompute_parallel(ctx, mode, n_scan, n_workers, stride):
    import queue as _q
    task_q = mp.Queue(maxsize=max(4, n_workers * 3)); result_q = mp.Queue()
    procs = [mp.Process(target=_pre_consumer, args=(task_q, result_q, ctx, mode), daemon=True)
             for _ in range(n_workers)]
    for pr in procs: pr.start()

    def _check_workers():
        """Fail the case IMMEDIATELY if any worker died without reporting (a
        hard MemoryError / OOM kill) — never block on a result that can't come."""
        dead = [pr for pr in procs if pr.exitcode not in (None, 0)]
        if dead:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
            raise RuntimeError(
                f"precompute worker died (exitcode {dead[0].exitcode}), "
                f"likely out of memory at {n_workers} workers")

    ck = _Ckpt(ctx.get("ckpt_base"), ctx.get("ckpt_key", ""),
               (n_scan + stride - 1) // stride)
    pending = [0]

    def _handle(msg):
        if msg[0] == "error":
            raise RuntimeError(f"precompute worker failed: {msg[1]}")
        _tag, n, gts, fc = msg
        ck.add(n, gts, fc)
        pending[0] -= 1

    def _drain_ready():
        while True:
            try:
                msg = result_q.get_nowait()
            except _q.Empty:
                return
            _handle(msg)

    rd = dv.RawReader(ctx["raw_file"], max_events=ctx["max_events"])
    _cfg, gt_rects_at = cs._apply_ctx(ctx); window_us = ctx["window_us"]
    try:
        for n in range(1, n_scan + 1):
            evs = rd.load_delta_t(window_us)
            if (n - 1) % stride != 0 or n in ck.done or n < PRECOMPUTE_START:
                continue                             # checkpointed: skip recompute
            while True:                              # bounded queue: put may wait
                try:
                    task_q.put((n, gt_rects_at(n), evs), timeout=5)
                    pending[0] += 1
                    break
                except _q.Full:
                    _check_workers()
                    _drain_ready()
            _drain_ready()
        for _ in procs:
            while True:
                try:
                    task_q.put(None, timeout=5)
                    break
                except _q.Full:
                    _check_workers()
                    _drain_ready()
        while pending[0] > 0:
            try:
                _handle(result_q.get(timeout=5))
            except _q.Empty:
                _check_workers()
        for pr in procs:
            pr.join(timeout=30)
    finally:
        ck.flush()                                   # keep whatever we have
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
    return ck


def _precompute_serial(ctx, mode, n_scan, stride):
    cfg, gt_rects_at = cs._apply_ctx(ctx)
    H, W = ctx["H"], ctx["W"]; window_us = ctx["window_us"]
    rd = dv.RawReader(ctx["raw_file"], max_events=ctx["max_events"])
    ck = _Ckpt(ctx.get("ckpt_base"), ctx.get("ckpt_key", ""),
               (n_scan + stride - 1) // stride)
    try:
        for n in range(1, n_scan + 1):
            evs = rd.load_delta_t(window_us)
            if (n - 1) % stride != 0 or n in ck.done or n < PRECOMPUTE_START:
                continue                             # checkpointed: skip recompute
            gts = gt_rects_at(n)
            prev_centers = _centers(gt_rects_at(n - 1))  # teacher-forced track origin
            ck.add(n, gts, _frame_candidates(n, gts, evs, cfg, mode, H, W,
                                             prev_centers))
    finally:
        ck.flush()                                   # keep whatever we have
    return ck


def _run_precompute(ctx, mode, n_scan, stride, n_workers):
    """Run the precompute with n_workers; on ANY failure of the multi-worker path
    (e.g. GPU OOM/contention when several processes share one GPU), fall back to a
    single in-process worker, which is robust. The fallback (and any external
    restart) RESUMES from the checkpoint files — completed windows are never
    recomputed."""
    if n_workers <= 1:
        return _precompute_serial(ctx, mode, n_scan, stride)
    try:
        return _precompute_parallel(ctx, mode, n_scan, n_workers, stride)
    except BaseException as e:
        print(f"  [warn] {n_workers}-worker precompute failed ({e!r}); "
              f"falling back to 1 worker (resuming from checkpoint) ...")
        return _precompute_serial(ctx, mode, n_scan, stride)


def _merge(parts, H, W):
    cols = [np.concatenate([p["cols"][i] for p in parts]) for i in range(_N_COLS)]
    fr, x, y, gc, size, sf, snr, ig, gb, dprev = cols
    order = np.argsort(fr, kind="stable")
    box_keys = np.array([k for p in parts for k, _c in p["box_counts"]], np.int32)
    box_cand = np.array([c for p in parts for _k, c in p["box_counts"]], np.int32)
    gtr = [r for p in parts for r in p["gt_rects"]]
    D = dict(
        fr=fr[order], x=x[order], y=y[order], gc=gc[order], size=size[order],
        sf=sf[order], snr=snr[order], ig=ig[order], gb=gb[order], dprev=dprev[order],
        box_keys=box_keys, box_cand=box_cand,
        out_per_frame=np.array([c for p in parts for c in p["out_pf"]], np.int32),
        gt_rects=(np.array(gtr, np.int32) if gtr else np.zeros((0, 5), np.int32)),
        n_frames=sum(p["n_frames"] for p in parts), n_gt=sum(p["n_gt"] for p in parts),
        H=H, W=W)
    return _finalize(D)


def _finalize(D):
    """Add per-frame slice index (table is frame-sorted) for the overlay."""
    fr = D["fr"]
    if fr.size:
        uniq, starts, counts = np.unique(fr, return_index=True, return_counts=True)
    else:
        uniq = np.array([], np.int32); starts = np.array([], np.int64); counts = np.array([], np.int64)
    D["uniq_frames"] = uniq; D["frame_starts"] = starts; D["frame_counts"] = counts
    D["H"] = int(D["H"]); D["W"] = int(D["W"])
    return D


# ── cache ─────────────────────────────────────────────────────────────────────────

def seq_dir(seq):
    return os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "fred", "code"), seq)

def cache_dir():
    """The one designated folder for all persistent caches this tool writes."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(d, exist_ok=True)
    return d

def features_path(seq, mode, variant=""):
    """Stable, descriptive per-(data-folder, mode) feature file in code/cache/.
    The name encodes the dataset and the data-folder number so it's obvious what
    each cache is, e.g.  features_cases_46_baseline.npz. This is BOTH
    the explorer's cache and the file ddhf_svm_explorer.py loads; it is refreshed
    whenever the structural params change. `variant` (see feature_variant) keeps
    optional spectral settings in their own file instead of overwriting this one."""
    ap = os.path.abspath(seq_dir(seq)).rstrip("/\\")
    seqbase = os.path.basename(ap) or "seq"
    dataset = os.path.basename(os.path.dirname(ap)) or "data"
    return os.path.join(cache_dir(), f"features_{dataset}_{seqbase}_{mode}{variant}.npz")

def feature_variant(cfg):
    """File-name suffix for optional spectral settings ('' when all are off, so
    existing feature files keep their names and keys)."""
    v = ""
    if getattr(cfg, "subharmonic_reject", False):
        v += (f"_subharm{cfg.subharmonic_half_width}_{cfg.subharmonic_min_hz:g}"
              f"_{cfg.subharmonic_ratio:g}")
        mode = getattr(cfg, "subharmonic_mode", "comb")
        k_min = int(getattr(cfg, "subharmonic_min_k", 2))
        if mode != "comb" or k_min != 2:      # only when non-default: old names kept
            v += f"_{mode}k{k_min}"
    if getattr(cfg, "comb_noise_stat", "median") != "median":
        v += f"_noise{cfg.comb_noise_stat}"
    if int(getattr(cfg, "spectral_oversample", 1)) > 1:
        v += f"_os{int(cfg.spectral_oversample)}"
    if float(getattr(cfg, "low_cut_hz", 0.0) or 0.0) > 0:
        v += f"_lowcut{cfg.low_cut_hz:g}"
    return v

FEATURES_VERSION = 2          # bump when the stored columns change (forces recompute)

def _cache_key(seq, mode, drone, stride, cfg, n_scan):
    return "|".join(str(v) for v in (
        f"v{FEATURES_VERSION}", seq, mode, cfg.window_ms, cfg.pixel_block, drone, stride,
        cfg.num_harmonics, cfg.min_harmonics, cfg.harmonic_half_width, cfg.max_freq_hz,
        cfg.noise_band_harmonics, cfg.min_present_harmonics, cfg.harmonic_present_factor,
        PRECOMPUTE_MIN_EVENTS, n_scan)
        + ((f"start{PRECOMPUTE_START}",) if PRECOMPUTE_START > 1 else ())) + feature_variant(cfg)

def _save_cache(path, D, key, meta):
    np.savez_compressed(path, key=key,
                        **{k: v for k, v in D.items() if isinstance(v, np.ndarray)},
                        n_frames=D["n_frames"], n_gt=D["n_gt"], H=D["H"], W=D["W"],
                        **{f"meta_{k}": v for k, v in meta.items()})

def _read_cache(path):
    """Load a feature file into a D dict (+ meta_*), or None if absent."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    D = {k: z[k] for k in z.files}
    D["n_frames"] = int(D["n_frames"]); D["n_gt"] = int(D["n_gt"])
    D["H"] = int(D["H"]); D["W"] = int(D["W"])
    return _finalize(D)

def load_features(seq, mode):
    """Public: load the saved feature file for (seq, mode), or None if it hasn't
    been produced yet (run ddhf_stats_explorer on that seq+mode first). Returns a
    D dict with gc/size/sf/snr/ig/fr/x/y/gt_rects + meta_* + frame index."""
    return _read_cache(features_path(seq, mode))


# ── build one sequence's dataset (cache or precompute) ───────────────────────────

def build_dataset(seq, mode, drone, stride, overrides, n_scan_cap, workers, gpu, refresh=False):
    DIR = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "fred", "code"), seq)
    import hf_cases as _hf
    RAW_FILE = _hf.event_file(DIR)          # events.raw or HF events.hdf5
    RGB_DIR = _hf.rgb_dir(DIR)             # PADDED_RGB/ or HF RGB/
    GT_FILE = os.path.join(DIR, "interpolated_coordinates.txt")
    if not os.path.exists(GT_FILE):          # HF case: coordinates.txt holds
        GT_FILE = os.path.join(DIR, "coordinates.txt")   # the same boxes
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"no event file for seq '{seq}' (looked in {RAW_FILE})")

    ov = dict(overrides); ov["min_events"] = PRECOMPUTE_MIN_EVENTS
    dv._COMPONENTS, dv._PIXELIZE, dv._POOLING = dv.MODE_FLAGS[mode]
    dv._USE_GPU = gpu
    cfg = build_cfg(**ov); cfg.rotor_bands_hz = dv.DRONE_BANDS[drone]
    window_us = int(round(cfg.window_ms * 1000.0))

    stream = dv.StreamState(RAW_FILE, RGB_DIR)
    H, W = stream.ev_height, stream.ev_width
    max_idx = stream.max_frame_idx(window_us); max_events = stream.max_events
    del stream
    n_scan = max_idx if n_scan_cap is None else min(max_idx, n_scan_cap)

    key = _cache_key(seq, mode, drone, stride, cfg, n_scan)
    cache = features_path(seq, mode, feature_variant(cfg))
    if not refresh and os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=False)
            ok = str(z["key"]) == key                # same structural params -> reuse
        except Exception as exc:                     # truncated/corrupt npz (e.g.
            print(f"[cache] {seq}: unreadable ({exc!r}) — deleting and "
                  "recomputing")                     # killed mid-save): rebuild
            try:
                os.remove(cache)
            except OSError:
                pass
            ok = False
        if ok:
            D = _read_cache(cache)
            print(f"[cache] {seq}: {D['gc'].size} candidate px over {D['n_frames']} frames "
                  f"<- {os.path.basename(cache)}")
            return D

    gt_times, gt_boxes = dv._load_gt_boxes(GT_FILE)
    ctx = dict(raw_file=RAW_FILE, window_us=window_us, H=H, W=W, max_events=max_events,
               mode=mode, drone=drone, gpu=gpu, overrides=ov,
               gt_times=gt_times, gt_boxes=gt_boxes, gt_tol_s=cfg.window_ms / 1000.0,
               ckpt_base=(cache[:-4] if cache.endswith(".npz") else cache),
               ckpt_key=key)
    if workers is not None:
        n_workers = max(1, workers)
    elif gpu:
        n_workers = 4                     # a few processes share the GPU (falls back to 1 on failure)
    else:
        n_workers = max(1, os.cpu_count() or 1)
    print(f"[precompute] {seq} mode={mode} window={cfg.window_ms}ms stride={stride} "
          f"-> {n_scan} windows, {n_workers} worker(s){' [GPU]' if gpu else ''} ...")
    ck = _run_precompute(ctx, mode, n_scan, stride, n_workers)
    D = _merge(ck.finish(), H, W)
    meta = dict(seq=str(seq), mode=str(mode), drone=str(drone),
                window_ms=float(cfg.window_ms), stride=int(stride),
                pixel_block=int(cfg.pixel_block))
    _save_cache(cache, D, key, meta)
    ck.cleanup()                                     # final npz saved: drop partials
    print(f"[precompute] {seq}: {D['gc'].size} candidate px over {D['n_frames']} frames "
          f"(GT {D['n_gt']}) -> saved {os.path.basename(cache)}")
    return D


# ── live stats from the precomputed table ────────────────────────────────────────

def _box_pass_mask(D, green, min_rotor_pixels, min_box_area):
    """Per-frame 8-connected components over GREEN pixels (small): keep components
    with >= min_rotor_pixels px and bbox area >= min_box_area. (perpixel modes)"""
    pass_mask = np.zeros(green.size, dtype=bool)
    gi = np.flatnonzero(green)
    if gi.size == 0:
        return pass_mask, False
    if gi.size > BOX_CAP:
        return green.copy(), True
    fr = D["fr"][gi]; xs = D["x"][gi].astype(np.int64); ys = D["y"][gi].astype(np.int64)
    for seg in np.split(np.arange(gi.size), np.flatnonzero(np.diff(fr)) + 1):
        sx = xs[seg]; sy = ys[seg]
        x0, y0 = sx.min(), sy.min()
        grid = np.zeros((sy.max() - y0 + 1, sx.max() - x0 + 1), dtype=bool)
        grid[sy - y0, sx - x0] = True
        lbl, ncomp = _label(grid, structure=np.ones((3, 3), int))
        if ncomp == 0:
            continue
        comp_px = np.bincount(lbl.ravel(), minlength=ncomp + 1)
        keep = np.zeros(ncomp + 1, dtype=bool)
        for ci, sl in enumerate(_find_objects(lbl), start=1):
            if sl is None or comp_px[ci] < min_rotor_pixels:
                continue
            if (sl[1].stop - sl[1].start) * (sl[0].stop - sl[0].start) >= min_box_area:
                keep[ci] = True
        pass_mask[gi[seg]] = keep[lbl[sy - y0, sx - x0]]
    return pass_mask, False


def compute_stats(D, p, mode):
    """Apply threshold params p to the precomputed table -> per-candidate stage
    array + per-region stage counts + metric summaries (inside/outside GT)."""
    gc, size, sf, snr, ig = D["gc"], D["size"], D["sf"], D["snr"], D["ig"].astype(bool)
    pass_me = gc >= p["min_events"]
    stage = np.empty(gc.size, dtype=np.int8)
    skipped = False
    if mode == "components":
        # gate order: min_events -> SF -> SNR -> min_rotor_pixels (blob size last).
        ok_sf   = pass_me & (sf <= p["tau_sf"])
        ok_snr  = ok_sf   & (snr >= p["tau_comb"])
        ok_size = ok_snr  & (size >= p["min_rotor_pixels"])
        stage[~pass_me]                        = F_MINEV
        stage[pass_me & (sf > p["tau_sf"])]    = F_SF
        stage[ok_sf  & (snr < p["tau_comb"])]  = F_SNR
        stage[ok_snr & (size < p["min_rotor_pixels"])] = F_MINPX
        stage[ok_size]                         = PASS
        measured = pass_me
    else:
        ok_sf  = pass_me & (sf <= p["tau_sf"])
        ok_snr = ok_sf   & (snr >= p["tau_comb"])
        green  = ok_snr
        box_pass, skipped = _box_pass_mask(D, green, p["min_rotor_pixels"], p["min_box_area"])
        stage[~pass_me]                       = F_MINEV
        stage[pass_me & (sf > p["tau_sf"])]   = F_SF
        stage[ok_sf  & (snr < p["tau_comb"])] = F_SNR
        stage[green  & ~box_pass]             = F_BOX
        stage[green  &  box_pass]             = PASS
        measured = pass_me

    res = {"skipped_box": skipped, "stage": stage}
    for name, mask in (("in", ig), ("out", ~ig)):
        res[name] = np.bincount(np.where(mask, stage, 0)[mask], minlength=6).astype(np.int64) \
            if mask.any() else np.zeros(6, np.int64)
        meas = mask & measured
        res[name + "_metrics"] = {"ev": _mm(gc[mask]), "sf": _mm(sf[meas]), "snr": _mm(snr[meas])}
    passed = stage == PASS
    pb = np.zeros(D["box_keys"].size, dtype=np.int64)
    gbp = D["gb"][passed]; gbp = gbp[gbp >= 0]
    if gbp.size and D["box_keys"].size:
        cnt = np.bincount(gbp)
        m = D["box_keys"] < cnt.size
        pb[m] = cnt[D["box_keys"][m]]
    res["pass_per_box"] = pb
    return res


def _mm(a):
    if a.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(a.mean()), float(np.median(a)), int(a.size))


# ── headless report (for --selftest) ──────────────────────────────────────────────

def print_report(D, p, mode):
    r = compute_stats(D, p, mode)
    print(f"\n[mode={mode}] " + "  ".join(f"{k}={v}" for k, v in p.items()))
    if r["skipped_box"]:
        print("  (box gate skipped: too many green px)")
    for name, title in (("in", "INSIDE GT"), ("out", "OUTSIDE GT")):
        c = r[name]; tot = int(c.sum())
        print(f"  {title}: {tot} candidate px")
        for code in stage_order(mode):
            if tot:
                print(f"      {STAGE_LABEL[code]:<18} {int(c[code]):>10} ({c[code]/tot*100:5.1f}%)")
        mt = r[name + "_metrics"]
        print(f"      metrics ev={mt['ev'][0]:.2f}/{mt['ev'][1]:.1f} "
              f"sf={mt['sf'][0]:.3f}/{mt['sf'][1]:.3f} snr={mt['snr'][0]:.3f}/{mt['snr'][1]:.3f}")


# ── interactive GUI ──────────────────────────────────────────────────────────────

def launch_gui(D0, seq0, mode, ctx_build):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox
    from matplotlib.patches import Rectangle

    G = {"D": D0, "seq": seq0}
    ORDER = stage_order(mode)

    state = dict(min_events=4, tau_sf=0.7, tau_comb=10.0,
                 min_rotor_pixels=30, min_box_area=50, frame_pos=0.0)

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"DDHF stats explorer [{mode}]")
    ax_in  = fig.add_axes([0.06, 0.60, 0.27, 0.33])
    ax_out = fig.add_axes([0.38, 0.60, 0.27, 0.33])
    ax_ov  = fig.add_axes([0.69, 0.52, 0.29, 0.41]); ax_ov.set_xticks([]); ax_ov.set_yticks([])
    ax_txt = fig.add_axes([0.06, 0.34, 0.60, 0.18]); ax_txt.axis("off")
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)

    def frame_default(D):
        if D["uniq_frames"].size == 0:
            return 0.0
        ig = D["ig"].astype(bool)
        if ig.any():
            ic = np.bincount(D["fr"][ig])
            best = int(np.argmax(ic))
            pos = int(np.searchsorted(D["uniq_frames"], best))
            pos = min(pos, D["uniq_frames"].size - 1)
        else:
            pos = D["uniq_frames"].size // 2
        return pos / max(1, D["uniq_frames"].size - 1)

    state["frame_pos"] = frame_default(G["D"])

    def draw_region(ax, c, title):
        ax.clear()
        tot = int(c.sum())
        xs = np.arange(len(ORDER))
        vals = [c[code] / tot * 100 if tot else 0 for code in ORDER]
        ax.bar(xs, vals, color=[STAGE_HEX[code] for code in ORDER])
        for xi, code, v in zip(xs, ORDER, vals):
            ax.text(xi, v + 1, f"{int(c[code])}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([STAGE_LABEL[code].replace("fail ", "✗") for code in ORDER],
                           fontsize=7, rotation=18)
        ax.set_ylim(0, 108); ax.set_ylabel("% of region")
        ax.set_title(f"{title}  ({tot} px)", fontsize=10)

    def draw_overlay(res):
        D = G["D"]; ax_ov.clear(); ax_ov.set_xticks([]); ax_ov.set_yticks([])
        if D["uniq_frames"].size == 0:
            ax_ov.set_title("overlay: no candidates"); return
        i = int(round(state["frame_pos"] * (D["uniq_frames"].size - 1)))
        i = max(0, min(i, D["uniq_frames"].size - 1))
        f = int(D["uniq_frames"][i]); s = int(D["frame_starts"][i]); e = s + int(D["frame_counts"][i])
        canvas = np.full((D["H"], D["W"], 3), 18, np.uint8)
        xs = D["x"][s:e].astype(np.int64); ys = D["y"][s:e].astype(np.int64)
        st = res["stage"][s:e]
        for code in (F_MINEV, F_SF, F_SNR, F_MINPX, F_BOX, PASS):
            m = st == code
            if m.any():
                canvas[ys[m], xs[m]] = STAGE_RGB[code]
        ax_ov.imshow(canvas, origin="upper", interpolation="nearest")
        gr = D["gt_rects"]
        for (gf, x1, y1, x2, y2) in gr[gr[:, 0] == f] if gr.size else []:
            ax_ov.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="cyan", lw=1.5))
        npx = e - s
        ax_ov.set_title(f"frame {f}: candidate px coloured by stage  (n={npx})", fontsize=9)

    def refresh(_=None):
        D = G["D"]
        r = compute_stats(D, state, mode)
        draw_region(ax_in, r["in"], "INSIDE GT")
        draw_region(ax_out, r["out"], "OUTSIDE GT")
        draw_overlay(r)
        mi = r["in_metrics"]; mo = r["out_metrics"]; pb = r["pass_per_box"]
        bc = D["box_cand"]; opf = D["out_per_frame"]
        cpb = (bc.mean(), np.median(bc)) if bc.size else (np.nan, np.nan)
        skip = "  [BOX GATE SKIPPED]" if r["skipped_box"] else ""
        lines = [
            f"seq={G['seq']}  mode={mode}  min_events>={state['min_events']}  "
            f"SF<={state['tau_sf']:.3f}  SNR>={state['tau_comb']:.2f}  "
            f"min_rotor_px={state['min_rotor_pixels']}"
            + (f"  min_box_area={state['min_box_area']}" if mode != "components" else "") + skip,
            "",
            f"{'':20}{'INSIDE GT':>22}{'OUTSIDE GT':>22}",
            f"{'events/px mean·med':20}{mi['ev'][0]:>11.2f}{mi['ev'][1]:>11.1f}"
            f"{mo['ev'][0]:>11.2f}{mo['ev'][1]:>11.1f}",
            f"{'SF mean·med':20}{mi['sf'][0]:>11.3f}{mi['sf'][1]:>11.3f}"
            f"{mo['sf'][0]:>11.3f}{mo['sf'][1]:>11.3f}",
            f"{'comb-SNR mean·med':20}{mi['snr'][0]:>11.3f}{mi['snr'][1]:>11.3f}"
            f"{mo['snr'][0]:>11.3f}{mo['snr'][1]:>11.3f}",
            "",
            (f"candidate px/GT box mean={cpb[0]:.1f} med={cpb[1]:.1f}   "
             f"PASS px/GT box mean={pb.mean():.2f} med={np.median(pb):.1f}" if pb.size else ""),
            f"outside-GT px/frame mean={opf.mean():.1f} med={np.median(opf):.1f}   "
            f"total cand={D['gc'].size}   frames={D['n_frames']} (GT {D['n_gt']})"
            if opf.size else f"total cand={D['gc'].size}",
        ]
        txt.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    # sliders (mode-dependent)
    sl_defs = [("min_events", 1, max(2, int(G["D"]["gc"].max()) if G["D"]["gc"].size else 10),
                state["min_events"], 1)]
    if mode == "components":
        sl_defs += [("min_rotor_pixels", 1, 400, state["min_rotor_pixels"], 1),
                    ("tau_sf", 0.0, 1.0, state["tau_sf"], None),
                    ("tau_comb", 0.0, 30.0, state["tau_comb"], None)]
    else:
        sl_defs += [("tau_sf", 0.0, 1.0, state["tau_sf"], None),
                    ("tau_comb", 0.0, 30.0, state["tau_comb"], None),
                    ("min_rotor_pixels", 1, 200, state["min_rotor_pixels"], 1),
                    ("min_box_area", 1, 2000, state["min_box_area"], 1)]
    sliders = {}
    for i, (name, lo, hi, init, step) in enumerate(sl_defs):
        ax_s = fig.add_axes([0.30, 0.22 - i * 0.032, 0.55, 0.020])
        s = Slider(ax_s, name, lo, hi, valinit=init, valstep=step)
        def _mk(nm):
            def _cb(val):
                state[nm] = int(val) if nm in ("min_events", "min_rotor_pixels", "min_box_area") else float(val)
                refresh()
            return _cb
        s.on_changed(_mk(name)); sliders[name] = s

    # frame slider (normalised 0..1 -> index into uniq_frames)
    ax_fr = fig.add_axes([0.69, 0.45, 0.29, 0.020])
    s_frame = Slider(ax_fr, "frame", 0.0, 1.0, valinit=state["frame_pos"])
    def _frame_cb(val):
        state["frame_pos"] = float(val); refresh()
    s_frame.on_changed(_frame_cb)

    # sequence picker
    ax_seq = fig.add_axes([0.10, 0.04, 0.18, 0.04])
    tb = TextBox(ax_seq, "seq: ", initial=os.path.basename(seq0.rstrip("/\\")))
    def _seq_submit(text):
        text = text.strip()
        if not text:
            return
        new_seq = text if ("/" in text or "\\" in text) else \
            os.path.join(os.path.dirname(seq0.rstrip("/\\")), text)
        try:
            print(f"[switch] loading seq '{new_seq}' ...")
            D = ctx_build(new_seq)
        except Exception as e:
            txt.set_text(f"could not load seq '{new_seq}':\n{e}"); fig.canvas.draw_idle(); return
        G["D"] = D; G["seq"] = new_seq
        state["frame_pos"] = frame_default(D)
        s_frame.eventson = False; s_frame.set_val(state["frame_pos"]); s_frame.eventson = True
        refresh()
    tb.on_submit(_seq_submit)

    refresh()
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=str, default="3")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--window-ms", type=float, default=33.333)
    add_cfg_cli_args(parser)
    parser.add_argument("--drone", choices=sorted(dv.DRONE_BANDS), default="all")
    parser.add_argument("--stride", type=int, default=4,
                        help="Precompute every Nth window (default 4). Structural (cache key).")
    parser.add_argument("--gpu", action="store_true",
                        help="Run the spectral stage on the GPU during precompute (4 workers "
                             "sharing the GPU, auto-fallback to 1 worker on any failure). "
                             "Override the count with --workers.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="No GUI: print a couple of stat tables.")
    args = parser.parse_args()
    stride = max(1, args.stride)

    overrides = cfg_overrides_from_args(args)
    overrides["window_ms"] = args.window_ms
    overrides["step_ms"]   = args.window_ms

    def make(seq, refresh=False):
        return build_dataset(seq, args.mode, args.drone, stride, overrides,
                             args.max_frames, args.workers, args.gpu, refresh=refresh)

    D = make(args.seq, refresh=args.refresh_cache)

    if args.selftest:
        if args.mode == "components":
            ps = (dict(min_events=4, min_rotor_pixels=30, tau_sf=0.7, tau_comb=10.0, min_box_area=0),
                  dict(min_events=2, min_rotor_pixels=10, tau_sf=0.7, tau_comb=6.0,  min_box_area=0))
        else:
            ps = (dict(min_events=4, tau_sf=0.7, tau_comb=10.0, min_rotor_pixels=30, min_box_area=50),
                  dict(min_events=2, tau_sf=0.7, tau_comb=6.0,  min_rotor_pixels=10, min_box_area=20))
        for p in ps:
            print_report(D, p, args.mode)
        return

    launch_gui(D, args.seq, args.mode, lambda s: make(s))


if __name__ == "__main__":
    main()
