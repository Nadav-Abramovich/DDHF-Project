"""
Spatial-pooling variant of the DDHF spectral detector.

Identical to utils.py except _detect_rotor_pixels: before the NDFT each
pixel's event set is augmented with events from its 8 immediate neighbours
(the full 3x3 block centred on that pixel).

Performance design
------------------
The naive "scatter" approach broadcasts every event to 9 target pixels,
producing a 9x-larger array that must be sorted again.  This module uses
a "gather" approach instead:

  1. Sort the original N events by pixel-id once  -- O(N log N)
  2. For each of the 9 offsets (dx, dy), shift the active pixel-ids and
     call np.searchsorted to find the matching event slice      -- O(A log N)
     (A = number of pixels with >= 1 own event, A << H*W)
  3. Concatenate the gathered slices and sort by TARGET pixel-id -- O(M log M)
     where M <= 9*N but typically much less because empty pixels
     are never candidates.

Total work: O(N log N + 9*A*log N) ≈ O(N log N) — roughly 9x less sorting
than the scatter approach, and no NDFT is ever run on a completely empty pixel.
"""

import os
import numpy as np
from scipy import ndimage as _ndi
from utils import (
    DDHFConfig, BoundingBox, DDHF,
    _ndft_power_batch, _spectral_flatness_batch, _comb_snr_batch,
    _extract_boxes as _extract_boxes_ref, _merge_boxes,
    _detect_rotor_pixels as _detect_rotor_pixels_plain,
    _normalise_polarity, _normalise_timestamps,
    rgb_frame_seconds,
)

# ── Optional GPU (CuPy) backend ────────────────────────────────────────────────
# The heavy per-window work is the NDFT power spectrum and the harmonic-comb
# sweep (dense (K, N) complex phasors + per-group reductions). When CuPy is
# importable AND a CUDA device is usable, those three steps run on the GPU; the
# cheap index work (3x3 gather, grouping, box extraction) stays on the CPU and
# only the small final per-pixel arrays are copied back. Set DDHF_NO_GPU=1 to
# force the CPU path. The maths mirrors utils.py exactly, so detections are the
# same up to floating-point reduction order.
_GPU_DISABLED = os.environ.get("DDHF_NO_GPU", "0") == "1"
# Precision of the GPU spectral math. float32 is markedly faster on consumer
# GPUs (the RTX 4050 runs float64 at 1/64 rate), at the cost of a handful of
# borderline pixels flipping vs the exact float64/CPU result. Default ON; set
# DDHF_GPU_FP32=0 to force the bit-identical float64 path.
_GPU_FP32 = os.environ.get("DDHF_GPU_FP32", "1") != "0"
try:
    import cupy as _cp                       # noqa: N816
except Exception:                            # cupy not installed / import error
    _cp = None

_gpu_checked = False
_gpu_ok = False

# Per-run accounting of which spectral path each window's _detect_rotor_pixels
# call took, so callers (e.g. ddhf_eval_pooled) can show GPU vs CPU-fallback at
# a glance. "gpu": ran on the device; "cpu_fallback": GPU attempted but OOM'd so
# the window was recomputed on the CPU; "cpu": GPU unavailable/disabled.
_PATH_STATS = {"gpu": 0, "cpu_fallback": 0, "cpu": 0}


def reset_path_stats():
    """Zero the per-window path counters (call once before a sequence)."""
    for k in _PATH_STATS:
        _PATH_STATS[k] = 0


def path_stats():
    """Snapshot of the per-window path counters as a plain dict."""
    return dict(_PATH_STATS)


def _gpu_dtypes():
    """(real, complex) CuPy dtypes for the GPU spectral math, per DDHF_GPU_FP32."""
    if _GPU_FP32:
        return _cp.float32, _cp.complex64
    return _cp.float64, _cp.complex128


def _gpu_available() -> bool:
    """True iff CuPy is importable and can actually compile/run a kernel."""
    global _gpu_checked, _gpu_ok
    if _gpu_checked:
        return _gpu_ok
    _gpu_checked = True
    if _cp is None or _GPU_DISABLED:
        _gpu_ok = False
        return _gpu_ok
    try:
        # Force an nvrtc kernel compile + device round-trip; fails fast if the
        # CUDA toolchain/driver isn't actually usable (e.g. missing headers).
        _probe = _cp.exp(1j * _cp.arange(4.0))
        float(_cp.abs(_probe).sum().get())
        _gpu_ok = True
    except Exception:
        _gpu_ok = False
    return _gpu_ok


def _ndft_power_batch_gpu(pol_sorted, tn_sorted, group_starts, K,
                          max_chunk_elems: int = 32_000_000):
    """GPU twin of utils._ndft_power_batch. Returns a CuPy (K, G) power array.

    Same chunked construction of the (K, n_e) phasor matrix and the same
    np.add.reduceat segment sum per pixel group, so each group's spectrum is
    computed identically to the CPU path (modulo float reduction order)."""
    rdt, cdt = _gpu_dtypes()
    G = group_starts.size
    N = int(tn_sorted.size)
    pol_g = _cp.asarray(pol_sorted, dtype=rdt)
    tn_g  = _cp.asarray(tn_sorted,  dtype=rdt)
    k     = _cp.arange(K, dtype=rdt)
    power = _cp.empty((K, G), dtype=rdt)

    gs = np.asarray(group_starts)                       # host: chunk bookkeeping
    target_events = max(1, max_chunk_elems // K)
    g0 = 0
    while g0 < G:
        e0 = int(gs[g0])
        g1 = g0 + 1
        while g1 < G and (int(gs[g1]) - e0) <= target_events:
            g1 += 1
        e1 = int(gs[g1]) if g1 < G else N

        phase = _cp.outer(k, tn_g[e0:e1])               # (K, n_e)
        E = _cp.exp(1j * phase)                          # (K, n_e) complex
        if E.dtype != cdt:                               # pin precision (e.g. fp32)
            E = E.astype(cdt)
        E *= pol_g[e0:e1][None, :]
        local_starts = _cp.asarray(gs[g0:g1] - e0)
        F = _cp.add.reduceat(E, local_starts, axis=1)   # (K, g1-g0)
        power[:, g0:g1] = _cp.abs(F) ** 2
        g0 = g1
    return power


def _spectral_flatness_batch_gpu(power_no_dc, eps: float = 1e-12):
    """GPU twin of utils._spectral_flatness_batch (operates on CuPy arrays)."""
    p = power_no_dc + eps
    geo = _cp.exp(_cp.mean(_cp.log(p), axis=0))
    arith = _cp.mean(p, axis=0)
    return geo / arith


def _comb_snr_batch_gpu(power, cfg: DDHFConfig, max_cols: int = 50_000,
                        return_omega: bool = False):
    """GPU twin of utils._comb_snr_batch. ``power`` is a CuPy (K, G) array;
    returns a CuPy (G,) array of the best harmonic-comb SNR per pixel group.
    With ``return_omega=True`` also returns the winning fundamental bin per pixel
    (G,) int64 — mirrors the CPU function and lets the debug viewer recover f0
    while running the exact GPU detection path.

    The harmonic / noise bin masks depend only on (K, cfg, omega) — not on the
    data — so they are built once on the host and reused across windows.

    On dense windows G (candidate pixels) can reach hundreds of thousands; the
    ``cp.median`` over the noise bins sorts a copy whose temp buffer was the
    allocation that OOM'd (the merge_sort failure). Each pixel column is fully
    independent, so the columns are processed in blocks of ``max_cols`` — this
    bounds every transient (sort buffer, peak stack) to (K, max_cols) and is
    bit-identical to the unchunked version regardless of block size."""
    K, G = power.shape
    rdt = power.dtype
    best = _cp.zeros(G, dtype=rdt)
    best_om = _cp.zeros(G, dtype=_cp.int64) if return_omega else None
    plan = _comb_plan(K, cfg)
    for c0 in range(0, G, max_cols):
        c1 = min(c0 + max_cols, G)
        pblock = power[:, c0:c1]
        bblock = _cp.zeros(c1 - c0, dtype=rdt)
        bomblock = _cp.zeros(c1 - c0, dtype=_cp.int64) if return_omega else None
        for omega, noise_idx, peak_slices in plan:
            peak_powers = [pblock[lo:hi, :].max(axis=0) for lo, hi in peak_slices]
            noise = pblock[noise_idx, :]
            median_noise = _cp.median(noise, axis=0)
            mean_peaks = _cp.mean(_cp.stack(peak_powers, axis=0), axis=0)
            snr = _cp.zeros(c1 - c0, dtype=rdt)
            ok = median_noise >= 1e-10
            snr[ok] = mean_peaks[ok] / median_noise[ok]

            # "Harmonics present" gate (matches utils._comb_snr_batch): require
            # >= min_present_harmonics windows with peak >= factor × noise-median.
            mph = getattr(cfg, "min_present_harmonics", 0)
            if mph > 0:
                factor = getattr(cfg, "harmonic_present_factor", 2.0)
                present = _cp.zeros(c1 - c0, dtype=_cp.int32)
                for pk in peak_powers:
                    present += (pk >= factor * median_noise)
                snr = _cp.where(present >= mph, snr, 0).astype(rdt)

            better = snr > bblock
            bblock[better] = snr[better]
            if return_omega:
                bomblock[better] = omega
        best[c0:c1] = bblock
        if return_omega:
            best_om[c0:c1] = bomblock
    if return_omega:
        return best, best_om
    return best


_COMB_PLAN_CACHE: dict = {}


def _comb_plan(K: int, cfg: DDHFConfig):
    """Precompute, once per (K, cfg), the per-omega harmonic peak slices and the
    noise-bin index array — identical selection logic to utils._comb_snr_batch.

    Returns a list of (omega, noise_idx[int64], [(lo, hi), ...]) where omega is
    the fundamental bin, each (lo, hi) is a half-open slice over the harmonic
    window and noise_idx are the bins outside every harmonic window (DC excluded).
    The slices/indices are GPU arrays so the per-window SNR loop never rebuilds
    them. omega is carried so callers can recover the winning fundamental."""
    key = (K, cfg.num_harmonics, cfg.min_harmonics, cfg.harmonic_half_width,
           cfg.window_ms, cfg.rotor_bands_hz,
           getattr(cfg, "noise_band_harmonics", 0.0))
    plan = _COMB_PLAN_CACHE.get(key)
    if plan is not None:
        return plan

    bands  = cfg.rotor_bands_hz
    bin_hz = 1000.0 / cfg.window_ms
    plan = []
    for omega in range(1, K // cfg.num_harmonics + 1):
        if bands and not any(lo <= omega * bin_hz <= hi for lo, hi in bands):
            continue
        harmonics = [m * omega for m in range(1, cfg.num_harmonics + 1)
                     if m * omega < K]
        if len(harmonics) < cfg.min_harmonics:
            continue
        harmonic_mask = np.zeros(K, dtype=bool)
        peak_slices = []
        for h in harmonics:
            lo = max(0, h - cfg.harmonic_half_width)
            hi = min(K - 1, h + cfg.harmonic_half_width)
            peak_slices.append((lo, hi + 1))            # half-open
            harmonic_mask[lo:hi + 1] = True
        noise_mask = ~harmonic_mask
        noise_mask[0] = False
        nbh = getattr(cfg, "noise_band_harmonics", 0.0)
        if nbh > 0:                                  # local noise floor (see utils._comb_snr_batch)
            band_hi = min(K, int(nbh * harmonics[-1]) + cfg.harmonic_half_width + 1)
            noise_mask[band_hi:] = False
        if not noise_mask.any():
            continue
        noise_idx = _cp.asarray(np.flatnonzero(noise_mask))
        plan.append((omega, noise_idx, peak_slices))
    _COMB_PLAN_CACHE[key] = plan
    return plan


def _extract_boxes(rotor, scores, cfg):
    """Faster drop-in for utils._extract_boxes producing the same boxes.

    The reference version scans the whole (H, W) label image once per connected
    component (``labeled == lid`` + ``np.where``), i.e. O(n_labels · H · W) — on
    dense windows with thousands of stray rotor pixels that dominates the whole
    per-window cost. Here every component's bounding box comes from a single
    ``ndimage.find_objects`` pass and every component's mean score from a single
    ``ndimage.mean`` pass, so the work is O(H · W) regardless of label count. Box
    geometry is identical; the score is computed in float64 (vs the reference's
    float32 reduction) so it can differ by ~1e-6, which does not affect the
    IoU-based evaluation."""
    if not rotor.any():
        return []
    labeled, n = _ndi.label(rotor, structure=np.ones((3, 3)))
    if n == 0:
        return []
    slices = _ndi.find_objects(labeled)
    means = np.atleast_1d(_ndi.mean(scores, labeled, index=np.arange(1, n + 1)))
    # Rotor-pixel count per component (index 0 = background). One bincount pass,
    # so the extra gate is essentially free on top of the existing label work.
    comp_pixels = np.bincount(labeled.ravel(), minlength=n + 1)

    boxes = []
    for i, sl in enumerate(slices):
        if sl is None:                       # label id with no pixels (defensive)
            continue
        if comp_pixels[i + 1] < cfg.min_rotor_pixels:   # sparse component -> noise
            continue
        y1, y2 = sl[0].start, sl[0].stop - 1
        x1, x2 = sl[1].start, sl[1].stop - 1
        b = BoundingBox(int(x1), int(y1), int(x2), int(y2), score=float(means[i]))
        if b.area >= cfg.min_box_area:
            boxes.append(b)
    return _merge_boxes(boxes, cfg.alpha)

__all__ = [
    "DDHFConfig", "BoundingBox", "DDHF",
    "_ndft_power_batch", "_spectral_flatness_batch", "_comb_snr_batch",
    "_extract_boxes", "_normalise_polarity", "_normalise_timestamps",
    "rgb_frame_seconds",
    "_gather_3x3", "_detect_rotor_pixels", "_detect_rotor_pixels_pixelized",
    "reset_path_stats", "path_stats",
]

_POOL_DX = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
_POOL_DY = np.array([-1,-1,-1,  0, 0, 0,  1, 1, 1], dtype=np.int32)


def _gather_3x3(
    pid_sorted:  np.ndarray,   # (N,) int64 — pixel-ids of ALL events, sorted
    pol_sorted:  np.ndarray,   # (N,) float64 — polarity, same order
    tn_sorted:   np.ndarray,   # (N,) float64 — normalised timestamps, same order
    active_pids: np.ndarray,   # (A,) int64 — unique pixel-ids with >= 1 own event
    ev_height:   int,
    ev_width:    int,
) -> tuple:
    """
    For each pixel in active_pids, collect all events in its 3x3 neighbourhood.

    Uses searchsorted on the pre-sorted event array — no 9x expansion.
    Each source event may appear up to 9 times in the output (once per active
    target pixel whose neighbourhood it falls in), but only events adjacent to
    an active pixel are ever emitted; background regions are skipped entirely.

    Returns
    -------
    pid_out : (M,) int64 — TARGET pixel-id for each gathered event
    pol_out : (M,) float64
    tn_out  : (M,) float64
    All three arrays are sorted by pid_out.
    """
    active_y = (active_pids // ev_width).astype(np.int32)
    active_x = (active_pids %  ev_width).astype(np.int32)

    chunks_pid: list = []
    chunks_pol: list = []
    chunks_tn:  list = []

    for dx, dy in zip(_POOL_DX, _POOL_DY):
        nx = active_x + dx
        ny = active_y + dy
        valid = (nx >= 0) & (nx < ev_width) & (ny >= 0) & (ny < ev_height)
        if not valid.any():
            continue

        nbr_pids = (ny[valid].astype(np.int64) * ev_width
                    + nx[valid].astype(np.int64))
        tgt_pids = active_pids[valid]

        lo = np.searchsorted(pid_sorted, nbr_pids, side='left')
        hi = np.searchsorted(pid_sorted, nbr_pids, side='right')

        counts = (hi - lo).astype(np.int64)
        has_ev = counts > 0
        if not has_ev.any():
            continue

        lo_f    = lo[has_ev];     hi_f  = hi[has_ev]
        tgt_f   = tgt_pids[has_ev]; cnt_f = counts[has_ev]
        total   = int(cnt_f.sum())

        # Vectorised range expansion: produce indices lo_f[i], lo_f[i]+1, ...
        # hi_f[i]-1 for every i, without a Python loop.
        cum    = np.empty(len(cnt_f) + 1, dtype=np.int64)
        cum[0] = 0
        np.cumsum(cnt_f, out=cum[1:])
        inc = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], cnt_f)
        idx = np.repeat(lo_f, cnt_f) + inc

        chunks_pol.append(pol_sorted[idx])
        chunks_tn.append(tn_sorted[idx])
        chunks_pid.append(np.repeat(tgt_f, cnt_f))

    if not chunks_pid:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    pid_cat = np.concatenate(chunks_pid)
    pol_cat = np.concatenate(chunks_pol)
    tn_cat  = np.concatenate(chunks_tn)

    order = np.argsort(pid_cat, kind='stable')
    return pid_cat[order], pol_cat[order], tn_cat[order]


def _scatter(valid, snr, scatter):
    """Place per-group (valid, snr) onto a full-res (rotor, scores) pair, using
    the geometry produced by _prep_pooled / _prep_pixelized. Shared so the
    single-window and batched paths build identical maps."""
    sel = np.flatnonzero(valid)
    if scatter[0] == "pooled":
        _, py, px, height, width = scatter
        rotor  = np.zeros((height, width), dtype=bool)
        scores = np.zeros((height, width), dtype=np.float32)
        rotor[py[sel], px[sel]]  = True
        scores[py[sel], px[sel]] = snr[sel].astype(np.float32)
        return rotor, scores
    # "pixelize": set accepted blocks then upscale each to its BxB footprint
    _, by, bx, B, Hb, Wb, height, width = scatter
    rotor_b  = np.zeros((Hb, Wb), dtype=bool)
    scores_b = np.zeros((Hb, Wb), dtype=np.float32)
    rotor_b[by[sel],  bx[sel]]  = True
    scores_b[by[sel], bx[sel]] = snr[sel].astype(np.float32)
    rotor  = np.repeat(np.repeat(rotor_b,  B, axis=0), B, axis=1)[:height, :width]
    scores = np.repeat(np.repeat(scores_b, B, axis=0), B, axis=1)[:height, :width]
    return rotor, scores


def _prep_pooled(x, y, p, t_norm, height, width, cfg):
    """CPU stage of the 3x3-pooled detector: gather each active pixel's 3x3 pool,
    apply the min_events (+ optional min_own_events) gate, and return the grouped
    events ready for the spectral stage, plus scatter geometry. None if nothing
    qualifies. Split out so multiple windows can share one spectral GPU call."""
    if x.size == 0:
        return None
    pixel_id   = y * width + x
    order      = np.argsort(pixel_id, kind='stable')
    pid_sorted = pixel_id[order]
    pol_sorted = p[order]
    tn_sorted  = t_norm[order]

    grp_starts  = np.concatenate(([0], np.flatnonzero(np.diff(pid_sorted)) + 1))
    active_pids = pid_sorted[grp_starts]           # unique pixel-ids, sorted
    own_sizes   = np.diff(np.append(grp_starts, pid_sorted.size))  # OWN events per active pixel

    pid_pool, pol_pool, tn_pool = _gather_3x3(
        pid_sorted, pol_sorted, tn_sorted, active_pids, height, width)
    if pid_pool.size == 0:
        return None

    gs_pool = np.concatenate(([0], np.flatnonzero(np.diff(pid_pool)) + 1))
    sz_pool = np.diff(np.append(gs_pool, pid_pool.size))
    big = sz_pool >= cfg.min_events
    if getattr(cfg, "min_own_events", 0) > 0:
        # Also require the CENTRE pixel's own events >= min_own_events.
        own_per_group = own_sizes[np.searchsorted(active_pids, pid_pool[gs_pool])]
        big &= own_per_group >= cfg.min_own_events
    if not big.any():
        return None

    keep   = np.repeat(big, sz_pool)
    pid_k  = pid_pool[keep]; pol_k = pol_pool[keep]; tn_k = tn_pool[keep]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_k)) + 1))
    grp_pid = pid_k[starts]
    return pol_k, tn_k, starts, ("pooled", grp_pid // width, grp_pid % width, height, width)


def _detect_rotor_pixels(x, y, p, t_norm, height, width, cfg):
    """3x3-pooled drop-in for utils._detect_rotor_pixels (gather variant).
    prep (CPU gather/gate) -> shared spectral stage (GPU/CPU) -> scatter."""
    prep = _prep_pooled(x, y, p, t_norm, height, width, cfg)
    if prep is None:
        return (np.zeros((height, width), dtype=bool),
                np.zeros((height, width), dtype=np.float32))
    pol_k, tn_k, starts, scatter = prep
    valid, snr = _spectral_valid_snr(pol_k, tn_k, starts, cfg.frequency_bins, cfg)
    return _scatter(valid, snr, scatter)


def _spectral_valid_snr(pol_k, tn_k, starts, K, cfg):
    """Shared spectral acceptance for grouped events (groups = pooled pixels or
    binned blocks): NDFT power -> spectral-flatness gate -> comb-SNR gate.

    Runs on the GPU when CuPy is usable (falling back to CPU per-window on OOM),
    so BOTH the pooled and the pixelize detectors get the GPU speed-up. Returns
    (valid_bool[G], snr[G]); the maths mirrors the CPU path up to float reduction
    order. Per-window GPU/CPU path is recorded in _PATH_STATS."""
    snr = valid = None
    oom_fallback = False
    if _gpu_available():
        try:
            power_g  = _ndft_power_batch_gpu(pol_k, tn_k, starts, K)
            valid_g  = _spectral_flatness_batch_gpu(power_g[1:]) <= cfg.tau_sf
            snr_g    = _comb_snr_batch_gpu(power_g, cfg)
            valid_g &= snr_g >= cfg.tau_comb
            snr      = _cp.asnumpy(snr_g)
            valid    = _cp.asnumpy(valid_g)
            _PATH_STATS["gpu"] += 1
        except (_cp.cuda.memory.OutOfMemoryError,
                _cp.cuda.runtime.CUDARuntimeError):
            # OOM in CuPy's pool (OutOfMemoryError) or inside a kernel such as the
            # merge_sort behind cp.median (CUDARuntimeError) -> free and use CPU.
            _cp.get_default_memory_pool().free_all_blocks()
            snr = valid = None
            oom_fallback = True
    if valid is None:
        _PATH_STATS["cpu_fallback" if oom_fallback else "cpu"] += 1
        power = _ndft_power_batch(None, pol_k, tn_k, starts, K)
        valid = _spectral_flatness_batch(power[1:]) <= cfg.tau_sf
        snr   = _comb_snr_batch(power, cfg)
        valid &= snr >= cfg.tau_comb
    return valid, snr


def _spectral_valid_snr_batched(preps, K, cfg):
    """Run the spectral stage ONCE over several windows' groups (a single GPU
    call) and split the result back per window. ``preps`` is a list of prep
    tuples (pol_k, tn_k, starts, scatter) or None (window with no candidates).
    Returns a list aligned to ``preps`` of (valid, snr) arrays or None.

    Each group is processed independently (segment-wise NDFT reduceat + per-column
    flatness/comb), and concatenating windows only offsets the group start indices
    — so the per-group result is identical to calling _spectral_valid_snr per
    window. Batching amortises the per-call GPU launch/transfer/sync overhead and
    keeps the device fed; peak memory is unchanged (the GPU kernels already chunk
    by events / columns internally)."""
    pol_parts, tn_parts, starts_parts, gcounts = [], [], [], []
    eoff = 0
    for prep in preps:
        if prep is None:
            gcounts.append(0)
            continue
        pol_k, tn_k, st, _scatter = prep
        pol_parts.append(pol_k)
        tn_parts.append(tn_k)
        starts_parts.append(st + eoff)          # group starts are event indices -> offset
        eoff += int(pol_k.size)
        gcounts.append(int(st.size))
    if not pol_parts:
        return [None] * len(preps)

    valid_all, snr_all = _spectral_valid_snr(
        np.concatenate(pol_parts), np.concatenate(tn_parts),
        np.concatenate(starts_parts), K, cfg)

    out, goff = [], 0
    for gc in gcounts:
        if gc == 0:
            out.append(None)
        else:
            out.append((valid_all[goff:goff + gc], snr_all[goff:goff + gc]))
            goff += gc
    return out


def _prep_pixelized(x, y, p, t_norm, height, width, cfg):
    """CPU stage of the pixelize detector: bin events into BxB blocks, apply the
    min_events gate, and return grouped events + scatter geometry (None if nothing
    qualifies). B=1 reduces to per-pixel/baseline grouping."""
    if x.size == 0:
        return None
    B  = max(1, int(getattr(cfg, "pixel_block", 1)))
    Wb = (width  + B - 1) // B
    Hb = (height + B - 1) // B

    bid   = (y // B) * Wb + (x // B)
    order = np.argsort(bid, kind="stable")
    bid_s = bid[order]; pol_s = p[order]; tn_s = t_norm[order]
    gstart = np.concatenate(([0], np.flatnonzero(np.diff(bid_s)) + 1))
    sizes  = np.diff(np.append(gstart, bid_s.size))

    big = sizes >= cfg.min_events
    if not big.any():
        return None
    keep   = np.repeat(big, sizes)
    bid_k  = bid_s[keep]; pol_k = pol_s[keep]; tn_k = tn_s[keep]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bid_k)) + 1))
    grp_bid = bid_k[starts]
    return pol_k, tn_k, starts, ("pixelize", grp_bid // Wb, grp_bid % Wb,
                                 B, Hb, Wb, height, width)


def _detect_rotor_pixels_pixelized(x, y, p, t_norm, height, width, cfg):
    """'Pixelization' detector: one NDFT per non-overlapping BxB block, result
    upscaled to full resolution so _extract_boxes works unchanged. Binning
    concentrates a faint/distant drone's sparse events into one super-pixel, so it
    clears cfg.min_events far more easily than per-pixel detection. Uses the
    shared GPU spectral path (_spectral_valid_snr), like pooling.
    """
    prep = _prep_pixelized(x, y, p, t_norm, height, width, cfg)
    if prep is None:
        return (np.zeros((height, width), dtype=bool),
                np.zeros((height, width), dtype=np.float32))
    pol_k, tn_k, starts, scatter = prep
    valid, snr = _spectral_valid_snr(pol_k, tn_k, starts, cfg.frequency_bins, cfg)
    return _scatter(valid, snr, scatter)
