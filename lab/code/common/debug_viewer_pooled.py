"""
Interactive per-pixel DDHF debugger — 3x3 spatial-pooling variant.

Identical to debug_viewer.py except that every pixel's NDFT is computed
over events from its full 3x3 neighbourhood rather than its own events
alone.  This mirrors main_pooled.py / utils_pooled.py exactly, so the
green mask and detection boxes here match those produced by main_pooled.

Differences from debug_viewer.py
---------------------------------
* compute_pixel_stats  — heatmap (view 3) still shows RAW per-pixel event
  counts so you can see the true event density; the NDFT / SF / comb-SNR
  are computed on the pooled 3x3 events.
* pixel_power_spectrum — clicking a pixel shows the pooled spectrum (all
  events in the 3x3 block), because that is exactly what the detector saw.
* rotor-mask gate      — a pixel is eligible for detection when its 3x3
  POOL has >= min_events (not when the pixel's OWN count does).

Usage:
    python debug_viewer_pooled.py --seq 28 --frame 42
    python debug_viewer_pooled.py --seq 28 --min-events 6 --tau-comb 1.5
    python debug_viewer_pooled.py --seq 28 --mode components   # 8-conn components
    python debug_viewer_pooled.py --seq 28 --mode pool         # 3x3 pooling
    python debug_viewer_pooled.py --seq 28 --mode baseline     # per-pixel
    python debug_viewer_pooled.py --seq 122 \
        --wp ../data/working_points/working_point_bayes_pixelize_track_r80.json

--mode selects the spatial aggregation: pixelize (default, BxB blocks), pool
(3x3), baseline (per-pixel), or components (8-connected components).

--wp replaces the boxes and the green mask with a WORKING POINT's own — replayed
through frame_explorer, so the gates this viewer has no flags for (min_snr_sum,
min_snr_avg, persistence, the coast/confirm/acquire/predict tracker) are applied
and what you see is what the working point scored. Every view, the heatmap and
the click-a-pixel spectrum stay this viewer's own per-pixel computation.
"""
import os
import re
import argparse
import dataclasses
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.widgets import TextBox
from matplotlib.lines import Line2D

plugin_folder = r"C:\Program Files\Prophesee\lib\metavision\hal\plugins"
dll_directories = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_deps", "openeb_bin"),
    r"C:\Program Files\Prophesee\bin",
    r"C:\Program Files\Prophesee\third_party\bin",
    plugin_folder,
    r"C:\Program Files\Prophesee\lib\hdf5\plugin"
]
for directory in dll_directories:
    if os.path.exists(directory):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(directory)
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
os.environ["MV_HAL_PLUGIN_PATH"] = plugin_folder
os.environ["HDF5_PLUGIN_PATH"] = r"C:\Program Files\Prophesee\lib\hdf5\plugin"

from metavision_core.event_io import RawReader as _MvRawReader
from metavision_core.event_io import EventsIterator as _MvEventsIterator
import hf_cases as _hf


class _Hdf5Reader:
    """RawReader-compatible reader for the HF Event/events.hdf5 streams.

    metavision RawReader only opens .raw, so HDF5 cases are served by an
    EventsIterator underneath. Only the surface this codebase uses is provided:
    get_size(), seek_time() and load_delta_t(), with the same (height, width)
    and the same ('x','y','p','t') record layout."""

    def __init__(self, path, max_events=None):
        self.path = str(path)
        self.max_events = max_events
        self._it = None
        self._delta_t = None
        self._pos = 0                                # next timestamp to return
        self._cov_end = 0                            # how far the iterator has read
        self._buf = None                             # events already read, not yet returned
        self._exhausted = False
        probe = _MvEventsIterator(self.path, delta_t=1000)
        self._size = probe.get_size()                # (height, width)
        del probe

    def get_size(self):
        return self._size

    def _open(self, delta_t):
        """EventsIterator only accepts a start_ts that is a multiple of delta_t,
        so it is opened at the aligned window at or before the requested
        position; load_delta_t then trims to the exact range asked for."""
        base = (int(self._pos) // delta_t) * delta_t
        self._it = iter(_MvEventsIterator(self.path, start_ts=int(base),
                                          delta_t=int(delta_t)))
        self._delta_t = delta_t
        self._cov_end = base
        self._buf = None
        self._exhausted = False

    def seek_time(self, t_us):
        """Reposition the stream; the next load_delta_t starts exactly at t_us."""
        self._pos = max(0, int(t_us))
        self._it = None                              # reopened on next read
        self._buf = None
        self._exhausted = False

    def load_delta_t(self, delta_t):
        """Events in [pos, pos+delta_t), advancing the cursor - the same
        contract as RawReader.load_delta_t, for any (unaligned) position."""
        delta_t = int(delta_t)
        if self._it is None or self._delta_t != delta_t:
            self._open(delta_t)
        lo, hi = self._pos, self._pos + delta_t
        parts = []
        if self._buf is not None and self._buf.size:
            parts.append(self._buf)
        self._buf = None
        while not self._exhausted and self._cov_end < hi:
            try:
                chunk = next(self._it)
            except StopIteration:
                self._exhausted = True
                break
            self._cov_end += delta_t
            if chunk.size:
                parts.append(chunk)
        evs = (parts[0] if len(parts) == 1 else
               np.concatenate(parts) if parts else _np_empty_events())
        out = evs[(evs["t"] >= lo) & (evs["t"] < hi)]
        rest = evs[evs["t"] >= hi]                   # keep for the next call
        self._buf = rest if rest.size else None
        self._pos = hi
        return out


def _np_empty_events():
    return np.empty(0, dtype=[("x", "<u2"), ("y", "<u2"),
                              ("p", "<i2"), ("t", "<i8")])


def RawReader(path, max_events=None, **kw):
    """Open a case's event stream: the metavision RawReader for .raw, the
    EventsIterator-backed shim for the HF .hdf5. Same call sites either way."""
    if _hf.is_hdf5(path):
        return _Hdf5Reader(path, max_events=max_events)
    if max_events is None:
        return _MvRawReader(str(path), **kw)
    return _MvRawReader(str(path), max_events=max_events, **kw)
from utils import (
    DDHFConfig,
    BoundingBox,
    _ndft_power_batch,
    _spectral_flatness_batch,
    _comb_snr_batch,
    _normalise_polarity,
    _normalise_timestamps,
    rgb_frame_seconds,
)
try:
    from scipy.ndimage import label as _scipy_label
    def _connected_components(active):
        return _scipy_label(active, structure=np.ones((3, 3), dtype=int))
except Exception:                                            # pragma: no cover
    import cv2 as _cv2
    def _connected_components(active):
        n, labels = _cv2.connectedComponents(active.astype(np.uint8), connectivity=8)
        return labels, n - 1
# Use the pooled box extractor (it applies min_rotor_pixels) so the viewer's
# boxes match ddhf_eval_pooled exactly.
from utils_pooled import _gather_3x3, _extract_boxes
# Optional GPU path (matches ddhf_eval_pooled's default fp32 detection exactly).
from utils_pooled import (
    _cp,
    _gpu_available,
    _ndft_power_batch_gpu,
    _spectral_flatness_batch_gpu,
    _comb_snr_batch_gpu,
)
# Shared detector preset + CLI flags, so the viewer's parameters and defaults
# are identical to batch_ddhf_eval_pooled (single source of truth).
from ddhf_config import build_cfg, add_cfg_cli_args, cfg_overrides_from_args

_USE_GPU   = False  # set by --gpu in main(); when True compute_pixel_stats runs on GPU
# Detector mode (set in main() from --mode). The three booleans below are derived
# from the chosen mode via MODE_FLAGS. _COMPONENTS takes precedence in detection;
# otherwise pixelize takes precedence over pooling, and with neither it's the
# baseline per-pixel detector (== block size 1).
_COMPONENTS = False # aggregate by 8-connected components, not blocks
_PIXELIZE  = True   # non-overlapping BxB blocks, one NDFT per block
_POOLING   = True   # 3x3 spatial pooling (only when not _PIXELIZE)

# --mode -> (_COMPONENTS, _PIXELIZE, _POOLING). Each triple reproduces the exact
# combination the old --pixelize/--no-pixelize/--no-pooling/--components flags set.
MODE_FLAGS = {
    "pixelize":   (False, True,  True),   # BxB blocks (old default)
    "pool":       (False, False, True),   # 3x3 pooling (old --no-pixelize)
    "baseline":   (False, False, False),  # per-pixel (old --no-pooling)
    "components": (True,  True,  True),    # connected components (old --components)
}
MODES = tuple(MODE_FLAGS)

FOV_BOX_COLOR = (255, 255, 0)
FOV_BOX_WIDTH = 3

DRONE_BANDS = {
    "mini2":   ((280.0, 340.0),),
    "mini3":   ((280.0, 340.0),),
    "tello":   ((450.0, 700.0),),
    "cineape": ((800.0, 1200.0),),
    "air75":   ((900.0, 1500.0),),
    "all":     ((280.0, 340.0), (450.0, 700.0), (800.0, 1500.0)),
    # Lab bench target, not a drone: a 7-blade Luxor FS-45 pedestal fan,
    # measured at 99/110/120 Hz blade-pass on low/medium/high (shaft 14.1 Hz,
    # 849 rpm on low). Far below every drone band, so it needs its own entry -
    # with "all" the comb search tests no fundamental at all and the fan is
    # undetectable however the rest of the detector is tuned. Wide enough to
    # cover the ~1.5% the speed drifts as the motor warms up.
    "fan":     ((80.0, 140.0),),
}
DDHF_COLOR    = (0, 255, 0)
DET_BOX_COLOR = (255, 0, 255)        # fallback when not colouring by GT match
DET_BOX_WIDTH = 3
GT_BOX_COLOR  = (0, 255, 255)
GT_BOX_WIDTH  = 3

# Per-box / per-frame match colouring (matches the eval's --match center criterion:
# a detection is "on a drone" if its centre is within scale × GT-box-half-diagonal
# of some GT box centre).
DET_TP_COLOR  = (0, 220, 0)          # detection near a GT  -> green
DET_OFF_COLOR = (235, 0, 0)          # GT present but this detection is near no GT -> red
DET_FP_COLOR  = (255, 150, 0)        # detection on a no-GT frame (false positive) -> orange
GT_MISS_COLOR = (80, 140, 255)       # GT with no nearby detection (false negative) -> blue

VIEW_PLAIN, VIEW_EVENTS, VIEW_HEATMAP, VIEW_POLARITY, VIEW_EVENTFRAME = 1, 2, 3, 4, 5
VIEW_GTBOX = 9                                            # 'c' / '9': GT-box pooled DDHF
N_VIEWS = 5                                              # 'v' cycles 1..5; view 9 is reached via 9/c
VIEW_NAMES = {
    VIEW_PLAIN:      "plain RGB",
    VIEW_EVENTS:     "RGB + events",
    VIEW_HEATMAP:    "event-count heatmap (RAW, not pooled)",
    VIEW_POLARITY:   "polarity (+ green / - magenta / both white)",
    VIEW_EVENTFRAME: "dataset event frame + GT & detection boxes",
    VIEW_GTBOX:      "GT-box pooled DDHF (yellow events + their transform)",
}
GTBOX_EVENT_COLOR = (255, 255, 0)                        # yellow: events inside the GT box
GTBOX_DIM_COLOR   = (90, 90, 90)                         # other events dimmed in this view
POL_POS_COLOR  = (0, 255, 0)
POL_NEG_COLOR  = (255, 0, 255)
POL_BOTH_COLOR = (255, 255, 255)
HEATMAP_LEVELS = (
    (1,  (140, 140, 140)),
    (6,  (40, 120, 255)),
    (15, (255, 165, 0)),
    (30, (255, 40, 40)),
)


# ── Core stats (CHANGED from debug_viewer.py) ────────────────────────────────

def compute_pixel_stats(evs, ev_height, ev_width, cfg):
    """
    Two-pass pixel statistics.

    Pass 1 — RAW count_map: each pixel's own event count (heatmap view).
    Pass 2 — NDFT/SF/comb-SNR on grouped events, only on groups with
              >= min_events:
        * pool (--mode pool): each pixel's full 3x3 neighbourhood (overlapping),
                   one NDFT per active pixel  (_gather_3x3) — matches main_pooled.
        * pixelize (--mode pixelize): non-overlapping cfg.pixel_block x pixel_block
                   blocks, ONE NDFT per occupied block; the block's SF/SNR/f0 are
                   broadcast to its whole BxB footprint, matching the eval's
                   pixelize detector (which upscales the per-block result).

    A pixel's SF/SNR are NaN when its pool/block has < min_events.
    """
    count_map = np.zeros((ev_height, ev_width), dtype=np.int32)
    sf_map    = np.full((ev_height, ev_width), np.nan)
    snr_map   = np.full((ev_height, ev_width), np.nan)
    f0_map    = np.zeros((ev_height, ev_width), dtype=np.float64)
    if evs.size == 0:
        return count_map, sf_map, snr_map, f0_map

    x     = evs['x'].astype(np.int64)
    y     = evs['y'].astype(np.int64)
    pol   = _normalise_polarity(evs['p'])
    t_norm = _normalise_timestamps(evs['t'].astype(np.float64),
                                   window_span=cfg.window_ms * 1000.0)

    # Pass 1: raw event count per pixel (for heatmap display).
    pid_raw   = y * ev_width + x
    ord_raw   = np.argsort(pid_raw, kind="stable")
    pid_s_raw = pid_raw[ord_raw]
    gs_raw    = np.concatenate(([0], np.flatnonzero(np.diff(pid_s_raw)) + 1))
    sz_raw    = np.diff(np.append(gs_raw, pid_s_raw.size))
    grp_raw   = pid_s_raw[gs_raw]
    count_map[grp_raw // ev_width, grp_raw % ev_width] = sz_raw

    # Pass 2: build per-group events. Blocks for pixelize (B=pixel_block) AND for
    # the baseline per-pixel detector (B=1); 3x3 gather for pooling.
    if _PIXELIZE or not _POOLING:
        B  = max(1, int(cfg.pixel_block)) if _PIXELIZE else 1
        Wb = (ev_width  + B - 1) // B
        Hb = (ev_height + B - 1) // B
        pid_g = (y // B) * Wb + (x // B)
        order = np.argsort(pid_g, kind="stable")
        pid_s = pid_g[order]; pol_s = pol[order]; tn_s = t_norm[order]
        gs = np.concatenate(([0], np.flatnonzero(np.diff(pid_s)) + 1))
        sz = np.diff(np.append(gs, pid_s.size))
        big = sz >= cfg.min_events
        if not big.any():
            return count_map, sf_map, snr_map, f0_map
        keep  = np.repeat(big, sz)
        pid_k = pid_s[keep]; pol_k = pol_s[keep]; tn_k = tn_s[keep]
    else:
        # 3x3-gather pooling: searchsorted over the sorted array for each of the
        # 9 offsets; only pixels with >= 1 own event are ever candidates.
        pid_pool, pol_pool, tn_pool = _gather_3x3(
            pid_s_raw, pol[ord_raw], t_norm[ord_raw], grp_raw, ev_height, ev_width)
        if pid_pool.size == 0:
            return count_map, sf_map, snr_map, f0_map
        gs_p = np.concatenate(([0], np.flatnonzero(np.diff(pid_pool)) + 1))
        sz_p = np.diff(np.append(gs_p, pid_pool.size))
        big = sz_p >= cfg.min_events
        if getattr(cfg, "min_own_events", 0) > 0:
            # also require the centre pixel's own events >= min_own_events
            # (grp_raw are the sorted active pids, sz_raw their own-event counts).
            own_per_group = sz_raw[np.searchsorted(grp_raw, pid_pool[gs_p])]
            big &= own_per_group >= cfg.min_own_events
        if not big.any():
            return count_map, sf_map, snr_map, f0_map
        keep  = np.repeat(big, sz_p)
        pid_k = pid_pool[keep]; pol_k = pol_pool[keep]; tn_k = tn_pool[keep]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_k)) + 1))

    # Optional de-DC (off by default): subtract each group's mean polarity before
    # the NDFT so a polarity imbalance (net brightness change) doesn't dump energy
    # into the low-frequency bins and pull the comb search toward ~30 Hz. Enable
    # via cfg.detrend_polarity=True.
    if getattr(cfg, "detrend_polarity", False) and pol_k.size:
        sizes_k = np.diff(np.append(starts, pol_k.size))
        grp_mean = np.add.reduceat(pol_k, starts) / sizes_k
        pol_k = pol_k - np.repeat(grp_mean, sizes_k)

    K = cfg.frequency_bins
    sf = snr = omega = None
    if _USE_GPU and _gpu_available():
        # Exact GPU detection path used by ddhf_eval_pooled (fp32 by default, set
        # DDHF_GPU_FP32=0 for fp64). Falls back to CPU for this frame on OOM.
        try:
            power_g    = _ndft_power_batch_gpu(pol_k, tn_k, starts, K)
            sf_g       = _spectral_flatness_batch_gpu(power_g[1:])
            snr_g, om_g = _comb_snr_batch_gpu(power_g, cfg, return_omega=True)
            sf    = _cp.asnumpy(sf_g)
            snr   = _cp.asnumpy(snr_g)
            omega = _cp.asnumpy(om_g)
        except (_cp.cuda.memory.OutOfMemoryError,
                _cp.cuda.runtime.CUDARuntimeError):
            _cp.get_default_memory_pool().free_all_blocks()
            sf = snr = omega = None

    if sf is None:
        power      = _ndft_power_batch(None, pol_k, tn_k, starts, K)
        sf         = _spectral_flatness_batch(power[1:])
        snr, omega = _comb_snr_batch(power, cfg, return_omega=True)

    bpid = pid_k[starts]
    f0   = omega * 1000.0 / cfg.window_ms
    if _PIXELIZE or not _POOLING:
        # per-block grids, then upscale each block to its BxB footprint (B=1 for
        # the baseline per-pixel detector, where upscaling is a no-op).
        sf_g  = np.full((Hb, Wb), np.nan)
        snr_g = np.full((Hb, Wb), np.nan)
        f0_g  = np.zeros((Hb, Wb), dtype=np.float64)
        bby = bpid // Wb; bbx = bpid % Wb
        sf_g[bby, bbx] = sf; snr_g[bby, bbx] = snr; f0_g[bby, bbx] = f0
        sf_map  = np.repeat(np.repeat(sf_g,  B, 0), B, 1)[:ev_height, :ev_width]
        snr_map = np.repeat(np.repeat(snr_g, B, 0), B, 1)[:ev_height, :ev_width]
        f0_map  = np.repeat(np.repeat(f0_g,  B, 0), B, 1)[:ev_height, :ev_width]
    else:
        bpy = bpid // ev_width; bpx = bpid % ev_width
        sf_map[bpy, bpx] = sf; snr_map[bpy, bpx] = snr; f0_map[bpy, bpx] = f0
    return count_map, sf_map, snr_map, f0_map


def pixel_power_spectrum(evs, xe, ye, cfg):
    """
    Power spectrum for a clicked pixel, over exactly the events the detector
    grouped: the 3x3 neighbourhood centred on (xe, ye) in pooling mode, or the
    cfg.pixel_block x pixel_block block that contains (xe, ye) in pixelize mode.
    """
    if evs.size == 0:
        return None, None
    xi = evs["x"].astype(np.int64); yi = evs["y"].astype(np.int64)
    if _PIXELIZE or not _POOLING:
        B = max(1, int(cfg.pixel_block)) if _PIXELIZE else 1
        sel = ((xi // B == xe // B) & (yi // B == ye // B))   # block (B=1 -> own pixel)
    else:
        sel = ((np.abs(xi - xe) <= 1) & (np.abs(yi - ye) <= 1))
    if not sel.any():
        return None, None
    pol    = _normalise_polarity(evs["p"])
    t_norm = _normalise_timestamps(evs["t"].astype(np.float64),
                                   window_span=cfg.window_ms * 1000.0)
    pol_p = pol[sel]
    tn_p  = t_norm[sel]
    k     = np.arange(cfg.frequency_bins, dtype=np.float64)
    F     = (pol_p[None, :] * np.exp(1j * np.outer(k, tn_p))).sum(axis=1)
    power = np.abs(F) ** 2
    return power, k * 1000.0 / cfg.window_ms


# ── Everything below is identical to debug_viewer.py ─────────────────────────

def best_comb(power, cfg):
    K = len(power)
    best_omega, best_snr = 0, 0.0
    bands  = cfg.rotor_bands_hz
    bin_hz = 1000.0 / cfg.window_ms
    for omega in range(1, K // cfg.num_harmonics + 1):
        if bands and not any(lo <= omega * bin_hz <= hi for lo, hi in bands):
            continue
        harmonics = [m * omega for m in range(1, cfg.num_harmonics + 1)
                     if m * omega < K]
        if len(harmonics) < cfg.min_harmonics:
            continue
        hmask = np.zeros(K, dtype=bool)
        peaks = []
        for h in harmonics:
            lo = max(0, h - cfg.harmonic_half_width)
            hi = min(K - 1, h + cfg.harmonic_half_width)
            peaks.append(power[lo:hi + 1].max())
            hmask[lo:hi + 1] = True
        nmask = ~hmask; nmask[0] = False
        nbh = getattr(cfg, "noise_band_harmonics", 0.0)
        if nbh > 0:                                  # local noise floor (matches detector)
            band_hi = min(K, int(nbh * harmonics[-1]) + cfg.harmonic_half_width + 1)
            nmask[band_hi:] = False
        noise = power[nmask]
        if noise.size == 0:
            continue
        med = np.median(noise)
        mff = getattr(cfg, "snr_median_floor_frac", 0.0)   # optional median floor (off by default)
        if mff > 0:
            med = max(med, mff * float(power[1:].mean()))
        if med < 1e-10:
            continue
        # "Harmonics present" gate (matches utils._comb_snr_batch): disqualify ω
        # unless enough harmonics individually rise above the noise floor.
        mph = getattr(cfg, "min_present_harmonics", 0)
        if mph > 0:
            factor = getattr(cfg, "harmonic_present_factor", 2.0)
            if sum(1 for pk in peaks if pk >= factor * med) < mph:
                continue
        snr = float(np.mean(peaks) / med)
        if snr > best_snr:
            best_snr, best_omega = snr, omega
    return best_omega, best_snr


def detect_components_mask(evs, H, W, cfg):
    """Connected-component aggregation (used when --components is set): 8-connect the
    per-window event activity, pool each component's events into one signal, run the
    DDHF spectral gate (size>=min_rotor_pixels, events>=min_events, SF<=tau_sf,
    comb-SNR>=tau_comb) on the aggregate, and turn each passing component into a box.
    Returns (rotor_mask, [BoundingBox], labels) — labels is the (H,W) component-id
    array (0=background) so callers can recover which component a pixel belongs to.
    Connected components are on the EVENT data."""
    rmask = np.zeros((H, W), dtype=bool)
    if evs.size == 0:
        return rmask, [], None
    x = evs["x"].astype(np.int64); y = evs["y"].astype(np.int64)
    pol = _normalise_polarity(evs["p"])
    tn  = _normalise_timestamps(evs["t"].astype(np.float64), window_span=cfg.window_ms * 1000.0)
    count  = np.bincount(y * W + x, minlength=H * W).reshape(H, W)
    labels, n = _connected_components(count > 0)
    if n == 0:
        return rmask, [], labels
    ev_label   = labels[y, x]
    ev_counts  = np.bincount(ev_label, minlength=n + 1)
    comp_sizes = np.bincount(labels.ravel(), minlength=n + 1)
    cands = [c for c in range(1, n + 1)
             if ev_counts[c] >= cfg.min_events and comp_sizes[c] >= cfg.min_rotor_pixels]
    if not cands:
        return rmask, [], labels
    karr  = np.arange(cfg.frequency_bins, dtype=np.float64)
    boxes = []
    for comp in cands:
        sel = ev_label == comp
        F = (pol[sel][None, :] * np.exp(1j * np.outer(karr, tn[sel]))).sum(axis=1)
        power = np.abs(F) ** 2
        pnd = power[1:] + 1e-12
        sf  = float(np.exp(np.mean(np.log(pnd))) / np.mean(pnd))
        _omega, snr = best_comb(power, cfg)
        if sf <= cfg.tau_sf and snr >= cfg.tau_comb:
            ys, xs = np.where(labels == comp)
            rmask[ys, xs] = True
            boxes.append(BoundingBox(int(xs.min()), int(ys.min()),
                                     int(xs.max()), int(ys.max()), score=float(snr)))
    return rmask, boxes, labels


def component_power_spectrum(evs, labels, xe, ye, cfg):
    """Power spectrum of the WHOLE connected component containing (xe, ye): pools
    every event in that component, so every pixel of a component yields the IDENTICAL
    spectrum — exactly what the --components detector tests. Returns
    (power, freqs, n_events), or (None, None, 0) if the pixel is in no component."""
    if evs.size == 0 or labels is None:
        return None, None, 0
    if not (0 <= ye < labels.shape[0] and 0 <= xe < labels.shape[1]):
        return None, None, 0
    comp = int(labels[ye, xe])
    if comp == 0:
        return None, None, 0                              # inactive (no-event) pixel
    x = evs["x"].astype(np.int64); y = evs["y"].astype(np.int64)
    sel = labels[y, x] == comp
    nsel = int(sel.sum())
    if nsel == 0:
        return None, None, 0
    pol = _normalise_polarity(evs["p"])[sel]
    tn  = _normalise_timestamps(evs["t"].astype(np.float64),
                                window_span=cfg.window_ms * 1000.0)[sel]
    k = np.arange(cfg.frequency_bins, dtype=np.float64)
    F = (pol[None, :] * np.exp(1j * np.outer(k, tn))).sum(axis=1)
    return np.abs(F) ** 2, k * 1000.0 / cfg.window_ms, nsel


def print_f0_consensus(f0_map, rotor_mask):
    f0 = f0_map[rotor_mask]; f0 = f0[f0 > 0]
    if f0.size == 0:
        print("  f0 consensus: no flagged pixels"); return
    vals, cnts = np.unique(f0, return_counts=True)
    top = np.argsort(cnts)[::-1][:6]
    frac_top2 = cnts[np.argsort(cnts)[::-1][:2]].sum() / f0.size
    verdict = ("LOOKS LIKE ROTOR(S)" if frac_top2 >= 0.5 else
               "SCATTERED -- mostly noise/burst flags")
    print(f"  f0 consensus over {f0.size} green px: "
          + "  ".join(f"{vals[i]:.0f}Hz x{cnts[i]}" for i in top)
          + f"   [top-2 modes = {frac_top2*100:.0f}% -> {verdict}]")


def _project_box_to_rgb(box, M, W, H):
    corners = np.array([
        [box.x1, box.y1, 1.0], [box.x2, box.y1, 1.0],
        [box.x2, box.y2, 1.0], [box.x1, box.y2, 1.0],
    ], dtype=np.float32).T
    pts = (M @ corners).T
    x1 = int(np.clip(np.floor(pts[:, 0].min()), 0, W - 1))
    y1 = int(np.clip(np.floor(pts[:, 1].min()), 0, H - 1))
    x2 = int(np.clip(np.ceil(pts[:, 0].max()),  0, W - 1))
    y2 = int(np.clip(np.ceil(pts[:, 1].max()),  0, H - 1))
    return x1, y1, x2, y2


def _load_gt_boxes(path):
    if not path or not os.path.exists(path):
        return None, None
    times, boxes = [], []
    with open(path) as f:
        for line in f:
            t_str, _, rest = line.strip().partition(":")
            try:
                t    = float(t_str)
                vals = [float(v) for v in rest.replace(",", " ").split()[:4]]
            except ValueError:
                continue
            if len(vals) >= 4:
                times.append(t); boxes.append(vals[:4])
    if not times:
        return None, None
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(boxes, dtype=np.float64)[order]


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _gt_allowed_dist(gt_box, scale, floor_px):
    """Centre-match tolerance for this GT box: scale × its half-diagonal, floored
    at floor_px (identical rule to ddhf_eval_pooled._gt_allowed_dist)."""
    w = gt_box[2] - gt_box[0]
    h = gt_box[3] - gt_box[1]
    return max(floor_px, scale * 0.5 * (w * w + h * h) ** 0.5)


def _det_near_any_gt(det_box, gt_rects, scale, floor_px):
    """True if det_box's centre is within the allowed distance of ANY GT centre."""
    dcx, dcy = _box_center(det_box)
    for g in gt_rects:
        gcx, gcy = _box_center(g)
        if (dcx - gcx) ** 2 + (dcy - gcy) ** 2 <= _gt_allowed_dist(g, scale, floor_px) ** 2:
            return True
    return False


def _draw_gt(draw, gt_rects, det_rects=(), scale=1.0, floor_px=0.0, match_color=False, evs=None):
    ex = ey = None
    if evs is not None and getattr(evs, "size", 0):
        ex = evs["x"]; ey = evs["y"]                      # event coords (== box coords at M=identity)
    for gt in gt_rects:
        x1, y1, x2, y2 = gt
        # FN highlight: a GT with no detection centre near it is a miss.
        miss    = match_color and not any(
            _det_near_any_gt(d, [gt], scale, floor_px) for d in det_rects)
        outline = GT_MISS_COLOR if miss else GT_BOX_COLOR
        draw.rectangle([x1,y1,x2,y2], outline=(0,0,0), width=GT_BOX_WIDTH+3)
        draw.rectangle([x1,y1,x2,y2], outline=outline, width=GT_BOX_WIDTH)
        best  = max((_iou(gt, d) for d in det_rects), default=0.0)
        label = (("MISS " if miss else "GT ") + (f"IoU={best:.2f}" if det_rects else "")).strip()
        if miss and ex is not None:                       # how many events the detector had in the box
            nev = int(((ex >= x1) & (ex <= x2) & (ey >= y1) & (ey <= y2)).sum())
            label = f"{label}  ev={nev}"
        draw.text((x1+2, max(0, y1-13)), label, fill=outline)


def _draw_fov(draw, fov_pts):
    for i in range(len(fov_pts) - 1):
        draw.line([fov_pts[i], fov_pts[i+1]], fill=FOV_BOX_COLOR, width=FOV_BOX_WIDTH)


def _draw_boxes(draw, boxes, M, W, H, gt_rects=None, scale=1.0, floor_px=0.0):
    """Draw detected boxes. When gt_rects is provided (a list, possibly empty),
    each box is coloured by the center-distance match: green if its centre is near
    a GT (within scale × GT half-diag), red if a GT is present but this box is near
    none, orange if there's no GT in the frame at all (false positive). gt_rects=None
    keeps the legacy single colour."""
    for b in (boxes or []):
        bx1, by1, bx2, by2 = _project_box_to_rgb(b, M, W, H)
        if gt_rects is None:
            color = DET_BOX_COLOR
        elif not gt_rects:                                   # no GT this frame -> FP
            color = DET_FP_COLOR
        elif _det_near_any_gt((bx1, by1, bx2, by2), gt_rects, scale, floor_px):
            color = DET_TP_COLOR
        else:
            color = DET_OFF_COLOR
        draw.rectangle([bx1,by1,bx2,by2], outline=color, width=DET_BOX_WIDTH)
        if b.score:                                      # --wp boxes carry no SNR
            draw.text((bx1+2, max(0, by1-12)), f"snr={b.score:.1f}", fill=color)


def render_plain(base_frame):
    return np.copy(base_frame)


def render_heatmap(base_frame, count_map, M, fov_pts):
    rgb = (np.copy(base_frame).astype(np.float32) * 0.35).astype(np.uint8)
    H, W = rgb.shape[:2]
    ys, xs = np.nonzero(count_map)
    if xs.size:
        counts = count_map[ys, xs]
        colors = np.empty((xs.size, 3), dtype=np.uint8)
        for thr, col in HEATMAP_LEVELS:
            colors[counts >= thr] = col
        src = np.stack([xs.astype(np.float32), ys.astype(np.float32),
                        np.ones(xs.size, dtype=np.float32)], axis=0)
        dst = (M @ src).T
        xr = np.rint(dst[:, 0]).astype(np.int32)
        yr = np.rint(dst[:, 1]).astype(np.int32)
        m  = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        rgb[yr[m], xr[m]] = colors[m]
    img  = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)
    _draw_fov(draw, fov_pts)
    return np.array(img)


def render_polarity(base_frame, evs, M, fov_pts):
    rgb = (np.copy(base_frame).astype(np.float32) * 0.45).astype(np.uint8)
    H, W = rgb.shape[:2]
    if evs.size > 0:
        ones = np.ones(evs.size, dtype=np.float32)
        src  = np.stack([evs['x'].astype(np.float32),
                         evs['y'].astype(np.float32), ones], axis=0)
        dst  = (M @ src).T
        xr   = np.rint(dst[:, 0]).astype(np.int32)
        yr   = np.rint(dst[:, 1]).astype(np.int32)
        mask = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        xr_m = xr[mask]; yr_m = yr[mask]; on = evs['p'][mask] == 1
        has_pos = np.zeros((H, W), dtype=bool)
        has_neg = np.zeros((H, W), dtype=bool)
        has_pos[yr_m[on],  xr_m[on]]  = True
        has_neg[yr_m[~on], xr_m[~on]] = True
        both = has_pos & has_neg
        rgb[has_pos & ~both] = POL_POS_COLOR
        rgb[has_neg & ~both] = POL_NEG_COLOR
        rgb[both]            = POL_BOTH_COLOR
    img  = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)
    _draw_fov(draw, fov_pts)
    return np.array(img)


def render_event_frame(event_img, base_frame, M, boxes=None,
                       gt_rects=None, scale=1.0, floor_px=0.0):
    if event_img is None:
        rgb = np.zeros_like(base_frame)
    elif event_img.shape[:2] != base_frame.shape[:2]:
        rgb = np.asarray(Image.fromarray(event_img).resize(
            (base_frame.shape[1], base_frame.shape[0])))
    else:
        rgb = np.copy(event_img)
    H, W = rgb.shape[:2]
    img  = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)
    _draw_boxes(draw, boxes, M, W, H, gt_rects, scale, floor_px)
    return np.array(img)


def render_overlay(base_frame, evs, M, fov_pts, rotor_mask, boxes=None,
                   gt_rects=None, scale=1.0, floor_px=0.0):
    rgb  = np.copy(base_frame)
    H, W = rgb.shape[:2]
    if evs.size > 0:
        ones = np.ones(evs.size, dtype=np.float32)
        src  = np.stack([evs['x'].astype(np.float32),
                         evs['y'].astype(np.float32), ones], axis=0)
        dst  = (M @ src).T
        xr   = np.rint(dst[:, 0]).astype(np.int32)
        yr   = np.rint(dst[:, 1]).astype(np.int32)
        mask = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        xr_m = xr[mask]; yr_m = yr[mask]; p_m = evs['p'][mask]
        on   = p_m == 1
        rgb[yr_m[on],  xr_m[on]]  = [255, 0, 0]
        rgb[yr_m[~on], xr_m[~on]] = [0, 0, 255]
        if rotor_mask is not None and rotor_mask.any():
            is_rotor = rotor_mask[evs['y'].astype(np.int64),
                                  evs['x'].astype(np.int64)]
            sel = is_rotor & mask
            if sel.any():
                rgb[yr[sel], xr[sel]] = DDHF_COLOR
    img  = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)
    _draw_fov(draw, fov_pts)
    _draw_boxes(draw, boxes, M, W, H, gt_rects, scale, floor_px)
    return np.array(img)


def _events_in_gt_mask(evs, gt_rects):
    """Boolean mask of events whose (x,y) is inside ANY GT box (event coords)."""
    if evs.size == 0 or not gt_rects:
        return np.zeros(evs.size, dtype=bool)
    x = evs["x"].astype(np.int64); y = evs["y"].astype(np.int64)
    m = np.zeros(evs.size, dtype=bool)
    for (x1, y1, x2, y2) in gt_rects:
        m |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return m


def gtbox_pool_spectrum(evs, gt_rects, cfg):
    """Pool ALL events inside the GT box(es) and return (power, freqs, n_events) —
    the spectrum the GT-box view / seq5_gtbox_ddhf computes."""
    m = _events_in_gt_mask(evs, gt_rects)
    n = int(m.sum())
    if n == 0:
        return None, None, 0
    pol = _normalise_polarity(evs["p"])[m]
    tn  = _normalise_timestamps(evs["t"].astype(np.float64),
                                window_span=cfg.window_ms * 1000.0)[m]
    k = np.arange(cfg.frequency_bins, dtype=np.float64)
    F = (pol[None, :] * np.exp(1j * np.outer(k, tn))).sum(axis=1)
    return np.abs(F) ** 2, k * 1000.0 / cfg.window_ms, n


def render_gtbox(base_frame, evs, M, gt_rects):
    """VIEW_GTBOX overlay: events inside the GT box(es) drawn YELLOW, all other
    events dimmed, GT boxes outlined. The 'these specific pixels' the view pools."""
    rgb  = np.copy(base_frame)
    H, W = rgb.shape[:2]
    if evs.size > 0:
        ones = np.ones(evs.size, dtype=np.float32)
        src  = np.stack([evs['x'].astype(np.float32), evs['y'].astype(np.float32), ones], axis=0)
        dst  = (M @ src).T
        xr   = np.rint(dst[:, 0]).astype(np.int32); yr = np.rint(dst[:, 1]).astype(np.int32)
        inb  = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        ingt = _events_in_gt_mask(evs, gt_rects)
        out  = inb & ~ingt; ins = inb & ingt
        rgb[yr[out], xr[out]] = GTBOX_DIM_COLOR
        rgb[yr[ins], xr[ins]] = GTBOX_EVENT_COLOR        # yellow GT-box events
    img  = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in (gt_rects or []):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=GT_BOX_WIDTH + 3)
        draw.rectangle([x1, y1, x2, y2], outline=GT_BOX_COLOR, width=GT_BOX_WIDTH)
    return np.array(img)


def invert_affine_2x3(M):
    A     = M[:, :2]; b = M[:, 2]
    A_inv = np.linalg.inv(A)
    return np.hstack([A_inv, (-A_inv @ b).reshape(2, 1)]).astype(np.float32)


def _track_features(seq, mode):
    """Per-window candidate pixels for the tracker, in the SELECTED mode at
    stride 1 — read from utility/frame_data/frames_<seq>_<mode>.{npz,csv.gz},
    the very table the optimizer scores working points on. None when that case
    hasn't been cached (caller then falls back to its own precompute)."""
    name = os.path.basename(os.path.normpath(seq))
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "frame_data", f"frames_{name}_{mode}")
    cols = ("frame", "x", "y", "n_events", "blob_size", "sf", "snr")
    if os.path.exists(base + ".npz"):
        z = np.load(base + ".npz")
        d = {c: z[c] for c in cols}
    elif os.path.exists(base + ".csv.gz"):
        import pandas as pd
        df = pd.read_csv(base + ".csv.gz")
        d = {c: df[c].to_numpy() for c in cols}
    else:
        return None
    fr = d["frame"].astype(np.int64)
    order = np.argsort(fr, kind="stable")                # frame-sorted slices
    fr = fr[order]
    uniq, starts, counts = np.unique(fr, return_index=True, return_counts=True)
    return dict(x=d["x"][order].astype(np.int64), y=d["y"][order].astype(np.int64),
                gc=d["n_events"][order], sf=d["sf"][order], snr=d["snr"][order],
                uniq=uniq, starts=starts, counts=counts, stride=1)


def _wp_arg_defaults(path):
    """working_point_*.json -> argparse defaults for the flags this viewer already
    has (the pixel/box gates + --mode), so the title, the pixel maps and the
    spectrum panel show the SAME thresholds the working point was scored at.
    The gates it has no flag for (min_snr_sum / min_snr_avg / persistence / the
    coast-confirm-acquire-predict tracker) are not approximated here — they are
    applied by replaying frame_explorer in _wp_replay()."""
    import json
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    p = meta.get("params", {})
    d = {}
    for key, cast in (("min_events", int), ("tau_sf", float), ("tau_comb", float),
                      ("min_rotor_pixels", int), ("min_box_area", int),
                      ("alpha", float)):
        if key in p:
            d[key] = cast(round(p[key])) if cast is int else cast(p[key])
    if meta.get("mode"):
        d["mode"] = meta["mode"]
    return d


def _wp_replay(wp_path, seq, data_dir=None):
    """Replay a working point through frame_explorer — the EXACT pipeline its
    scores came from (cached features + _track_pass/compute_all + persistence +
    skip_start), the same call bayes_working_point._case_row scores on.

    Returns (dets, pass_by_frame, meta): dets {frame: [(x1,y1,x2,y2), ...]} and
    pass_by_frame {frame: (xs, ys)} for the pixels that survived every gate.
    Needs utility/frame_data/frames_<seq>_<mode>.npz (build it with
    utility/frame_features.py)."""
    import json
    import sys
    util = os.path.dirname(os.path.abspath(__file__))   # lab/code/common is flat
    if util not in sys.path:
        sys.path.insert(0, util)
    import frame_explorer as FE
    import ddhf_stats_explorer as EX

    with open(wp_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    mode = meta.get("mode", "baseline")
    name = os.path.basename(os.path.normpath(seq))
    D = FE.load_D(name, mode, data_dir or FE.DEFAULT_DATA)
    # The FULL parameter set: pixel gates, component/box gates, persistence and
    # the tracker's shape. Keys absent here silently fall back to _track_pass's
    # "off" defaults, which is exactly what makes an approximate replay disagree.
    params = dict(meta.get("params", {}),
                  tracking=bool(meta.get("track", False)),
                  track_sigma=float(meta.get("track_sigma", 20.0)),
                  match=meta.get("match", "center"),
                  skip_start=bool(meta.get("skip_start", False)))
    res, dets, _caches, _comps = FE._compute_job(D, mode, params)
    stage = res["stage"]
    # With skip_start the score ignores everything before the first GT sighting,
    # and _compute_job has already dropped those detections — keep the boundary so
    # the viewer can label that stretch instead of just showing an empty frame.
    meta["_first_gt_frame"] = (FE.first_gt_frame(D) if params["skip_start"] else None)

    uf = D["uniq_frames"]; starts = D["frame_starts"]; counts = D["frame_counts"]
    X = D["x"]; Y = D["y"]
    pass_by_frame = {}
    for i in range(uf.size):
        s = int(starts[i]); e = s + int(counts[i])
        m = stage[s:e] == EX.PASS
        if m.any():
            pass_by_frame[int(uf[i])] = (X[s:e][m].astype(np.int64),
                                         Y[s:e][m].astype(np.int64))
    return dets, pass_by_frame, meta


def _wp_pass_mask(pass_xy, height, width, block):
    """PASS candidates -> the detector's full-res green mask. In pixelize mode a
    candidate stands for its whole BxB block (utils_pooled._scatter upscales the
    block result the same way), so each one paints its block's footprint."""
    mask = np.zeros((height, width), dtype=bool)
    if pass_xy is None:
        return mask
    xs, ys = pass_xy
    B = max(1, int(block))
    if B == 1:
        ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[ok], xs[ok]] = True
        return mask
    for bx, by in set(zip((xs // B).tolist(), (ys // B).tolist())):
        mask[by * B:(by + 1) * B, bx * B:(bx + 1) * B] = True
    return mask


def _sidecar_rgb_offset_us(seq_dir):
    """Always 0: the RGB and event streams of a FRED case share a time origin.
    (The old per-case rgb_offsets.json picked the wrong RGB frame and was
    removed; --rgb-offset-ms is still honoured as a manual override.)"""
    return 0.0


class StreamState:
    def __init__(self, raw_path, rgb_dir, rgb_offset_us=0.0):
        self.raw_path = raw_path
        self.rgb_offset_us = float(rgb_offset_us)    # events start earlier by this
        rgb_files = sorted(
            (f for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg")),
            key=rgb_frame_seconds,
        )
        if not rgb_files:
            raise FileNotFoundError(f"No .jpg RGB frames found in {rgb_dir}")
        self.rgb_paths = [os.path.join(rgb_dir, f) for f in rgb_files]
        pts_s          = np.array([rgb_frame_seconds(f) for f in rgb_files], dtype=np.float64)
        self.pts_us    = (pts_s - pts_s[0]) * 1e6
        self.t_end     = int(self.pts_us[-1])
        self.max_events = 20_000_000
        self.mv_reader  = RawReader(raw_path, max_events=self.max_events)
        self.ev_height, self.ev_width = self.mv_reader.get_size()
        self.mv_cursor_us = 0
        self.video_idx    = -1
        self.last_frame   = None
        frames_dir = os.path.join(os.path.dirname(raw_path), "Frames")
        pairs = []
        if os.path.isdir(frames_dir):
            for f in os.listdir(frames_dir):
                m = re.search(r"frame_(\d+)\.png$", f, re.IGNORECASE)
                if m:
                    pairs.append((int(m.group(1)), os.path.join(frames_dir, f)))
            pairs.sort()
        self.ev_frame_us    = np.array([p[0] for p in pairs], dtype=np.int64)
        self.ev_frame_paths = [p[1] for p in pairs]
        self.ev_frame_k     = -1
        self.ev_frame_img   = None

    def video_frame_at(self, target_idx):
        target_idx = max(0, min(target_idx, len(self.rgb_paths) - 1))
        if target_idx != self.video_idx:
            self.last_frame = np.asarray(
                Image.open(self.rgb_paths[target_idx]).convert('RGB'))
            self.video_idx = target_idx
        return self.last_frame

    def event_frame_at(self, t_us):
        if not self.ev_frame_paths:
            return None, None
        i    = int(np.searchsorted(self.ev_frame_us, t_us))
        cand = [k for k in (i-1, i) if 0 <= k < len(self.ev_frame_us)]
        k    = min(cand, key=lambda j: abs(int(self.ev_frame_us[j]) - t_us))
        if k != self.ev_frame_k:
            self.ev_frame_img = np.asarray(
                Image.open(self.ev_frame_paths[k]).convert('RGB'))
            self.ev_frame_k = k
        return self.ev_frame_img, int(self.ev_frame_us[k])

    def _seek_event_stream(self, t_us):
        if t_us < self.mv_cursor_us:
            try:
                del self.mv_reader
            except Exception:
                pass
            self.mv_reader = RawReader(self.raw_path, max_events=self.max_events)
        self.mv_reader.seek_time(int(t_us))
        self.mv_cursor_us = int(t_us)

    def load_window(self, frame_idx, window_us, verbose=True):
        t_start = frame_idx * window_us
        if t_start + window_us - self.rgb_offset_us > self.t_end:
            raise IndexError(
                f"frame_idx {frame_idx} (t={t_start/1e6:.3f}s) past stream end "
                f"({(self.t_end + self.rgb_offset_us)/1e6:.3f}s)")
        seek_us    = max(0, t_start - window_us)
        self._seek_event_stream(seek_us)
        evs        = self.mv_reader.load_delta_t(int(window_us))
        # load_delta_t has advanced the reader PAST this window; record its true
        # position so re-loading this frame (or an earlier one) resets the reader
        # instead of asking RawReader for an illegal backward seek.
        self.mv_cursor_us = seek_us + window_us
        # rgb_time = event_time - offset (the event recorder starts first)
        center_us  = t_start + window_us / 2.0 - self.rgb_offset_us
        vid_idx    = int(np.searchsorted(self.pts_us, center_us, side='right') - 1)
        vid_idx    = max(0, vid_idx)
        base_frame = self.video_frame_at(vid_idx)
        if base_frame is None:
            raise IndexError(f"video has < {vid_idx+1} frames")
        if verbose:
            print(f"Window {frame_idx}: t=[{seek_us/1e6:.3f}s, "
                  f"{(seek_us+window_us)/1e6:.3f}s], vid={vid_idx}, evs={evs.size}")
        return base_frame, evs, vid_idx

    def max_frame_idx(self, window_us):
        return max(0, int(self.t_end + self.rgb_offset_us) // window_us - 1)


def main():
    # --wp is read before the real parse so a working point's gates become the
    # defaults for the flags below (explicitly passed flags still win).
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--wp")
    _wp_path = _pre.parse_known_args()[0].wp

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",        type=str,   default="28")
    parser.add_argument("--frame",      type=int,   default=0)
    parser.add_argument("--window-ms",  type=float, default=33.333)
    parser.add_argument("--wp",         type=str,   default=None,
                        help="working_point_*.json: show the detections that "
                             "working point actually scored, replayed through "
                             "frame_explorer (min_snr_sum/min_snr_avg/persistence/"
                             "coast-confirm-acquire-predict included). The boxes "
                             "and the green mask come from the replay; every view, "
                             "the heatmap and the click-a-pixel spectrum stay this "
                             "viewer's own. Needs the case's frame_data cache.")
    # Detector hyperparameters: --tau-sf/--tau-comb/--min-events/--min-box-area/
    # --alpha/--min-rotor-pixels/--num-harmonics/--min-harmonics/
    # --harmonic-half-width/--max-freq-hz/--pixel-block — identical NAMES and
    # DEFAULTS to batch_ddhf_eval_pooled (shared ddhf_config preset).
    add_cfg_cli_args(parser)
    # Detector mode (single flag, mirroring the batch aggregations):
    #   pixelize   - BxB blocks, one NDFT per block (--pixel-block)        [default]
    #   pool       - 3x3 spatial pooling
    #   baseline   - per-pixel, no spatial aggregation
    #   components - 8-connected components, one NDFT per component
    parser.add_argument("--mode", choices=MODES, default="pixelize",
                        help="Spatial aggregation: 'pixelize' (BxB blocks, default), "
                             "'pool' (3x3 pooling), 'baseline' (per-pixel, no aggregation), "
                             "or 'components' (8-connected components, one spectral test each; "
                             "uses --min-events / --min-rotor-pixels as the per-component gates).")
    # Viewer-only options.
    parser.add_argument("--snr-sig",    type=float, default=None,
                        help="Spectrum-panel comb significance threshold (display only; default 8).")
    parser.add_argument("--drone",      choices=sorted(DRONE_BANDS), default="all",
                        help="Restrict the comb search to one drone's bands (default 'all' = preset).")
    parser.add_argument("--align-dx",   type=float, default=0.0)
    parser.add_argument("--align-dy",   type=float, default=0.0)
    parser.add_argument("--align-scale",type=float, default=1.0)
    # Box / IoU-graph match colouring (mirrors ddhf_eval_pooled --match center):
    # a detection is "on a drone" if its centre is within scale × GT-half-diagonal
    # of a GT centre. Drives green/red/orange/blue box colours and the IoU-graph tint.
    parser.add_argument("--center-dist-scale", type=float, default=1.0,
                        help="Match tolerance = scale × GT box half-diagonal (default 1.0).")
    parser.add_argument("--center-dist-px", type=float, default=0.0,
                        help="Absolute floor (px) on the match tolerance (default 0).")
    parser.add_argument("--gpu", action="store_true",
                        help="Run the spectral stage on the GPU (CuPy), matching "
                             "ddhf_eval_pooled's default fp32 path. Set DDHF_GPU_FP32=0 "
                             "for fp64. Falls back to CPU if no usable GPU.")
    # Closed-loop single-drone tracking (pool gate). When --track, the displayed
    # box is the tracker's emitted box (replayed from frame 0); the strict gate is
    # the --tau-sf/--tau-comb/--min-events/... above, relaxed toward the floor near
    # the previous detection by a Gaussian of width --sigma. See ddhf_pool_tracker.
    parser.add_argument("--rgb-offset-ms", type=float, default=None,
                        help="RGB starts this many ms AFTER the event stream "
                             "(rgb_time = event_time - offset). Manual override; "
                             "default 0 (the streams share a time origin).")
    parser.add_argument("--flag-neighborhood", type=int, default=1,
                        help="When a pixel is detected, flag its NxN neighbourhood (mask "
                             "dilation before box extraction). 1 = off (just the pixel).")
    parser.add_argument("--track", action="store_true",
                        help="Single-drone closed-loop tracking on the pool gate.")
    parser.add_argument("--sigma", type=float, default=120.0)
    parser.add_argument("--lose-after", type=int, default=24)
    parser.add_argument("--floor-min-events", type=int, default=1)
    parser.add_argument("--floor-tau-sf", type=float, default=1.0)
    parser.add_argument("--floor-tau-comb", type=float, default=1.0)
    if _wp_path:
        parser.set_defaults(**_wp_arg_defaults(_wp_path))
    args = parser.parse_args()
    if args.wp and args.track:
        parser.error("--wp replays the working point's own tracker; "
                     "--track (this viewer's single-drone tracker) would "
                     "override it. Pass only one.")

    cds = args.center_dist_scale     # match tolerance scale (× GT half-diagonal)
    cdp = args.center_dist_px        # absolute floor on the tolerance

    global _USE_GPU, _PIXELIZE, _POOLING, _COMPONENTS
    _USE_GPU    = args.gpu
    # --mode selects one aggregation; the (_COMPONENTS, _PIXELIZE, _POOLING)
    # triple it expands to reproduces the old flag combinations exactly.
    _COMPONENTS, _PIXELIZE, _POOLING = MODE_FLAGS[args.mode]
    if _USE_GPU:
        ok = _gpu_available()
        prec = "fp32" if os.environ.get("DDHF_GPU_FP32", "1") != "0" else "fp64"
        print(f"[GPU] requested: {'using CuPy GPU (' + prec + ')' if ok else 'NO usable GPU — falling back to CPU'}")

    DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.seq)
    RAW_FILE = _hf.event_file(DIR)          # events.raw or HF events.hdf5
    RGB_DIR  = _hf.rgb_dir(DIR)             # PADDED_RGB/ or HF RGB/
    GT_FILE  = os.path.join(DIR, "interpolated_coordinates.txt")
    if not os.path.exists(GT_FILE):                  # corrected GT preferred —
        GT_FILE = os.path.join(DIR, "coordinates.txt")   # same file format

    if args.snr_sig is None:
        args.snr_sig = 8.0

    # Build the config from the shared preset + CLI overrides (identical to batch);
    # --window-ms drives windowing and the spectrum; --drone overrides the comb
    # bands (default 'all' == the preset's bands).
    overrides = cfg_overrides_from_args(args)
    overrides["window_ms"] = args.window_ms
    overrides["step_ms"]   = args.window_ms
    cfg = build_cfg(**overrides)
    cfg.rotor_bands_hz = DRONE_BANDS[args.drone]

    mode_label = ("connected-components" if _COMPONENTS else
                  (f"pixelize {cfg.pixel_block}x{cfg.pixel_block}" if _PIXELIZE
                   else ("3x3 pooled" if _POOLING else "no pooling (per-pixel)")))
    window_us = int(round(args.window_ms * 1000.0))
    bin_hz    = 1000.0 / args.window_ms
    k_bins    = cfg.frequency_bins
    print(f"[{mode_label}]  window={args.window_ms}ms  bin={bin_hz:.1f}Hz  "
          f"frequency_bins={k_bins}  max_freq={(k_bins-1)*bin_hz:.0f}Hz")

    M     = np.array([[args.align_scale, 0.0, args.align_dx],
                      [0.0, args.align_scale, args.align_dy]], dtype=np.float32)
    M_inv = invert_affine_2x3(M)

    rgb_off_us = (args.rgb_offset_ms * 1000.0 if args.rgb_offset_ms is not None
                  else _sidecar_rgb_offset_us(DIR))
    if rgb_off_us:
        print(f"[rgb-sync] rgb_time = event_time - {rgb_off_us / 1e6:+.3f}s "
              f"({'--rgb-offset-ms' if args.rgb_offset_ms is not None else 'default'})")
    stream    = StreamState(RAW_FILE, RGB_DIR, rgb_offset_us=rgb_off_us)
    ev_height = stream.ev_height
    ev_width  = stream.ev_width
    max_idx   = stream.max_frame_idx(window_us)

    # ── working-point replay (--wp) ──────────────────────────────────────────────
    # Run frame_explorer's pipeline over the whole case ONCE; compute_frame then
    # shows ITS detections and pass mask instead of the viewer's own, so what is
    # drawn is what the working point scored.
    wp_dets = wp_pass = None
    if args.wp:
        print(f"[wp] replaying {os.path.basename(args.wp)} through frame_explorer ...")
        wp_dets, wp_pass, wp_meta = _wp_replay(args.wp, args.seq)
        n_box = sum(len(v) for v in wp_dets.values())
        print(f"[wp] {n_box} detections over {len(wp_dets)} frames "
              f"[{wp_meta.get('mode')}"
              f"{', tracked sigma=%g' % wp_meta.get('track_sigma', 20.0) if wp_meta.get('track') else ''}"
              f"{', skip_start' if wp_meta.get('skip_start') else ''}]")
        print("[wp] boxes + green mask = the working point's; the heatmap, the "
              "polarity view and the click-a-pixel spectrum remain this viewer's "
              "own per-pixel computation.")

    # ── closed-loop tracking (pool gate) ─────────────────────────────────────────
    # When --track, replay the single-drone PoolTracker over every window ONCE and
    # store the emitted box per frame (disk-cached); compute_frame then shows that
    # box. Tracking uses the pool gate, so force pool aggregation.
    tracked = {}
    if args.track:
        from ddhf_pool_tracker import PoolTracker
        strict = dict(min_events=cfg.min_events, tau_sf=cfg.tau_sf, tau_comb=cfg.tau_comb)
        floor  = dict(min_events=args.floor_min_events, tau_sf=args.floor_tau_sf,
                      tau_comb=args.floor_tau_comb)
        # Track on the SELECTED --mode's cached candidates (stride 1), so the box
        # shown belongs to the same detector variant as the rest of the display.
        F = _track_features(args.seq, args.mode)
        if F is not None:
            src = f"frame_data cache, {args.mode}, stride 1"
        else:                                            # not cached: own precompute
            import ddhf_stats_explorer as _ex
            D = _ex.build_dataset(args.seq, args.mode, args.drone, 4, {}, None,
                                  None, False)
            F = dict(x=D["x"].astype(np.int64), y=D["y"].astype(np.int64),
                     gc=D["gc"], sf=D["sf"], snr=D["snr"], uniq=D["uniq_frames"],
                     starts=D["frame_starts"], counts=D["frame_counts"],
                     stride=int(D["meta_stride"]) if "meta_stride" in D else 4)
            src = (f"explorer precompute, {args.mode}, stride {F['stride']} "
                   f"(boxes carried across the gaps)")
        stride_t = F["stride"]
        tr = PoolTracker(strict, floor, args.sigma, args.lose_after,
                         cfg.min_rotor_pixels, cfg.min_box_area, cfg.alpha, args.flag_neighborhood)
        n_box = 0
        for i in range(F["uniq"].size):
            f = int(F["uniq"][i]); s = int(F["starts"][i]); e = s + int(F["counts"][i])
            box, _ = tr.step(F["x"][s:e], F["y"][s:e], F["gc"][s:e],
                             F["sf"][s:e], F["snr"][s:e])
            if box is not None:
                n_box += 1
                for k in range(f, min(max_idx, f + stride_t - 1) + 1):   # 1 window when stride 1
                    tracked[k] = box
        print(f"[track] {src}; sigma={args.sigma} lose_after={args.lose_after} "
              f"floor=(me{args.floor_min_events},sf{args.floor_tau_sf},comb{args.floor_tau_comb}); "
              f"emitted a box in {n_box}/{F['uniq'].size} windows")
        print("[track] NOTE: this is ddhf_pool_tracker (single-drone, last-position "
              "prior) — NOT frame_explorer's _track_pass (multi-track, "
              "coast/confirm/acquire/predict) that the working points were tuned "
              "with; use frame_explorer for an exact replay.")

    gt_times, gt_boxes = _load_gt_boxes(GT_FILE)
    if gt_times is not None:
        print(f"Loaded {len(gt_times)} GT boxes from {GT_FILE}")
    else:
        print(f"No GT boxes (looked for {GT_FILE})")
    gt_tol_s = cfg.window_ms / 1000.0

    def gt_rects_at(frame_idx):
        if gt_times is None:
            return []
        t_s  = frame_idx * window_us / 1e6
        j    = int(np.searchsorted(gt_times, t_s))
        cand = [k for k in (j-1, j) if 0 <= k < len(gt_times)]
        if not cand:
            return []
        k = min(cand, key=lambda i: abs(gt_times[i] - t_s))
        if abs(gt_times[k] - t_s) > gt_tol_s:
            return []
        sel = np.flatnonzero(np.isclose(gt_times, gt_times[k]))
        return [tuple(int(round(v)) for v in gt_boxes[i]) for i in sel]

    ev_corners = np.array([
        [0, 0, 1], [ev_width-1, 0, 1],
        [ev_width-1, ev_height-1, 1], [0, ev_height-1, 1],
    ], dtype=np.float32).T
    fov_corners = (M @ ev_corners).T
    fov_pts     = [(float(x), float(y)) for x, y in fov_corners]
    fov_pts.append(fov_pts[0])

    state = {
        "frame_idx": max(0, min(args.frame, max_idx)),
        "view": VIEW_EVENTS,
        "base_frame": None, "boxes": None, "gt_rects": [],
        "count_map": None,  "sf_map": None, "snr_map": None,
        "rotor_mask": None, "evs": None,
    }

    frame_cache: "OrderedDict[int, dict]" = OrderedDict()
    FRAME_CACHE_MAX = 24

    def compute_frame(idx):
        cached = frame_cache.get(idx)
        if cached is not None:
            frame_cache.move_to_end(idx)
            return cached
        base_frame, evs, _ = stream.load_window(idx, window_us)
        # Per-pixel maps kept for the click-a-pixel spectrum panel regardless of mode.
        cmap, sfm, snrm, f0m = compute_pixel_stats(evs, ev_height, ev_width, cfg)
        cc_labels = None
        if _COMPONENTS:
            # Connected-component aggregation drives the mask + boxes instead of the
            # per-pixel/block gate (which still feeds the inspection maps above).
            rmask, boxes, cc_labels = detect_components_mask(evs, ev_height, ev_width, cfg)
        else:
            # Gate: pixel's 3x3 pool had >= min_events iff its SF was computed (not NaN).
            rmask = (
                ~np.isnan(sfm)  & (sfm  <= cfg.tau_sf)
                & ~np.isnan(snrm) & (snrm >= cfg.tau_comb)
            )
            if args.flag_neighborhood > 1:               # flag each detection's NxN neighbourhood
                from scipy.ndimage import binary_dilation as _bdil
                rmask = _bdil(rmask, structure=np.ones((args.flag_neighborhood,
                                                        args.flag_neighborhood), bool))
            boxes = _extract_boxes(rmask, np.nan_to_num(snrm), cfg)
        if args.track:                                   # show the tracker's box, not the per-frame boxes
            tb = tracked.get(idx)
            boxes = [tb] if tb is not None else []
        if wp_dets is not None:                          # --wp: the scored pipeline's
            boxes = [BoundingBox(int(x1), int(y1), int(x2), int(y2))
                     for (x1, y1, x2, y2) in wp_dets.get(idx, [])]
            rmask = _wp_pass_mask(wp_pass.get(idx), ev_height, ev_width,
                                  cfg.pixel_block if _PIXELIZE else 1)
        print_f0_consensus(f0m, rmask)
        active = cmap[cmap > 0]
        if active.size:
            print(f"  total_events={int(evs.size):>7d}  active_px={active.size:>6d}  "
                  f"pooled_eligible={(~np.isnan(sfm)).sum()}  "
                  f"green={int(rmask.sum())}  boxes={len(boxes)}")
        fr = dict(base_frame=base_frame, evs=evs,
                  count_map=cmap, sf_map=sfm.astype(np.float32),
                  snr_map=snrm.astype(np.float32), rotor_mask=rmask,
                  boxes=boxes, cc_labels=cc_labels,
                  gt_rects=gt_rects_at(idx))
        frame_cache[idx] = fr
        if len(frame_cache) > FRAME_CACHE_MAX:
            frame_cache.popitem(last=False)
        return fr

    fr         = compute_frame(state["frame_idx"])
    state.update(frame_idx=state["frame_idx"], **fr)
    base_frame = fr["base_frame"]
    H, W       = base_frame.shape[:2]
    print(f"RGB={W}x{H}, event sensor={ev_width}x{ev_height}, max frame idx={max_idx}")

    def render_state():
        v = state["view"]
        if v == VIEW_PLAIN:
            img = render_plain(state["base_frame"])
        elif v == VIEW_HEATMAP:
            img = render_heatmap(state["base_frame"], state["count_map"], M, fov_pts)
        elif v == VIEW_POLARITY:
            img = render_polarity(state["base_frame"], state["evs"], M, fov_pts)
        elif v == VIEW_EVENTFRAME:
            ev_img, _ = stream.event_frame_at(state["frame_idx"] * window_us)
            img = render_event_frame(ev_img, state["base_frame"], M, state["boxes"],
                                     state["gt_rects"], cds, cdp)
        elif v == VIEW_GTBOX:
            return render_gtbox(state["base_frame"], state["evs"], M, state["gt_rects"])
        else:
            img = render_overlay(state["base_frame"], state["evs"], M, fov_pts,
                                 state["rotor_mask"], state["boxes"],
                                 state["gt_rects"], cds, cdp)
        if state["gt_rects"]:
            H_, W_ = img.shape[:2]
            det_rects = [_project_box_to_rgb(b, M, W_, H_)
                         for b in (state["boxes"] or [])]
            pim = Image.fromarray(img)
            _draw_gt(ImageDraw.Draw(pim), state["gt_rects"], det_rects,
                     cds, cdp, match_color=True, evs=state["evs"])
            img = np.array(pim)
        return img

    def view_title(idx):
        v = state["view"]
        return (f"[{mode_label}] View {v}/{N_VIEWS} [{VIEW_NAMES[v]}] — "
                f"frame {idx}  (click pixel for stats)")

    overlay = render_state()

    plt.rcParams.update({
        "axes.titlesize": 10, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })

    fig     = plt.figure(figsize=(16, 9))
    ax_over = fig.add_axes([0.02, 0.275, 0.96, 0.665])
    ax_spec = fig.add_axes([0.08, 0.10,  0.88, 0.12])
    ax_graph = fig.add_axes([0.08, 0.10, 0.88, 0.12]); ax_graph.set_visible(False)
    ax_iou   = fig.add_axes([0.08, 0.10, 0.88, 0.12]); ax_iou.set_visible(False)
    ax_iou.set_navigate(False)   # 'i' view: disable toolbar pan/zoom (click-to-jump still works)

    im_over    = ax_over.imshow(overlay)
    title_over = ax_over.set_title(view_title(state["frame_idx"]))
    ax_over.axis("off")

    param_text = fig.text(0.5, 0.998, "", ha="center", va="top",
                          family="monospace", fontsize=8)
    ax_spec.set_title("Click a pixel — shows POOLED 3x3 spectrum (frequency in Hz)")
    ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")

    layout  = {"split_y": 0.24}
    guide_h = Line2D([0,0],[0,0], transform=fig.transFigure,
                     color="0.5", lw=4, alpha=0.35, zorder=10)
    fig.add_artist(guide_h)

    def apply_layout():
        sy = layout["split_y"]
        ax_over.set_position([0.02, sy+0.03, 0.96, max(0.06, 0.94-(sy+0.03))])
        spec_h = max(0.05, (sy-0.02) - 0.10)
        ax_spec.set_position( [0.08, 0.10, 0.88, spec_h])
        ax_graph.set_position([0.08, 0.10, 0.88, spec_h])
        ax_iou.set_position(  [0.08, 0.10, 0.88, spec_h])
        guide_h.set_data([0.02, 0.98], [sy, sy])
        fig.canvas.draw_idle()

    info_text = fig.text(
        0.5, 0.005,
        "Click pixel for POOLED stats. 1-5/'v'=views, 'c'/9=GT-box DDHF (yellow). "
        "'b'=best-SNR, 'd'=densest, 'g'=usable-px graph, 'i'=IoU graph. Drag grey line to resize. ←/→ ±1, Shift±10.",
        ha="center", va="bottom", family="monospace", fontsize=8,
    )

    tb_ax     = fig.add_axes([0.88, 0.040, 0.08, 0.026])
    frame_box = TextBox(tb_ax, f"Go to frame (0..{max_idx}): ",
                        initial=str(state["frame_idx"]))
    apply_layout()

    def update_overlay_title():
        bands_s = ",".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in cfg.rotor_bands_hz)
        wp_s = ""
        if args.wp:                                      # keep it short — the full
            wp_s = f"  [wp {os.path.basename(args.wp)}]"  # note is printed at startup
            f0 = wp_meta.get("_first_gt_frame")
            if f0 is not None and state["frame_idx"] < f0:
                wp_s += f"  PRE-GT <{f0}: unscored"
        param_text.set_text(
            f"[{mode_label}]  frame {state['frame_idx']}/{max_idx}  "
            f"window={cfg.window_ms}ms  min_events>={cfg.min_events}  "
            f"SF<={cfg.tau_sf}  SNR>={cfg.tau_comb}  "
            f"f0 bands={bands_s} Hz  GREEN px={int(state['rotor_mask'].sum())}"
            + wp_s
        )

    def reload(new_idx):
        new_idx = max(0, min(new_idx, max_idx))
        if new_idx == state["frame_idx"] and state["count_map"] is not None:
            return
        try:
            fr = compute_frame(new_idx)
        except IndexError as e:
            print(f"reload({new_idx}) failed: {e}"); return
        state.update(frame_idx=new_idx, **fr)
        im_over.set_data(render_state())
        ax_spec.clear()
        ax_spec.set_title("Click a pixel — shows POOLED 3x3 spectrum (frequency in Hz)")
        ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")
        title_over.set_text(view_title(new_idx))
        update_overlay_title()
        info_text.set_text(f"Loaded frame {new_idx} — GREEN px={int(fr['rotor_mask'].sum())}.")
        if graph["marker"]   is not None: graph["marker"].set_xdata([new_idx, new_idx])
        if iougraph["marker"] is not None: iougraph["marker"].set_xdata([new_idx, new_idx])
        if state["view"] == VIEW_GTBOX:
            show_gtbox_spectrum()                # keep the GT-box transform in sync
        frame_box.eventson = False
        frame_box.set_val(str(new_idx))
        frame_box.eventson = True
        fig.canvas.draw_idle()

    update_overlay_title()

    def show_pixel(xe, ye):
        if graph["on"]:    toggle_graph()
        if iougraph["on"]: toggle_iou()
        rgbv   = M @ np.array([xe, ye, 1.0], dtype=np.float32)
        x_rgb  = int(round(float(rgbv[0]))); y_rgb = int(round(float(rgbv[1])))
        cmap   = state["count_map"]; sfm = state["sf_map"]
        snrm   = state["snr_map"];   rmask = state["rotor_mask"]
        n_own  = int(cmap[ye, xe])          # pixel's OWN event count (raw)
        rotor  = bool(rmask[ye, xe])
        # Spectrum source depends on aggregation: in --components mode the WHOLE
        # connected component is pooled (so every pixel of a component yields the
        # SAME spectrum/SF/SNR); else the 3x3 pool / BxB block around the pixel.
        if _COMPONENTS:
            power, freqs, n_pool = component_power_spectrum(
                state["evs"], state.get("cc_labels"), xe, ye, cfg)
            pool_word, spec_word = "component", "COMPONENT"
            if power is not None:
                _pnd = power[1:] + 1e-12
                sf  = float(np.exp(np.mean(np.log(_pnd))) / np.mean(_pnd))
                _o, snr = best_comb(power, cfg)
            else:
                sf = float("nan"); snr = float("nan")
        else:
            sf = sfm[ye, xe]; snr = snrm[ye, xe]
            power, freqs = pixel_power_spectrum(state["evs"], xe, ye, cfg)
            _blocks = _PIXELIZE or not _POOLING
            _B = (max(1, int(cfg.pixel_block)) if _PIXELIZE else 1) if _blocks else None
            pool_word, spec_word = ((f"{_B}x{_B} block", "BLOCK") if _PIXELIZE
                                    else ("3x3 pool", "POOLED") if _POOLING
                                    else ("pixel", "PIXEL"))
            n_pool = 0
            if power is not None:
                evx = state["evs"]["x"].astype(np.int64)
                evy = state["evs"]["y"].astype(np.int64)
                if _blocks:
                    n_pool = int(((evx // _B == xe // _B) & (evy // _B == ye // _B)).sum())
                else:
                    n_pool = int(((np.abs(evx - xe) <= 1) & (np.abs(evy - ye) <= 1)).sum())
        ok_n   = not np.isnan(sf)
        ok_sf  = (not np.isnan(sf))  and sf  <= cfg.tau_sf
        ok_snr = (not np.isnan(snr)) and snr >= cfg.tau_comb
        sf_s   = f"{sf:.4f}"  if not np.isnan(sf)  else "n/a"
        snr_s  = f"{snr:.4f}" if not np.isnan(snr) else "n/a"
        info_text.set_text(
            f"RGB ({x_rgb},{y_rgb}) -> event ({xe},{ye})   "
            f"own_events={n_own}  {pool_word}_eligible=[{'PASS' if ok_n else 'FAIL'}]   "
            f"SF={sf_s} [{'PASS' if ok_sf else 'FAIL'} <={cfg.tau_sf}]   "
            f"SNR={snr_s} [{'PASS' if ok_snr else 'FAIL'} >={cfg.tau_comb}]   "
            f"=> {'GREEN' if rotor else 'not flagged'}"
        )
        print(info_text.get_text())

        ax_spec.clear()
        ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")
        if power is None:
            ax_spec.set_title(f"event px ({xe},{ye}) -- no events in {pool_word}")
        else:
            ax_spec.plot(freqs[1:], power[1:], color="steelblue", lw=1.2)
            omega, snr_best = best_comb(power, cfg)
            if omega > 0:
                significant    = snr_best >= args.snr_sig
                c_fund, c_harm = ("red","orange") if significant else ("0.45","0.65")
                bin_hz_        = 1000.0 / cfg.window_ms
                hw_hz          = cfg.harmonic_half_width * bin_hz_
                for m in range(1, cfg.num_harmonics + 1):
                    fh = m * omega * bin_hz_
                    if fh <= freqs[-1]:
                        ax_spec.axvspan(fh-hw_hz, fh+hw_hz, color=c_harm, alpha=0.12,
                                        label=(f"±Δk ({cfg.harmonic_half_width} bins = ±{hw_hz:.0f}Hz)")
                                              if m == 1 else None)
                        if m == 1:
                            ax_spec.axvline(fh, color=c_fund, ls="-", lw=1.8, alpha=0.9,
                                            label=f"fundamental {fh:.0f} Hz")
                        else:
                            ax_spec.axvline(fh, color=c_harm, ls="--", lw=1.0, alpha=0.6,
                                            label="harmonics" if m == 2 else None)
                ax_spec.legend(loc="upper right", fontsize=7)
                fund = omega * 1000.0 / cfg.window_ms
                ax_spec.set_title(
                    f"{spec_word} spectrum — event px ({xe},{ye}), "
                    f"own={n_own} ev / pool={n_pool} ev")
                verdict = (f"COMB: f0~{fund:.0f}Hz (bin {omega})" if significant else
                           f"NO significant comb — best ~{fund:.0f}Hz "
                           f"(SNR {snr_best:.1f} < {args.snr_sig:g})")
                ax_spec.text(
                    0.01, 0.97,
                    f"{verdict}\ncomb SNR = {snr_best:.2f}\nSF = {sf_s}",
                    transform=ax_spec.transAxes, ha="left", va="top",
                    family="monospace", fontsize=7,
                    bbox=dict(boxstyle="round",
                              fc="white" if significant else "0.92",
                              ec="0.7", alpha=0.85),
                )
            else:
                ax_spec.set_title(
                    f"{spec_word} spectrum — event px ({xe},{ye}), "
                    f"own={n_own} ev / pool={n_pool} ev  (no comb, SF={sf_s})")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes in (ax_graph, ax_iou):
            if event.xdata is not None: reload(int(round(event.xdata)))
            return
        if event.inaxes is not ax_over: return
        if event.xdata is None or event.ydata is None: return
        x_rgb = int(round(event.xdata)); y_rgb = int(round(event.ydata))
        ve    = M_inv @ np.array([x_rgb, y_rgb, 1.0], dtype=np.float32)
        xe    = int(round(float(ve[0]))); ye = int(round(float(ve[1])))
        if not (0 <= xe < ev_width and 0 <= ye < ev_height):
            info_text.set_text(f"RGB ({x_rgb},{y_rgb}) -> event ({xe},{ye}) -- outside FOV")
            fig.canvas.draw_idle(); return
        show_pixel(xe, ye)

    def jump_to_densest():
        cmap = state["count_map"]
        if cmap is None or cmap.max() <= 0: return
        ye, xe = np.unravel_index(int(np.argmax(cmap)), cmap.shape)
        print(f"densest pixel: event ({int(xe)},{int(ye)}) with {int(cmap[ye,xe])} own events")
        show_pixel(int(xe), int(ye))

    def jump_to_best_snr():
        snrm = state["snr_map"]
        if snrm is None or np.all(np.isnan(snrm)): return
        ye, xe = np.unravel_index(int(np.nanargmax(snrm)), snrm.shape)
        print(f"best-SNR pixel: event ({int(xe)},{int(ye)}) SNR={snrm[ye,xe]:.1f}")
        show_pixel(int(xe), int(ye))

    GRAPH_THRS = tuple(sorted({cfg.min_events, 6, 15, 30}))
    graph      = {"on": False, "counts": None, "marker": None}
    cache_file = os.path.join(DIR, f".usable_px_w{cfg.window_ms:g}_me{cfg.min_events}.npz")
    iougraph   = {"on": False, "iou": None, "cls": None, "marker": None}
    _mode_key = ("cc" if _COMPONENTS else
                 (f"pix{cfg.pixel_block}" if _PIXELIZE
                  else ("pool" if _POOLING else "plain")))
    iou_cache_file = os.path.join(
        DIR, f".iou_{_mode_key}_w{cfg.window_ms:g}_sf{cfg.tau_sf:g}_cb{cfg.tau_comb:g}"
             f"_me{cfg.min_events}_mba{cfg.min_box_area}_a{cfg.alpha:g}"
             f"_cs{cds:g}_cp{cdp:g}.npz")

    def compute_usable_counts():
        if os.path.exists(cache_file):
            z = np.load(cache_file)
            if tuple(z["thrs"]) == GRAPH_THRS and z["counts"].shape[1] >= max_idx+1:
                return z["counts"][:, :max_idx+1]
        print(f"Scanning {max_idx+1} windows for usable-pixel counts ...")
        rd     = RawReader(RAW_FILE, max_events=20_000_000)
        counts = np.zeros((len(GRAPH_THRS), max_idx+1), dtype=np.int32)
        for i in range(max_idx+1):
            evs = rd.load_delta_t(window_us)
            if evs.size:
                pid = evs["y"].astype(np.int64)*ev_width + evs["x"].astype(np.int64)
                _, c = np.unique(pid, return_counts=True)
                for j, thr in enumerate(GRAPH_THRS):
                    counts[j, i] = int((c >= thr).sum())
            if i % 500 == 0: print(f"  window {i}/{max_idx}")
        np.savez(cache_file, thrs=np.array(GRAPH_THRS), counts=counts)
        return counts

    def toggle_graph():
        if not graph["on"] and graph["counts"] is None:
            graph["counts"] = compute_usable_counts()
            xs     = np.arange(graph["counts"].shape[1])
            tags   = {cfg.min_events:"configured gate", 6:"paper n_min",
                      15:"comb-grade (1 win)", 30:"strong"}
            colors = ["0.7","steelblue","darkorange","crimson"]
            for j, thr in enumerate(GRAPH_THRS):
                ax_graph.plot(xs, graph["counts"][j], lw=0.8, color=colors[j % len(colors)],
                              label=f">={thr} ev ({tags.get(thr,'')})")
            ax_graph.axhline(cfg.min_box_area, color="red", ls=":", lw=0.8,
                             label=f"min_box_area={cfg.min_box_area}")
            graph["marker"] = ax_graph.axvline(state["frame_idx"], color="0.3", lw=1.2)
            ax_graph.set_xlim(0, xs.size-1)
            ax_graph.set_yscale("symlog", linthresh=10)
            ax_graph.set_xlabel("frame"); ax_graph.set_ylabel("pixels")
            ax_graph.set_title("Pixels with enough events, per window — click to jump")
            ax_graph.legend(loc="upper right", fontsize=7, ncol=len(GRAPH_THRS)+1)
        graph["on"] = not graph["on"]
        if graph["on"] and iougraph["on"]:
            iougraph["on"] = False; ax_iou.set_visible(False)
        ax_graph.set_visible(graph["on"])
        ax_spec.set_visible(not (graph["on"] or iougraph["on"]))
        fig.canvas.draw_idle()

    # Per-frame class codes for the IoU-graph colouring (center-distance metric).
    CLS_TN, CLS_TP, CLS_OFF, CLS_FN, CLS_FP = 0, 1, 2, 3, 4

    def compute_iou_per_frame():
        # Cache must carry 'cls' (older caches without it are recomputed).
        if os.path.exists(iou_cache_file):
            z = np.load(iou_cache_file)
            if "cls" in z and z["iou"].shape[0] >= max_idx+1:
                return z["iou"][:max_idx+1], z["cls"][:max_idx+1]
        print(f"Scanning {max_idx+1} frames for per-frame IoU + match class (pooled) ...")
        rd  = RawReader(RAW_FILE, max_events=20_000_000)
        iou = np.full(max_idx+1, np.nan)
        cls = np.zeros(max_idx+1, dtype=np.int8)        # default TN/none
        for k in range(max_idx):
            evs = rd.load_delta_t(window_us)
            n   = k + 1
            gts = gt_rects_at(n)
            if _COMPONENTS:
                _rm, boxes, _lbl = detect_components_mask(evs, ev_height, ev_width, cfg)
            else:
                cmap, sfm, snrm, _ = compute_pixel_stats(evs, ev_height, ev_width, cfg)
                rmask = (~np.isnan(sfm) & (sfm <= cfg.tau_sf)
                         & ~np.isnan(snrm) & (snrm >= cfg.tau_comb))
                boxes = _extract_boxes(rmask, np.nan_to_num(snrm), cfg)
            det   = [_project_box_to_rgb(b, M, W, H) for b in boxes]
            if gts:
                iou[n] = float(np.mean([max((_iou(g, d) for d in det), default=0.0)
                                        for g in gts]))
                if not det:
                    cls[n] = CLS_FN                      # GT present, nothing detected
                elif all(_det_near_any_gt(d, gts, cds, cdp) for d in det):
                    cls[n] = CLS_TP                      # every detection on a drone
                else:
                    cls[n] = CLS_OFF                     # detection(s), but not all near a GT
            else:
                cls[n] = CLS_FP if det else CLS_TN       # spurious box vs clean empty frame
            if n % 500 == 0: print(f"  frame {n}/{max_idx}")
        np.savez(iou_cache_file, iou=iou, cls=cls)
        return iou, cls

    def toggle_iou():
        if not iougraph["on"] and iougraph["iou"] is None:
            iougraph["iou"], iougraph["cls"] = compute_iou_per_frame()
            iou = iougraph["iou"]; cls = iougraph["cls"]
            xs  = np.arange(iou.shape[0])
            # Per-frame class colouring (center-distance match): tint each frame's
            # column by TP/off-target/FN/FP; TN frames left blank.
            from matplotlib.lines import Line2D
            cls_spec = [
                (CLS_TP,  "#00b400", "TP (near GT)"),
                (CLS_OFF, "#e00000", "detection off-target"),
                (CLS_FN,  "#4a90ff", "FN (GT, no detection)"),
                (CLS_FP,  "#ff9600", "FP (no GT, detection)"),
            ]
            for code, col, _lbl in cls_spec:
                idx = np.flatnonzero(cls == code)
                if idx.size:
                    ax_iou.vlines(idx, 0, 1, color=col, lw=1.0, alpha=0.45)
            ax_iou.plot(xs, iou, lw=0.9, color="black", label="best IoU(det, GT)")
            ax_iou.axhline(0.5, color="0.4", ls="--", lw=0.8)
            iougraph["marker"] = ax_iou.axvline(state["frame_idx"], color="0.15", lw=1.4)
            valid  = iou[~np.isnan(iou)]
            # Confusion counts: FN = off-target + no-detection on GT frames (both miss
            # the center criterion); TP/FP/TN as marked.
            n_tp   = int((cls == CLS_TP).sum())
            n_off  = int((cls == CLS_OFF).sum())
            n_fnz  = int((cls == CLS_FN).sum())
            n_fp   = int((cls == CLS_FP).sum())
            n_tn   = int((cls == CLS_TN).sum())
            n_fn   = n_off + n_fnz
            n_gt   = n_tp + n_fn
            succ_s = f"succ={n_tp/n_gt*100:.1f}%" if n_gt else "no GT frames"
            mean_s = f"meanIoU={valid.mean():.3f}" if valid.size else ""
            ax_iou.set_xlim(0, xs.size-1); ax_iou.set_ylim(0, 1)
            ax_iou.set_xlabel("frame"); ax_iou.set_ylabel("IoU")
            ax_iou.set_title(
                f"Per-frame match [{_mode_key} scale={cds:g}]  {succ_s}  "
                f"TP={n_tp} FN={n_fn} (off={n_off},none={n_fnz}) FP={n_fp} TN={n_tn}  "
                f"{mean_s} — click to jump")
            handles = [Line2D([0],[0], color=c, lw=4, alpha=0.6, label=l) for _, c, l in cls_spec]
            handles.append(Line2D([0],[0], color="black", lw=1.2, label="best IoU"))
            ax_iou.legend(handles=handles, loc="upper right", fontsize=6, ncol=3)
        iougraph["on"] = not iougraph["on"]
        if iougraph["on"] and graph["on"]:
            graph["on"] = False; ax_graph.set_visible(False)
        ax_iou.set_visible(iougraph["on"])
        ax_spec.set_visible(not (graph["on"] or iougraph["on"]))
        fig.canvas.draw_idle()

    def show_gtbox_spectrum():
        """Plot the pooled-GT-box spectrum (the 'transform of these specific pixels')
        with its comb-SNR and spectral flatness — shown automatically in VIEW_GTBOX."""
        power, freqs, n = gtbox_pool_spectrum(state["evs"], state.get("gt_rects") or [], cfg)
        ax_spec.clear()
        ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")
        if power is None:
            ax_spec.set_title("GT-box pooled spectrum — no GT drone / no events in box this frame")
        else:
            pnd = power[1:] + 1e-12
            sf = float(np.exp(np.mean(np.log(pnd))) / np.mean(pnd))
            omega, snr = best_comb(power, cfg)
            ax_spec.plot(freqs[1:], power[1:], color="goldenrod", lw=1.2)
            if omega > 0:
                bin_hz = 1000.0 / cfg.window_ms
                for m in range(1, cfg.num_harmonics + 1):
                    fh = m * omega * bin_hz
                    if fh <= freqs[-1]:
                        ax_spec.axvline(fh, color=("red" if m == 1 else "orange"),
                                        ls=("-" if m == 1 else "--"),
                                        lw=(1.8 if m == 1 else 1.0), alpha=0.85,
                                        label=(f"f0={fh:.0f} Hz" if m == 1 else None))
                ax_spec.legend(loc="upper right", fontsize=7)
            verdict = "PASS" if (sf <= cfg.tau_sf and snr >= cfg.tau_comb) else "FAIL"
            ax_spec.set_title(f"GT-BOX pooled spectrum — {n} ev   "
                              f"comb SNR={snr:.2f} (>= {cfg.tau_comb})   "
                              f"SF={sf:.3f} (<= {cfg.tau_sf})   -> {verdict}")
        ax_spec.set_visible(True)
        fig.canvas.draw_idle()

    def set_view(v):
        if v == state["view"]: return
        state["view"] = v
        im_over.set_data(render_state())
        title_over.set_text(view_title(state["frame_idx"]))
        info_text.set_text(f"View {v}: {VIEW_NAMES[v]}.")
        if v == VIEW_GTBOX:
            if graph["on"]:    toggle_graph()
            if iougraph["on"]: toggle_iou()
            show_gtbox_spectrum()                # auto-show the GT-box transform
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("1","2","3","4","5"):
            set_view(int(event.key)); return
        if event.key in ("9", "c"):
            set_view(VIEW_GTBOX); return
        if event.key == "v":
            set_view(state["view"] % N_VIEWS + 1); return
        if event.key == "d": jump_to_densest(); return
        if event.key == "b": jump_to_best_snr(); return
        if event.key == "g": toggle_graph(); return
        if event.key == "i": toggle_iou(); return
        if event.key in ("left","right","shift+left","shift+right","home","end"):
            step = 1 if "right" in event.key else -1
            if event.key.startswith("shift+"): step *= 10
            if event.key == "home":   target = 0
            elif event.key == "end":  target = max_idx
            else:                     target = state["frame_idx"] + step
            reload(target)

    def on_frame_submit(text):
        try:
            target = int(text.strip())
        except ValueError:
            info_text.set_text(f"Invalid frame index: {text!r}")
            fig.canvas.draw_idle(); return
        reload(target)

    frame_box.on_submit(on_frame_submit)

    drag = {"mode": None}
    TOL  = 0.02

    def _fig_xy(event):
        if event.x is None or event.y is None: return None, None
        return fig.transFigure.inverted().transform((event.x, event.y))

    def on_press(event):
        fx, fy = _fig_xy(event)
        if fx is None: return
        if abs(fy - layout["split_y"]) < TOL and 0.04 < fx < 0.98:
            drag["mode"] = "h"

    def on_motion(event):
        if drag["mode"] is None: return
        fx, fy = _fig_xy(event)
        if fx is None: return
        layout["split_y"] = min(0.80, max(0.16, fy))
        apply_layout()

    def on_release(event):
        drag["mode"] = None

    fig.canvas.mpl_connect("button_press_event",   on_click)
    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event",      on_key)
    plt.show()


if __name__ == "__main__":
    main()
