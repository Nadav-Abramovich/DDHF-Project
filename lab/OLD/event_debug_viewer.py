"""
Interactive event-only DDHF debugger for lab recordings (no RGB).

A trimmed sibling of code/debug_viewer_pooled.py that works purely from event
data — it reads a Metavision RAW recording (``data.raw``) plus its bias file
(``data.bias``) and needs NO RGB frames, no coordinates/GT, and no RGB-sync
offset. Everything is drawn on a black canvas the size of the event sensor, so
the display coordinates ARE the event coordinates (the affine is the identity).

The spectral detector itself is imported unchanged from
code/debug_viewer_pooled.py, so the green rotor mask, per-pixel/pooled/block/
component spectra and detection boxes are identical to the main viewer — this
file only swaps the data source (RAW events instead of an RGB+event dataset)
and the rendering background (black instead of an RGB frame).

About data.bias: a Metavision bias file is the camera configuration that was
active while the recording was made; those biases are already baked into the
events in data.raw. This tool therefore uses the biases for DISPLAY/reference
only (shown in the header and printed at startup) — there is no live camera to
re-apply them to when replaying a file.

Views (event-only):
    1 / VIEW_EVENTS   — events (red +, blue -) + green rotor mask + boxes  [default]
    2 / VIEW_HEATMAP  — per-pixel event-count heatmap
    3 / VIEW_POLARITY — polarity (+ green / - magenta / both white)

Keys:  SPACE play/pause at ~30 fps, 1-3 / 'v' cycle views, 'd' densest pixel,
       'b' best-SNR pixel, 'g' usable-pixel graph, ←/→ ±1 frame (Shift ±10),
       Home/End. Click a pixel for its spectrum.

Usage:
    python event_debug_viewer.py
    python event_debug_viewer.py --raw data.raw --bias data.bias --frame 40
    python event_debug_viewer.py --mode pool --window-ms 33.333 --drone air75
    python event_debug_viewer.py --mode components --gpu
"""
import os
import sys
import argparse
from collections import OrderedDict

if hasattr(sys.stdout, "reconfigure"):        # the help text has non-ASCII arrows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Prophesee / Metavision runtime setup (must precede the metavision import) ──
plugin_folder = r"C:\Program Files\Prophesee\lib\metavision\hal\plugins"
dll_directories = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_deps", "openeb_bin"),
    r"C:\Program Files\Prophesee\bin",
    r"C:\Program Files\Prophesee\third_party\bin",
    plugin_folder,
    r"C:\Program Files\Prophesee\lib\hdf5\plugin",
]
for directory in dll_directories:
    if os.path.exists(directory):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(directory)
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
os.environ["MV_HAL_PLUGIN_PATH"] = plugin_folder
os.environ["HDF5_PLUGIN_PATH"] = r"C:\Program Files\Prophesee\lib\hdf5\plugin"

# ── Locate the shared DDHF code (utils / ddhf_config / debug_viewer_pooled) ────
# (kept in lab/OLD - not used by any other script or launcher; it uses the
#  detector modules in ../code)
_HERE = os.path.dirname(os.path.abspath(__file__))           # lab/OLD


def _find_code_dir():
    """Directory holding the detector modules (this folder, lab/code)."""
    candidates = [os.path.join(os.path.dirname(_HERE), "code", "common")]   # lab/code/common
    for c in candidates:
        if os.path.exists(os.path.join(c, "debug_viewer_pooled.py")):
            return c
    raise FileNotFoundError(
        "Could not find the shared DDHF code (debug_viewer_pooled.py). Looked in:\n  "
        + "\n  ".join(candidates))


sys.path.insert(0, _find_code_dir())
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "camera"))     # hot_pixels

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from metavision_core.event_io import RawReader

# Reuse the EXACT detector + renderers from the main viewer (single source of
# truth). These functions read the module-level mode flags (_PIXELIZE/_POOLING/
# _COMPONENTS/_USE_GPU) on the debug_viewer_pooled module, which we set in main().
import debug_viewer_pooled as dv
from ddhf_config import build_cfg, add_cfg_cli_args, cfg_overrides_from_args
from utils_pooled import _extract_boxes, _gpu_available
import hot_pixels

MODE_FLAGS = dv.MODE_FLAGS
MODES = tuple(MODE_FLAGS)
DRONE_BANDS = dv.DRONE_BANDS

VIEW_EVENTS, VIEW_HEATMAP, VIEW_POLARITY = 1, 2, 3
N_VIEWS = 3
VIEW_NAMES = {
    VIEW_EVENTS:   "events (+ red / - blue) + rotor mask + boxes",
    VIEW_HEATMAP:  "event-count heatmap",
    VIEW_POLARITY: "polarity (+ green / - magenta / both white)",
}


# ── Bias file ─────────────────────────────────────────────────────────────────

def parse_bias(path):
    """Parse a Metavision bias file into an OrderedDict {name: int_value}.

    Lines look like ``299  % bias_diff`` (value, then a '%'-prefixed name). Blank
    and malformed lines are skipped. Returns an empty dict if the file is missing.
    """
    biases = OrderedDict()
    if not path or not os.path.exists(path):
        return biases
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or "%" not in line:
                continue
            val_str, _, name = line.partition("%")
            name = name.strip()
            try:
                val = int(float(val_str.strip()))
            except ValueError:
                continue
            if name:
                biases[name] = val
    return biases


def bias_summary(biases):
    """Compact one-line rendering of the biases, with the ``bias_`` prefix dropped."""
    if not biases:
        return "no bias file"
    return "  ".join(f"{k.replace('bias_', '')}={v}" for k, v in biases.items())


# ── Event stream (RAW, fully in memory) ───────────────────────────────────────

class EventStream:
    """Loads a Metavision RAW recording fully into memory and serves fixed-width
    time windows by timestamp. Lab recordings are short, so in-memory slicing is
    simpler and faster (arbitrary seek, no backward-recreate) than streaming."""

    def __init__(self, raw_path, hot_path=""):
        self.raw_path = raw_path
        reader = RawReader(raw_path, max_events=100_000_000)
        self.ev_height, self.ev_width = reader.get_size()
        chunks = []
        while not reader.is_done():
            e = reader.load_delta_t(1_000_000)     # 1 s chunks
            if e.size:
                chunks.append(e.copy())
        if chunks:
            self.events = np.concatenate(chunks)
        else:
            self.events = np.empty((0,), dtype=[("x", "<u2"), ("y", "<u2"),
                                                ("p", "<i2"), ("t", "<i8")])
        # A stuck pixel is recorded like any other event, and this sensor
        # exposes no I_DigitalEventMask, so it cannot be dropped at capture
        # time - it has to go here if it goes anywhere.
        #
        # What this does NOT do is recover a missed detection. This viewer
        # analyses whole windows per pixel/block with no subsampling, so a
        # defect at (20,420) cannot touch the spectrum of a block at (300,160).
        # What it buys is honest statistics: (20,420) was 54% of
        # rec_20260916_104120.raw, so every event count, Mev/s figure and
        # whole-frame spectrum was mostly a measurement of one dead pixel.
        if hot_path and self.events.size:
            mask = hot_pixels.Mask.load(hot_path, self.ev_width, self.ev_height)
            before = int(self.events.size)
            self.events = mask(self.events)
            dropped = before - int(self.events.size)
            print("[hot] %s: dropped %d of %d events (%.1f%%) from %d pixel(s)"
                  % (os.path.basename(hot_path), dropped, before,
                     100.0 * dropped / max(before, 1), len(mask.pixels)))
        self.t = self.events["t"].astype(np.int64)
        self.t_start = int(self.t[0]) if self.t.size else 0
        self.t_end = int(self.t[-1]) if self.t.size else 0
        self._black = np.zeros((self.ev_height, self.ev_width, 3), dtype=np.uint8)

    def max_frame_idx(self, window_us):
        if self.t_end <= 0:
            return 0
        return max(0, int(self.t_end // window_us))

    def load_window(self, frame_idx, window_us, verbose=True):
        """Return (base_frame, evs) for window [idx*w, (idx+1)*w). base_frame is a
        fresh black canvas (event coords == display coords, affine = identity)."""
        t0 = frame_idx * window_us
        t1 = t0 + window_us
        lo = int(np.searchsorted(self.t, t0, side="left"))
        hi = int(np.searchsorted(self.t, t1, side="left"))
        evs = self.events[lo:hi]
        if verbose:
            print(f"Window {frame_idx}: t=[{t0/1e6:.3f}s, {t1/1e6:.3f}s], evs={evs.size}")
        return self._black.copy(), evs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw",  type=str, default=os.path.join(os.path.dirname(_HERE), "data.raw"),
                        help="Metavision RAW event recording (default: ./data.raw).")
    parser.add_argument("--bias", type=str, default=os.path.join(os.path.dirname(_HERE), "data.bias"),
                        help="Bias file recorded alongside data.raw (default: ./data.bias).")
    parser.add_argument("--hot-pixels", type=str, default=None, metavar="FILE",
                        help="Drop the pixels listed in this .hotpx file (build one with "
                             "hot_pixels.py --scan, lens capped). Defaults to data/hot_pixels/cam.hotpx "
                             "when present; pass an empty string to disable.")
    parser.add_argument("--frame",     type=int,   default=0)
    parser.add_argument("--window-ms", type=float, default=33.333)
    # Detector hyperparameters (identical names/defaults to the batch/main viewer).
    add_cfg_cli_args(parser)
    parser.add_argument("--mode", choices=MODES, default="pixelize",
                        help="Spatial aggregation: 'pixelize' (BxB blocks, default), "
                             "'pool' (3x3 pooling), 'baseline' (per-pixel), or "
                             "'components' (8-connected components).")
    parser.add_argument("--drone", choices=sorted(DRONE_BANDS), default="all",
                        help="Restrict the comb search to one drone's bands (default 'all').")
    parser.add_argument("--snr-sig", type=float, default=None,
                        help="Spectrum-panel comb significance threshold (display only; default 8).")
    parser.add_argument("--flag-neighborhood", type=int, default=1,
                        help="When a pixel is detected, flag its NxN neighbourhood before box "
                             "extraction. 1 = off (just the pixel).")
    parser.add_argument("--gpu", action="store_true",
                        help="Run the spectral stage on the GPU (CuPy) if available.")
    args = parser.parse_args()

    if args.snr_sig is None:
        args.snr_sig = 8.0

    # Drive the shared detector's module-level mode flags.
    dv._USE_GPU = args.gpu
    dv._COMPONENTS, dv._PIXELIZE, dv._POOLING = MODE_FLAGS[args.mode]
    if args.gpu:
        ok = _gpu_available()
        prec = "fp32" if os.environ.get("DDHF_GPU_FP32", "1") != "0" else "fp64"
        print(f"[GPU] {'using CuPy GPU (' + prec + ')' if ok else 'NO usable GPU — CPU fallback'}")

    if not os.path.exists(args.raw):
        parser.error(f"RAW file not found: {args.raw}")

    biases = parse_bias(args.bias)
    print(f"[bias] {args.bias if os.path.exists(args.bias) else '(missing)'}: {bias_summary(biases)}")

    # Config from the shared preset + CLI overrides (window drives windowing + spectrum).
    overrides = cfg_overrides_from_args(args)
    overrides["window_ms"] = args.window_ms
    overrides["step_ms"] = args.window_ms
    cfg = build_cfg(**overrides)
    cfg.rotor_bands_hz = DRONE_BANDS[args.drone]

    mode_label = ("connected-components" if dv._COMPONENTS else
                  (f"pixelize {cfg.pixel_block}x{cfg.pixel_block}" if dv._PIXELIZE
                   else ("3x3 pooled" if dv._POOLING else "no pooling (per-pixel)")))
    window_us = int(round(args.window_ms * 1000.0))
    bin_hz = 1000.0 / args.window_ms
    print(f"[{mode_label}]  window={args.window_ms}ms  bin={bin_hz:.1f}Hz  "
          f"frequency_bins={cfg.frequency_bins}  max_freq={(cfg.frequency_bins-1)*bin_hz:.0f}Hz")

    # Identity affine: display coordinates ARE event coordinates (no RGB warp).
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    M_inv = dv.invert_affine_2x3(M)

    hot_path = args.hot_pixels
    if hot_path is None:
        _default = os.path.join(os.path.dirname(_HERE), "data", "hot_pixels", "cam.hotpx")
        hot_path = _default if os.path.exists(_default) else ""
    stream = EventStream(args.raw, hot_path)
    ev_height, ev_width = stream.ev_height, stream.ev_width
    max_idx = stream.max_frame_idx(window_us)
    print(f"event sensor={ev_width}x{ev_height}, duration={stream.t_end/1e6:.3f}s, "
          f"events={stream.events.size}, max frame idx={max_idx}")

    # Full-frame FOV outline (harmless border drawn by the reused renderers).
    fov_pts = [(0.0, 0.0), (ev_width - 1.0, 0.0),
               (ev_width - 1.0, ev_height - 1.0), (0.0, ev_height - 1.0), (0.0, 0.0)]

    state = {
        "frame_idx": max(0, min(args.frame, max_idx)),
        "view": VIEW_EVENTS,
        "base_frame": None, "boxes": None,
        "count_map": None, "sf_map": None, "snr_map": None,
        "rotor_mask": None, "evs": None, "cc_labels": None,
    }

    frame_cache: "OrderedDict[int, dict]" = OrderedDict()
    FRAME_CACHE_MAX = 24

    def compute_frame(idx):
        cached = frame_cache.get(idx)
        if cached is not None:
            frame_cache.move_to_end(idx)
            return cached
        base_frame, evs = stream.load_window(idx, window_us)
        cmap, sfm, snrm, f0m = dv.compute_pixel_stats(evs, ev_height, ev_width, cfg)
        cc_labels = None
        if dv._COMPONENTS:
            rmask, boxes, cc_labels = dv.detect_components_mask(evs, ev_height, ev_width, cfg)
        else:
            rmask = (~np.isnan(sfm) & (sfm <= cfg.tau_sf)
                     & ~np.isnan(snrm) & (snrm >= cfg.tau_comb))
            if args.flag_neighborhood > 1:
                from scipy.ndimage import binary_dilation as _bdil
                rmask = _bdil(rmask, structure=np.ones((args.flag_neighborhood,
                                                        args.flag_neighborhood), bool))
            boxes = _extract_boxes(rmask, np.nan_to_num(snrm), cfg)
        dv.print_f0_consensus(f0m, rmask)
        active = cmap[cmap > 0]
        if active.size:
            print(f"  total_events={int(evs.size):>7d}  active_px={active.size:>6d}  "
                  f"eligible={(~np.isnan(sfm)).sum()}  green={int(rmask.sum())}  boxes={len(boxes)}")
        fr = dict(base_frame=base_frame, evs=evs,
                  count_map=cmap, sf_map=sfm.astype(np.float32),
                  snr_map=snrm.astype(np.float32), rotor_mask=rmask,
                  boxes=boxes, cc_labels=cc_labels)
        frame_cache[idx] = fr
        if len(frame_cache) > FRAME_CACHE_MAX:
            frame_cache.popitem(last=False)
        return fr

    fr = compute_frame(state["frame_idx"])
    state.update(frame_idx=state["frame_idx"], **fr)

    def render_state():
        v = state["view"]
        if v == VIEW_HEATMAP:
            return dv.render_heatmap(state["base_frame"], state["count_map"], M, fov_pts)
        if v == VIEW_POLARITY:
            return dv.render_polarity(state["base_frame"], state["evs"], M, fov_pts)
        # VIEW_EVENTS: overlay with rotor mask + boxes (gt_rects=None -> magenta boxes).
        return dv.render_overlay(state["base_frame"], state["evs"], M, fov_pts,
                                 state["rotor_mask"], state["boxes"], gt_rects=None)

    def view_title(idx):
        v = state["view"]
        return (f"[{mode_label}] View {v}/{N_VIEWS} [{VIEW_NAMES[v]}] — "
                f"frame {idx}/{max_idx}  (click pixel for stats)")

    plt.rcParams.update({
        "axes.titlesize": 10, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig = plt.figure(figsize=(15, 9))
    ax_over = fig.add_axes([0.02, 0.30, 0.96, 0.63])
    ax_spec = fig.add_axes([0.08, 0.10, 0.88, 0.15])
    ax_graph = fig.add_axes([0.08, 0.10, 0.88, 0.15]); ax_graph.set_visible(False)

    im_over = ax_over.imshow(render_state())
    title_over = ax_over.set_title(view_title(state["frame_idx"]))
    ax_over.axis("off")

    param_text = fig.text(0.5, 0.998, "", ha="center", va="top",
                          family="monospace", fontsize=8)
    ax_spec.set_title("Click a pixel for its spectrum (frequency in Hz)")
    ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")

    info_text = fig.text(
        0.5, 0.005,
        "Click pixel for stats. 1-3/'v'=views, 'b'=best-SNR, 'd'=densest, "
        "'g'=usable-px graph. ←/→ ±1, Shift±10, Home/End.",
        ha="center", va="bottom", family="monospace", fontsize=8)

    tb_ax = fig.add_axes([0.88, 0.035, 0.08, 0.028])
    frame_box = TextBox(tb_ax, f"Go to frame (0..{max_idx}): ", initial=str(state["frame_idx"]))

    def update_param_text():
        bands_s = ",".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in cfg.rotor_bands_hz)
        param_text.set_text(
            f"[{mode_label}]  frame {state['frame_idx']}/{max_idx}  window={cfg.window_ms}ms  "
            f"min_events>={cfg.min_events}  SF<={cfg.tau_sf}  SNR>={cfg.tau_comb}  "
            f"f0 bands={bands_s} Hz  GREEN px={int(state['rotor_mask'].sum())}\n"
            f"biases: {bias_summary(biases)}")

    update_param_text()

    # ── click-a-pixel spectrum ────────────────────────────────────────────────
    def show_pixel(xe, ye):
        if graph["on"]:
            toggle_graph()
        cmap = state["count_map"]; sfm = state["sf_map"]; snrm = state["snr_map"]
        rmask = state["rotor_mask"]
        n_own = int(cmap[ye, xe])
        rotor = bool(rmask[ye, xe])
        if dv._COMPONENTS:
            power, freqs, n_pool = dv.component_power_spectrum(
                state["evs"], state.get("cc_labels"), xe, ye, cfg)
            spec_word = "COMPONENT"
            if power is not None:
                _pnd = power[1:] + 1e-12
                sf = float(np.exp(np.mean(np.log(_pnd))) / np.mean(_pnd))
                _o, snr = dv.best_comb(power, cfg)
            else:
                sf = snr = float("nan")
        else:
            sf = sfm[ye, xe]; snr = snrm[ye, xe]
            power, freqs = dv.pixel_power_spectrum(state["evs"], xe, ye, cfg)
            _blocks = dv._PIXELIZE or not dv._POOLING
            _B = (max(1, int(cfg.pixel_block)) if dv._PIXELIZE else 1) if _blocks else None
            spec_word = ("BLOCK" if dv._PIXELIZE else "POOLED" if dv._POOLING else "PIXEL")
            n_pool = 0
            if power is not None:
                evx = state["evs"]["x"].astype(np.int64); evy = state["evs"]["y"].astype(np.int64)
                if _blocks:
                    n_pool = int(((evx // _B == xe // _B) & (evy // _B == ye // _B)).sum())
                else:
                    n_pool = int(((np.abs(evx - xe) <= 1) & (np.abs(evy - ye) <= 1)).sum())
        ok_n = not np.isnan(sf)
        ok_sf = (not np.isnan(sf)) and sf <= cfg.tau_sf
        ok_snr = (not np.isnan(snr)) and snr >= cfg.tau_comb
        sf_s = f"{sf:.4f}" if not np.isnan(sf) else "n/a"
        snr_s = f"{snr:.4f}" if not np.isnan(snr) else "n/a"
        info_text.set_text(
            f"event ({xe},{ye})  own_events={n_own}  eligible=[{'PASS' if ok_n else 'FAIL'}]   "
            f"SF={sf_s} [{'PASS' if ok_sf else 'FAIL'} <={cfg.tau_sf}]   "
            f"SNR={snr_s} [{'PASS' if ok_snr else 'FAIL'} >={cfg.tau_comb}]   "
            f"=> {'GREEN' if rotor else 'not flagged'}")
        print(info_text.get_text())

        ax_spec.clear()
        ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")
        if power is None:
            ax_spec.set_title(f"event px ({xe},{ye}) -- no events in aggregation")
        else:
            ax_spec.plot(freqs[1:], power[1:], color="steelblue", lw=1.2)
            omega, snr_best = dv.best_comb(power, cfg)
            if omega > 0:
                significant = snr_best >= args.snr_sig
                c_fund, c_harm = ("red", "orange") if significant else ("0.45", "0.65")
                bin_hz_ = 1000.0 / cfg.window_ms
                hw_hz = cfg.harmonic_half_width * bin_hz_
                for m in range(1, cfg.num_harmonics + 1):
                    fh = m * omega * bin_hz_
                    if fh <= freqs[-1]:
                        ax_spec.axvspan(fh - hw_hz, fh + hw_hz, color=c_harm, alpha=0.12)
                        if m == 1:
                            ax_spec.axvline(fh, color=c_fund, ls="-", lw=1.8, alpha=0.9,
                                            label=f"fundamental {fh:.0f} Hz")
                        else:
                            ax_spec.axvline(fh, color=c_harm, ls="--", lw=1.0, alpha=0.6,
                                            label="harmonics" if m == 2 else None)
                ax_spec.legend(loc="upper right", fontsize=7)
                fund = omega * 1000.0 / cfg.window_ms
                verdict = (f"COMB: f0~{fund:.0f}Hz (bin {omega})" if significant else
                           f"NO significant comb — best ~{fund:.0f}Hz "
                           f"(SNR {snr_best:.1f} < {args.snr_sig:g})")
                ax_spec.set_title(f"{spec_word} spectrum — event px ({xe},{ye}), "
                                  f"own={n_own} ev / agg={n_pool} ev")
                ax_spec.text(0.01, 0.97, f"{verdict}\ncomb SNR = {snr_best:.2f}\nSF = {sf_s}",
                             transform=ax_spec.transAxes, ha="left", va="top",
                             family="monospace", fontsize=7,
                             bbox=dict(boxstyle="round",
                                       fc="white" if significant else "0.92", ec="0.7", alpha=0.85))
            else:
                ax_spec.set_title(f"{spec_word} spectrum — event px ({xe},{ye}), "
                                  f"own={n_own} ev / agg={n_pool} ev  (no comb, SF={sf_s})")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is ax_graph:
            if event.xdata is not None:
                reload(int(round(event.xdata)))
            return
        if event.inaxes is not ax_over:
            return
        if event.xdata is None or event.ydata is None:
            return
        ve = M_inv @ np.array([round(event.xdata), round(event.ydata), 1.0], dtype=np.float32)
        xe = int(round(float(ve[0]))); ye = int(round(float(ve[1])))
        if not (0 <= xe < ev_width and 0 <= ye < ev_height):
            info_text.set_text(f"event ({xe},{ye}) -- outside sensor")
            fig.canvas.draw_idle(); return
        show_pixel(xe, ye)

    def jump_to_densest():
        cmap = state["count_map"]
        if cmap is None or cmap.max() <= 0:
            return
        ye, xe = np.unravel_index(int(np.argmax(cmap)), cmap.shape)
        print(f"densest pixel: event ({int(xe)},{int(ye)}) with {int(cmap[ye, xe])} own events")
        show_pixel(int(xe), int(ye))

    def jump_to_best_snr():
        snrm = state["snr_map"]
        if snrm is None or np.all(np.isnan(snrm)):
            return
        ye, xe = np.unravel_index(int(np.nanargmax(snrm)), snrm.shape)
        print(f"best-SNR pixel: event ({int(xe)},{int(ye)}) SNR={snrm[ye, xe]:.1f}")
        show_pixel(int(xe), int(ye))

    # ── usable-pixel-count graph (computed from the in-memory events) ──────────
    GRAPH_THRS = tuple(sorted({cfg.min_events, 6, 15, 30}))
    graph = {"on": False, "counts": None, "marker": None}

    def compute_usable_counts():
        print(f"Scanning {max_idx+1} windows for usable-pixel counts ...")
        counts = np.zeros((len(GRAPH_THRS), max_idx + 1), dtype=np.int32)
        t = stream.t
        for i in range(max_idx + 1):
            lo = int(np.searchsorted(t, i * window_us, side="left"))
            hi = int(np.searchsorted(t, (i + 1) * window_us, side="left"))
            if hi > lo:
                e = stream.events[lo:hi]
                pid = e["y"].astype(np.int64) * ev_width + e["x"].astype(np.int64)
                _, c = np.unique(pid, return_counts=True)
                for j, thr in enumerate(GRAPH_THRS):
                    counts[j, i] = int((c >= thr).sum())
        return counts

    def toggle_graph():
        if not graph["on"] and graph["counts"] is None:
            graph["counts"] = compute_usable_counts()
            xs = np.arange(graph["counts"].shape[1])
            colors = ["0.7", "steelblue", "darkorange", "crimson"]
            for j, thr in enumerate(GRAPH_THRS):
                ax_graph.plot(xs, graph["counts"][j], lw=0.8, color=colors[j % len(colors)],
                              label=f">={thr} ev")
            ax_graph.axhline(cfg.min_box_area, color="red", ls=":", lw=0.8,
                             label=f"min_box_area={cfg.min_box_area}")
            graph["marker"] = ax_graph.axvline(state["frame_idx"], color="0.3", lw=1.2)
            ax_graph.set_xlim(0, max(1, xs.size - 1))
            ax_graph.set_yscale("symlog", linthresh=10)
            ax_graph.set_xlabel("frame"); ax_graph.set_ylabel("pixels")
            ax_graph.set_title("Pixels with enough events, per window — click to jump")
            ax_graph.legend(loc="upper right", fontsize=7, ncol=len(GRAPH_THRS) + 1)
        graph["on"] = not graph["on"]
        ax_graph.set_visible(graph["on"])
        ax_spec.set_visible(not graph["on"])
        fig.canvas.draw_idle()

    def reload(new_idx):
        new_idx = max(0, min(new_idx, max_idx))
        if new_idx == state["frame_idx"] and state["count_map"] is not None:
            return
        fr = compute_frame(new_idx)
        state.update(frame_idx=new_idx, **fr)
        im_over.set_data(render_state())
        ax_spec.clear()
        ax_spec.set_title("Click a pixel for its spectrum (frequency in Hz)")
        ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Power")
        title_over.set_text(view_title(new_idx))
        update_param_text()
        info_text.set_text(f"Loaded frame {new_idx} — GREEN px={int(fr['rotor_mask'].sum())}.")
        if graph["marker"] is not None:
            graph["marker"].set_xdata([new_idx, new_idx])
        frame_box.eventson = False
        frame_box.set_val(str(new_idx))
        frame_box.eventson = True
        fig.canvas.draw_idle()

    def set_view(v):
        if v == state["view"]:
            return
        state["view"] = v
        im_over.set_data(render_state())
        title_over.set_text(view_title(state["frame_idx"]))
        info_text.set_text(f"View {v}: {VIEW_NAMES[v]}.")
        fig.canvas.draw_idle()

    # -- playback -----------------------------------------------------------
    player = {"timer": None}

    def compute_frame_fast(idx):
        """Events only: no NDFT, no component pass, no console output.

        compute_pixel_stats IS the per-frame cost - a few hundred ms at these
        event counts, which is what pins playback near 2 fps. While the frames
        are running you are watching the event stream, not reading a rotor
        mask, so skip the detector entirely: slicing the in-memory array and
        histogramming it is microseconds. Frames already in the cache come back
        complete, so stretches you have scrubbed keep their masks and boxes.
        """
        cached = frame_cache.get(idx)
        if cached is not None:
            frame_cache.move_to_end(idx)
            return cached
        base_frame, evs = stream.load_window(idx, window_us, verbose=False)
        cmap = np.zeros((ev_height, ev_width), np.int32)
        if evs.size:
            np.add.at(cmap, (evs["y"].astype(np.int64),
                             evs["x"].astype(np.int64)), 1)
        nan = np.full((ev_height, ev_width), np.nan, np.float32)
        return dict(base_frame=base_frame, evs=evs, count_map=cmap,
                    sf_map=nan, snr_map=nan,
                    rotor_mask=np.zeros((ev_height, ev_width), bool),
                    boxes=[], cc_labels=None)

    def stop_play(reason=""):
        was_playing = player["timer"] is not None
        if was_playing:
            player["timer"].stop()
            player["timer"] = None
        if reason:
            info_text.set_text("Playback stopped: %s." % reason)
        if was_playing:
            # The fast path left this frame with no detector output. Recompute
            # it so whatever you paused on shows its mask and boxes again.
            state["count_map"] = None          # defeat reload's early-out
            reload(state["frame_idx"])
        elif reason:
            fig.canvas.draw_idle()

    def _advance():
        if state["frame_idx"] >= max_idx:
            stop_play("reached the last frame")
            return
        idx = state["frame_idx"] + 1
        state.update(frame_idx=idx, **compute_frame_fast(idx))
        im_over.set_data(render_state())
        title_over.set_text(view_title(idx))
        if graph["marker"] is not None:
            graph["marker"].set_xdata([idx, idx])
        fig.canvas.draw_idle()

    def toggle_play():
        """Space: run the frames forward at ~30 fps, or pause.

        Playback runs the cheap path (see compute_frame_fast) - the detector is
        skipped, so the green mask and boxes are absent while the frames move
        and come back the moment you pause. Pausing re-runs the full detector on
        whatever frame you stopped on, which is the one you actually want it for.
        """
        if player["timer"] is not None:
            stop_play("paused at frame %d" % state["frame_idx"])
            return
        if state["frame_idx"] >= max_idx:
            reload(0)
        timer = fig.canvas.new_timer(interval=33)   # ~30 fps
        timer.add_callback(_advance)
        player["timer"] = timer
        timer.start()
        info_text.set_text("Playing at ~30 fps (events only, detector off) - "
                           "space to pause and re-run the detector here.")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            toggle_play(); return
        if event.key in ("1", "2", "3"):
            set_view(int(event.key)); return
        if event.key == "v":
            set_view(state["view"] % N_VIEWS + 1); return
        if event.key == "d":
            jump_to_densest(); return
        if event.key == "b":
            jump_to_best_snr(); return
        if event.key == "g":
            toggle_graph(); return
        if event.key in ("left", "right", "shift+left", "shift+right", "home", "end"):
            stop_play()                 # manual stepping takes over from playback
            step = 1 if "right" in event.key else -1
            if event.key.startswith("shift+"):
                step *= 10
            if event.key == "home":
                target = 0
            elif event.key == "end":
                target = max_idx
            else:
                target = state["frame_idx"] + step
            reload(target)

    def on_frame_submit(text):
        stop_play()
        try:
            target = int(text.strip())
        except ValueError:
            info_text.set_text(f"Invalid frame index: {text!r}")
            fig.canvas.draw_idle(); return
        reload(target)

    frame_box.on_submit(on_frame_submit)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"event sensor={ev_width}x{ev_height}, max frame idx={max_idx}")
    plt.show()


if __name__ == "__main__":
    main()
