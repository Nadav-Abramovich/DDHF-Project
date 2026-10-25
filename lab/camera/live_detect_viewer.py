#!/usr/bin/env python
"""
Live SilkyEvCam viewer with the DDHF rotor detector drawn on top.

Shows the event stream as it arrives and colours the pixels the project's own
detector flags as rotor: code/utils.py's _detect_rotor_pixels runs the DDHF
pipeline (per-pixel NDFT -> spectral flatness -> harmonic-comb SNR) on every
completed window, and the pixels that pass every gate are painted by their
comb-SNR. So what you see highlighted is exactly what the offline pipeline
would keep, at the biases currently on the camera.

Mouse
    left-click the view ... probe that pixel's spectrum (opens "Pixel spectrum")
    right-click the view .. stop probing
    in "Biases" ........... drag a bar, click -/+, right-click a row for nominal

Four windows:
  * "SilkyEvCam - detections"  events, with detections coloured over them
  * "Detector"                 live gates: min_events, tau_comb, tau_sf, pool,
                               detection hold, and the accumulation time
  * "Pixel spectrum"           opens when you left-click a pixel in the view:
                               that cell's NDFT power spectrum, live, with the
                               searchable bands shaded, the winning comb's
                               harmonics marked, and which gate rejected it
                               spelled out. Right-click the view to close the
                               probe. This is the tool for "why is my rotor
                               not detected" - it says whether the periodicity
                               is in the events at all, and if it is, exactly
                               which threshold threw it away.
  * "Biases"                   every sensor bias, with a trackbar and a row of
                               its own: drag the bar, click -/+, right-click a
                               row for that bias's nominal value, and use the
                               footer buttons for nominal / startup / load /
                               save / recommended-vs-full range. The value
                               shown is the readback, not what was asked for -
                               the DAC quantises - and it turns amber when it
                               sits outside the recommended range.

What the frequency readout is
    Whenever a detection is confirmed the header shows the blade-pass
    fundamental of the harmonic comb that won it - the same omega the comb-SNR
    was scored on - and each box is labelled with its own. It is quantised to
    the NDFT bin, 1000/window_ms Hz (30 Hz at the 33.3 ms window), because the
    comb search only tests fundamentals on that grid. The same number is
    printed to the console when a detection starts and once a second while it
    lasts, so it survives the overlay fading out. --blades turns it into rpm;
    blade-pass = blades x rotor rate, so the rpm is only as right as --blades.

Setting biases
    Three ways, all equivalent once running: --bias <file>, one flag per bias
    (--bias-diff-on 420 ...), or the Biases window. Flags are applied after
    the file, so they override it. Values are clamped to the sensor's
    recommended range unless --bypass-range-check is passed, which the
    window's "range" button also toggles at any time.

Why two worker threads
    Nothing that can block is allowed in the draw loop, because everything the
    window does - repainting, the trackbars, the keys, the mouse - happens
    there. Two things block. The SDK's next() waits for the camera to produce
    something, which on a quiet scene is a long wait and froze the viewer
    exactly when the least was happening; EventReader now does that waiting.
    And the detector cannot keep up (below); it gets the other thread.

Why the detector runs on its own thread
    It cannot keep up. The NDFT is linear in events - 11 ms at ~6k, 160 ms at
    40k, 3.8 s at 300k - while a window only covers 33 ms, so a busy window
    costs several times the time it represents. Run in the draw loop that
    froze everything for the length of every window. It now gets a one-slot
    mailbox: the newest completed window is analysed and any window that piles
    up behind it is dropped, which keeps the display responsive at the cost of
    analysing fewer windows. The header shows how many are being dropped;
    lower --max-window-events or raise --pool to drop fewer, or
    --sync-detector to go back to analysing every window in the draw loop.

Why pooling and hold default on
    A live rotor is far sparser than the cached recordings the preset was tuned
    against - measured ~1-4 events per pixel per 33 ms window against
    min_events=16, so per-pixel grouping passes nothing at all. BxB pooling
    multiplies events per group by B^2. And detections flicker: the same rotor
    lights a different cell from window to window (2-5 hits per cell out of
    60), so each detection is held for a few windows and faded out, otherwise
    the overlay strobes and you cannot see what was found.

Keys
    up/down .......... select a bias          left/right ... selected bias -1/+1
    pgup/pgdn ........ selected bias +-10     , / . ....... selected bias -+50
    a / z ............ accumulation -/+       p ........... cycle palette
    d ................ restore startup biases n ........... nominal biases
    l ................ load the --bias file   s ........... save biases
    g ................ bias range: recommended <-> full allowed
    b ................ boxes on/off           x ........... detections on/off
    u ................ show/hide unconfirmed (noise) cells
    r ................ start/stop RAW recording (writes a .bias sidecar too)
    c ................ clear                  SPACE ....... freeze
    h ................ print this help        q / ESC ..... quit

Usage
    python live_detect_viewer.py
    python live_detect_viewer.py --bias bias_opt.bias      # optimised biases
    python live_detect_viewer.py --bias-diff-on 420 --bias-hpf 1448
    python live_detect_viewer.py --nominal --bias-refr 1506 --bypass-range-check
    python live_detect_viewer.py --pool 6 --min-events 8 --blades 3
    python live_detect_viewer.py --hold 8 --scale 1.5
"""
import os
import sys
import time
import copy
import json
import argparse
import datetime
import threading
from collections import deque

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB  = os.path.dirname(_HERE)                       # lab/ (data/, runs/, compare_replay)
for _p in (_HERE, os.path.join(_LAB, "code", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import silky_bias_studio as studio        # DLL/plugin bootstrap + HAL helpers
import cv2
from metavision_core.event_io import EventsIterator
from utils import (DDHFConfig, _detect_rotor_pixels, _extract_boxes,
                   _normalise_polarity, _normalise_timestamps,
                   _ndft_power_batch, _spectral_flatness_batch, _comb_snr_batch)
import ddhf_config
import hot_pixels

COMPONENT_GATES = ("min_rotor_pixels", "min_fill_ratio", "min_snr_sum",
                   "min_snr_avg", "min_box_area")


def load_working_point(path):
    """DDHFConfig fields and component gates from a working-point JSON."""
    with open(path) as fh:
        wp = json.load(fh)
    params = wp.get("params", wp)
    preset = {k: v for k, v in params.items() if k in ddhf_config._PRESET}
    gates = {k: float(params[k]) for k in COMPONENT_GATES if k in params}
    return preset, gates, wp

BIAS_NAMES = studio.BIAS_NAMES
NOMINAL = studio.NOMINAL
VIEW_WIN = "SilkyEvCam - detections"
CTRL_WIN = "Detector"
BIAS_WIN = "Biases"
SPEC_WIN = "Pixel spectrum"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Spectrum panel geometry: the plot area inside the window.
SP_W, SP_H = 760, 380
SP_L, SP_R, SP_T, SP_B = 52, 12, 96, 30

# Bias panel geometry, in the bias window's own pixels. Kept as constants
# rather than measured while drawing because the mouse callback has to hit-test
# the same rectangles the renderer drew, and the two run in different calls.
BP_W = 620                     # panel width
BP_ROW_H = 32                  # one bias per row
BP_TOP = 30                    # first row's top edge
BP_X_NAME = 10
BP_X_VALUE = 138
BP_X_MINUS = (186, 210)
BP_X_TRACK = (218, 474)
BP_X_PLUS = (480, 504)
BP_X_RANGE = 512
BP_BTN_W = 104
BP_BTN_H = 24
BP_BUTTONS = (("nominal", "nominal"), ("startup", "startup"),
              ("load file", "load"), ("save", "save"), ("range", "bypass"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Live event viewer with DDHF rotor detections drawn on top.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--bias", default="",
                   help=".bias file applied at startup and reloaded with l")
    biases = p.add_argument_group(
        "camera biases",
        "One flag per sensor bias, e.g. --bias-diff-on 420. Applied after "
        "--bias, so a flag overrides that file's value for the same bias. "
        "Values are clamped to the sensor's recommended range unless "
        "--bypass-range-check is given. Ignored when replaying with -i.")
    for _name in BIAS_NAMES:
        biases.add_argument("--" + _name.replace("_", "-"), dest=_name,
                            type=int, default=None, metavar="N",
                            help="%s (nominal %d)"
                                 % (studio.HINTS.get(_name, _name),
                                    NOMINAL.get(_name, 0)))
    biases.add_argument("--nominal", action="store_true",
                        help="start from Prophesee's nominal biases (the n key) "
                             "instead of whatever the camera came up with; the "
                             "individual flags still override it")
    biases.add_argument("--bypass-range-check", action="store_true",
                        help="allow biases outside the sensor's recommended "
                             "range - the full allowed range becomes settable, "
                             "from the flags and from the trackbars/keys")
    p.add_argument("--pool", type=int, default=4,
                   help="BxB pooling before the NDFT (default 4; 1 = per-pixel)")
    p.add_argument("--hold", type=int, default=6,
                   help="windows a detection stays on screen, fading (default 6)")
    p.add_argument("--accumulation-time", type=float, default=8.0,
                   help="displayed event history in ms (default 8)")
    p.add_argument("--refresh-time", type=float, default=10.0,
                   help="event slice length fetched per frame, ms (default 10)")
    p.add_argument("--scale", type=float, default=1.0, help="display scale")
    p.add_argument("--blades", type=int, default=2,
                   help="blades per rotor, used only to turn the detected "
                        "blade-pass frequency into an rpm readout (default 2). "
                        "0 hides the rpm")
    p.add_argument("--working-point", default="",
                   help="working-point JSON. Sets the detector gates AND the "
                        "component gates (min_rotor_pixels, min_fill_ratio, "
                        "min_snr_sum, min_snr_avg) that DDHFConfig has no "
                        "fields for - these are what suppress false positives")
    p.add_argument("--hide-unconfirmed", action="store_true",
                   help="draw only detections that survive the blob gate, "
                        "hiding the isolated cells that are almost all noise")
    p.add_argument("--no-boxes", action="store_true",
                   help="do not draw the detector's bounding boxes")
    p.add_argument("--max-window-events", type=int, default=20000,
                   help="events per window handed to the NDFT; denser windows "
                        "are uniformly subsampled to this (default 20000). "
                        "NDFT cost is linear in events - measured 11 ms at "
                        "~6k events, 160 ms at 40k, 3.8 s at 300k - against a "
                        "window that only covers 33 ms, so this sets how many "
                        "windows get analysed rather than dropped. Raise it "
                        "for fidelity, lower it to analyse more windows; the "
                        "header shows ms/window and the percentage dropped.")
    p.add_argument("--sync-detector", action="store_true",
                   help="run the detector in the draw loop instead of on its "
                        "own thread. Analyses every window, at the cost of "
                        "freezing the display for the length of each one")
    p.add_argument("-i", "--input", default="",
                   help="replay a Metavision RAW file instead of the live "
                        "camera; biases are read-only in that case")
    p.add_argument("--record", action="store_true",
                   help="start recording immediately, as if r were pressed")
    p.add_argument("--serial", default="")
    p.add_argument("--record-dir", default="",
                   help="where r writes its .raw and .bias (default: cwd)")
    p.add_argument("--hot-pixels", default=None, metavar="FILE",
                   help="drop the pixels listed in this .hotpx file (build one "
                        "with hot_pixels.py --scan). Defaults to cam.hotpx when "
                        "it exists; pass an empty string to disable. Worth it "
                        "because --max-window-events subsamples uniformly: one "
                        "defect at 54%% of the stream costs the rotor half of "
                        "every window handed to the NDFT")
    ddhf_config.add_cfg_cli_args(p)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class Detector:
    """Runs DDHF on completed windows and keeps a fading map of what it found.

    The score map is kept at the pooled resolution and expanded for display, so
    a detection paints the whole BxB cell it was measured over - that cell is
    the group the detector actually tested, not a single pixel.

    The analysis runs on its own thread, because it cannot keep up and must not
    be allowed to hold the display hostage while it tries. The NDFT is linear
    in events - 11 ms at ~6k, 160 ms at 40k, 3.8 s at 300k - so a busy window
    at the default cap costs several times the 33 ms it covers. Run in the draw
    loop that froze everything: events, panning, trackbars and keys all stopped
    for the length of every window. So `submit` only leaves the window in a
    one-slot mailbox and returns; the worker takes the newest one waiting and
    DROPS whatever it could not get to. Dropping is the right trade for a live
    viewer - a stale detection is worth less than a responsive one - and
    `dropped` is on screen so it is never silently behind.
    """

    def __init__(self, width, height, cfg, pool, hold, max_window_events=300000,
                 gates=None, threaded=True):
        self.width = width
        self.height = height
        self.cfg = cfg
        self.pool = max(1, pool)
        self.hold = max(1, hold)
        self.max_window_events = max_window_events
        self.gates = dict(gates or {})
        self.gates.setdefault("min_box_area", cfg.min_box_area)
        # The comb search only tests fundamentals on the NDFT grid, so every
        # frequency it can report is a multiple of this - it is the readout's
        # resolution, not a rounding of a finer estimate.
        self.bin_hz = 1000.0 / cfg.window_ms
        # Guards every map below: the worker writes them, the draw loop reads
        # them. Held only around the assignments, never around the NDFT.
        self.lock = threading.RLock()
        self.probe = None             # sensor (x, y) whose spectrum to publish
        self.probe_result = None      # what _probe_spectrum last measured there
        self._inbox = None
        self._wake = threading.Event()
        self._closing = False
        self._gen = 0                 # bumped by reset(); stale results are dropped
        self.dropped = 0
        self.reset()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._serve, name="ddhf",
                                            daemon=True)
            self._thread.start()

    # -- handoff ------------------------------------------------------------
    def submit(self, x, y, p, t):
        """Hand over a completed window. Never blocks; may drop an older one.

        The fade is advanced here rather than in the worker so that a detection
        ages by wall-clock windows even while windows are being dropped -
        otherwise a backlog would freeze the overlay instead of fading it.
        """
        with self.lock:
            self.age = np.minimum(self.age + 1, self.hold + 1)
            self.freq_age = min(self.freq_age + 1, self.hold + 1)
            self.windows += 1
            gen = self._gen
        if self._thread is None:
            self._analyse(x, y, p, t, gen)         # --sync-detector
            return
        with self.lock:
            if self._inbox is not None:
                self.dropped += 1
            self._inbox = (x, y, p, t, gen)
        self._wake.set()

    def _serve(self):
        while not self._closing:
            self._wake.wait(0.2)
            self._wake.clear()
            while not self._closing:
                with self.lock:
                    job, self._inbox = self._inbox, None
                if job is None:
                    break
                try:
                    self._analyse(*job)
                except Exception as exc:           # never take the thread down
                    print("[det] window failed: %s" % exc)

    def close(self):
        self._closing = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset(self):
        with self.lock:
            self._gen += 1
            self._inbox = None
            self.pw = (self.width + self.pool - 1) // self.pool
            self.ph = (self.height + self.pool - 1) // self.pool
            self.age = np.full((self.ph, self.pw), self.hold + 1, np.int16)
            self.snr = np.zeros((self.ph, self.pw), np.float32)
            self.freq = np.zeros((self.ph, self.pw), np.float32)
            self.stuck = np.zeros((self.ph, self.pw), bool)
            self.confirmed = np.zeros((self.ph, self.pw), bool)
            self.boxes = []
            self.box_freq = []
            self.windows = 0
            self.dropped = 0
            self.last_count = 0
            self.last_peak = 0.0
            self.last_freq = 0.0    # last analysed window's blade-pass freq, Hz
            self.held_freq = 0.0    # last confirmed one, kept while freq_age <= hold
            self.freq_age = self.hold + 1
            self.last_ms = 0.0
            self.overloaded = 0

    def set_pool(self, pool):
        if pool != self.pool:
            self.pool = max(1, pool)
            self.reset()

    def mark_stuck(self, x, y):
        """Exclude a pixel that fires on its own - a sensor defect, not a rotor.

        This camera has one at (20,420) emitting ~30 kev/s, 14% of the sensor's
        whole output; it clears min_events every window and its steady firing
        scores a comb-SNR above tau_comb, so without this it lights up as a
        permanent false detection.
        """
        with self.lock:
            self.stuck[y // self.pool, x // self.pool] = True

    def _analyse(self, x, y, p, t, gen):
        """One window, on the worker thread. The NDFT runs outside the lock."""
        started = time.time()
        if len(t) < self.cfg.min_events:
            with self.lock:
                self.last_count = 0
                self.last_freq = 0.0
                self.overloaded = 0
                if self.probe is not None:
                    # Say so rather than leaving the last spectrum up: a stale
                    # plot during a dead window is exactly the wrong answer to
                    # "is anything reaching this pixel".
                    self.probe_result = {
                        "pixel": tuple(self.probe), "cell": (0, 0),
                        "pool": self.pool, "n": 0, "power": None, "sf": None,
                        "snr": 0.0, "omega": 0, "freq": 0.0, "passed": False,
                        "verdict": "whole window had %d events - nothing to "
                                   "transform" % len(t)}
            return
        self.overloaded = 0
        if len(t) > self.max_window_events:
            # Subsample rather than skip: the NDFT costs linearly in events
            # (11 ms at ~6k, 3.8 s at 300k), and a thinned point process keeps
            # the periodicity that the comb SNR is looking for - it just
            # measures it against more shot noise. Skipping outright would
            # blank the overlay exactly when the scene is busiest.
            self.overloaded = len(t)
            keep = np.random.default_rng(0).choice(len(t),
                                                   self.max_window_events,
                                                   replace=False)
            keep.sort()
            x, y, p, t = x[keep], y[keep], p[keep], t[keep]
        px = (x // self.pool).astype(np.int64)
        py = (y // self.pool).astype(np.int64)
        tn = _normalise_timestamps(t, window_span=self.cfg.window_ms * 1000.0)
        pol = _normalise_polarity(p)
        ph, pw = self.ph, self.pw
        probe = self.probe
        if probe is not None:
            self._probe_spectrum(probe, px, py, pol, tn)
        rotor, scores, freqs = _detect_rotor_pixels(
            px, py, pol, tn, ph, pw, self.cfg, return_freq=True)
        rotor &= ~self.stuck
        conf = self._confirm(rotor, scores)
        boxes = ([] if not conf.any()
                 else _extract_boxes(conf, scores, self._pooled_cfg()))
        box_freq = [self._cell_freq(self._box_mask(b, conf), freqs, scores)
                    for b in boxes]
        freq = self._cell_freq(conf, freqs, scores) if conf.any() else 0.0
        elapsed = (time.time() - started) * 1000.0
        with self.lock:
            if gen != self._gen:
                return    # pool changed under us; these maps are the wrong shape
            self.age[rotor] = 0
            self.snr[rotor] = scores[rotor]
            self.freq[rotor] = freqs[rotor]
            self.confirmed[rotor] = conf[rotor]
            self.boxes = boxes
            self.box_freq = box_freq
            self.last_freq = freq
            if freq > 0.0:
                self.held_freq = freq
                self.freq_age = 0
            self.last_count = int(rotor.sum())
            self.last_peak = float(scores[rotor].max()) if self.last_count else 0.0
            self.last_ms = elapsed

    def _probe_spectrum(self, probe, px, py, pol, tn):
        """NDFT of one pooled cell, plus how it fared at each gate.

        Computed here, inside the window the detector is already analysing,
        rather than in a pass of its own: the events have by this point been
        pooled, polarity-normalised, time-normalised and - on a dense window -
        SUBSAMPLED. A spectrum taken from the raw events instead would be a
        different spectrum from the one that was scored, and would explain the
        wrong thing.

        The cell is derived from the sensor pixel every window rather than
        stored, so the probe stays put when the pool size changes under it.
        """
        cfg = self.cfg
        cx, cy = probe[0] // self.pool, probe[1] // self.pool
        sel = (px == cx) & (py == cy)
        n = int(sel.sum())
        out = {"pixel": tuple(probe), "cell": (int(cx), int(cy)),
               "pool": self.pool, "n": n, "power": None, "sf": None,
               "snr": 0.0, "omega": 0, "freq": 0.0, "verdict": "",
               "passed": False}
        if n < 2:
            out["verdict"] = "no events in this cell this window"
            with self.lock:
                self.probe_result = out
            return
        # One group, so group_starts is just [0].
        power = _ndft_power_batch(None, pol[sel], tn[sel],
                                  np.zeros(1, np.int64), cfg.frequency_bins)
        sf = float(_spectral_flatness_batch(power[1:])[0])
        snr, omega = _comb_snr_batch(power, cfg, return_omega=True)
        snr, omega = float(snr[0]), int(omega[0])
        out.update(power=power[:, 0], sf=sf, snr=snr, omega=omega,
                   freq=omega * self.bin_hz)
        # First gate that rejects it, in the order the detector applies them.
        if n < cfg.min_events:
            out["verdict"] = ("REJECTED at min_events: %d events, needs %d"
                              % (n, cfg.min_events))
        elif sf > cfg.tau_sf:
            out["verdict"] = ("REJECTED at tau_sf: flatness %.3f > %.2f "
                              "(spectrum too flat - looks like noise)"
                              % (sf, cfg.tau_sf))
        elif omega == 0:
            out["verdict"] = ("REJECTED: no candidate fundamental inside "
                              "rotor_bands_hz - nothing was even tested")
        elif snr < cfg.tau_comb:
            out["verdict"] = ("REJECTED at tau_comb: comb SNR %.1f < %.1f"
                              % (snr, cfg.tau_comb))
        else:
            out["verdict"] = ("PASSED both gates at %.0f Hz (comb SNR %.1f)"
                              % (out["freq"], snr))
            out["passed"] = True
        with self.lock:
            self.probe_result = out

    def _cell_freq(self, mask, freqs, scores):
        """One blade-pass frequency for a set of cells, in Hz.

        The SNR-weighted MEDIAN, not the mean: cells over the same rotor
        occasionally lock onto a harmonic or a neighbouring bin, and a mean
        would land between two candidates on a frequency no cell actually
        reported. The median returns a frequency that was genuinely measured,
        and weighting by comb-SNR lets the strongest cells decide which one.
        """
        f = freqs[mask]
        w = scores[mask]
        ok = f > 0.0
        if not ok.any():
            return 0.0
        f, w = f[ok], np.maximum(w[ok], 1e-6)
        order = np.argsort(f)
        f, w = f[order], w[order]
        cum = np.cumsum(w)
        return float(f[np.searchsorted(cum, 0.5 * cum[-1])])

    @staticmethod
    def _box_mask(box, conf):
        mask = np.zeros_like(conf)
        mask[int(box.y1):int(box.y2) + 1, int(box.x1):int(box.x2) + 1] = True
        return mask & conf

    def freq_readout(self, blades=0):
        """(text, fresh) for the held frequency, or ("", False) if none."""
        with self.lock:
            age, held = self.freq_age, self.held_freq
        if age > self.hold or held <= 0.0:
            return "", False
        txt = "%.0f Hz blade-pass (+-%.0f Hz bin)" % (held, self.bin_hz)
        if blades > 0:
            txt += "  ~%.0f rpm at %d blades" % (60.0 * held / blades, blades)
        return txt, age == 0

    def _pooled_cfg(self):
        """cfg with the area gate expressed in pooled cells, not sensor pixels.

        min_box_area is 50 sensor pixels; run against a BxB-pooled mask it was
        demanding 50 CELLS, i.e. 50*B^2 = 800 pixels at B=4. That is why the
        box count stayed 0 while hundreds of cells were passing.
        """
        cfg = copy.copy(self.cfg)
        cfg.min_box_area = 1      # _confirm already applied the real gates
        return cfg

    def _confirm(self, rotor, scores):
        """Keep only components that pass the working point's component gates.

        Applied at sensor resolution: pixelize broadcasts a block's value to
        every pixel of the block, so min_rotor_pixels and min_snr_sum are
        counted in sensor pixels. Gating on the pooled mask instead would make
        min_rotor_pixels=6 mean 6*pool^2 pixels and min_snr_sum pool^2 times
        easier to reach - neither gate would mean what it was tuned to.

        The pixel gates alone pass about 1% of pure-noise groups per window
        (measured on synthetic Poisson events), which across thousands of
        groups is a steady rain of isolated specks. These gates are what the
        offline pipeline uses to drop them.
        """
        empty = np.zeros_like(rotor)
        if not rotor.any():
            return empty
        P = self.pool
        if P > 1:
            big_r = np.repeat(np.repeat(rotor, P, 0), P, 1)[:self.height, :self.width]
            big_s = np.repeat(np.repeat(scores, P, 0), P, 1)[:self.height, :self.width]
        else:
            big_r, big_s = rotor, scores
        lab, n = ndimage.label(big_r, structure=np.ones((3, 3)))
        if n == 0:
            return empty
        keep = np.zeros(n + 1, bool)
        g = self.gates
        for lid, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            sub = lab[sl] == lid
            size = int(sub.sum())
            area = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            vals = big_s[sl][sub]
            snr_sum = float(vals.sum())
            keep[lid] = (size >= g.get("min_rotor_pixels", 0)
                         and area >= g.get("min_box_area", 0)
                         and (size / float(area) if area else 0.0)
                         >= g.get("min_fill_ratio", 0.0)
                         and snr_sum >= g.get("min_snr_sum", 0.0)
                         and (snr_sum / size if size else 0.0)
                         >= g.get("min_snr_avg", 0.0))
        conf = keep[lab]
        return conf[::P, ::P][:self.ph, :self.pw] if P > 1 else conf

    def overlay(self, frame, show_raw=True):
        """Paint held detections onto a BGR frame, fading with age.

        Snapshots the maps under the lock, then paints outside it: the worker
        must not wait on the draw, and the draw must not see a half-published
        window.
        """
        with self.lock:
            live = self.age <= self.hold
            if not live.any():
                return 0, 0                  # (confirmed, unconfirmed)
            confirmed = self.confirmed.copy()
            snr = self.snr.copy()
            age = self.age.copy()
        strong = live & confirmed
        # colour by comb-SNR: dim red just past the gate, through orange, to
        # yellow well above it, so a strong rotor is obvious at a glance
        rel = np.clip(snr / max(self.cfg.tau_comb * 4.0, 1e-6), 0.0, 1.0)
        fade = 1.0 - (age.astype(np.float32) / (self.hold + 1.0))
        b = np.zeros_like(rel)
        g = rel * fade
        r = fade
        cell = np.dstack([b, g, r]) * 255.0
        big = np.repeat(np.repeat(cell, self.pool, 0), self.pool, 1)[:self.height, :self.width]
        if show_raw:
            # unconfirmed passes: dim blue, so isolated noise is visible as
            # noise instead of masquerading as a detection
            raw = np.repeat(np.repeat(live & ~confirmed, self.pool, 0),
                            self.pool, 1)[:self.height, :self.width]
            frame[raw] = (90, 40, 0)
        mask = np.repeat(np.repeat(strong, self.pool, 0),
                         self.pool, 1)[:self.height, :self.width]
        frame[mask] = big[mask].astype(np.uint8)
        return int(strong.sum()), int((live & ~confirmed).sum())

    def draw_boxes(self, frame):
        with self.lock:
            boxes, box_freq = list(self.boxes), list(self.box_freq)
        for i, box in enumerate(boxes):
            # boxes come back in pooled cells; scale them to sensor pixels
            x0, y0 = int(box.x1) * self.pool, int(box.y1) * self.pool
            x1 = (int(box.x2) + 1) * self.pool - 1
            y1 = (int(box.y2) + 1) * self.pool - 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
            f = box_freq[i] if i < len(box_freq) else 0.0
            label = ("%.0f Hz  snr %.0f" % (f, box.score) if f > 0
                     else "snr %.0f" % box.score)
            cv2.putText(frame, label, (x0, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
class EventAccumulator:
    """Event slices covering the last accumulation_us microseconds."""

    def __init__(self, width, height, accumulation_us):
        self.width = width
        self.height = height
        self.accumulation_us = accumulation_us
        self.slices = deque()

    def clear(self):
        self.slices.clear()

    def add(self, events, end_ts):
        if events is not None and len(events):
            self.slices.append((end_ts, events))
        while self.slices and end_ts - self.slices[0][0] > self.accumulation_us:
            self.slices.popleft()

    def render(self, palette):
        _, bg, on_color, off_color = palette
        frame = np.full((self.height, self.width, 3), bg, dtype=np.uint8)
        for _, e in self.slices:
            x = e["x"].astype(np.int32)
            y = e["y"].astype(np.int32)
            off = e["p"] == 0
            if off.any():
                frame[y[off], x[off]] = off_color
            if (~off).any():
                frame[y[~off], x[~off]] = on_color
        return frame


class EventReader:
    """Pulls event slices off the SDK iterator on its own thread.

    next() blocks until the camera has something to give. On a quiet scene -
    tight biases, nothing moving - that is a long time, and driving the
    iterator from the draw loop meant the window stopped repainting and stopped
    answering clicks until the next event happened to arrive. It looked like a
    hang, and it was worst exactly when the least was going on. Reading here
    means the display keeps its frame rate no matter how little the sensor is
    producing, and an empty screen is visibly empty instead of frozen.
    """

    def __init__(self, stream, max_slices=512):
        self.stream = stream
        self.max_slices = max_slices
        self.lock = threading.Lock()
        self.slices = deque()
        self.dropped = 0
        self.finished = False
        self.error = None
        self._closing = False
        self._thread = threading.Thread(target=self._serve, name="events",
                                        daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _serve(self):
        try:
            for e in self.stream:
                if self._closing:
                    break
                if not len(e):
                    continue
                # Copy: the SDK hands back a view over a buffer it reuses for
                # the next slice, and both the accumulator and the detector
                # keep slices well past the frame they arrived in.
                e = np.array(e, copy=True)
                with self.lock:
                    if len(self.slices) >= self.max_slices:
                        self.slices.popleft()
                        self.dropped += 1
                    self.slices.append(e)
        except Exception as exc:               # StopIteration ends the for loop
            self.error = exc
        self.finished = True

    def take(self):
        """Every slice that arrived since the last call. Never blocks."""
        with self.lock:
            out, self.slices = list(self.slices), deque()
        return out

    def close(self):
        self._closing = True


def find_stuck_pixels(counts, min_count, isolation=0.10):
    """Busy pixels whose eight neighbours are quiet - defects, not rotors."""
    busy = counts > min_count
    if not busy.any():
        return np.zeros(counts.shape, bool)
    neigh = np.zeros_like(counts)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                neigh += np.roll(np.roll(counts, dy, 0), dx, 1)
    return busy & (neigh < isolation * counts)


def draw_text(frame, lines):
    y = 16
    for text, color in lines:
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)
        y += 17
    return frame


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
K_LEFT, K_UP, K_RIGHT, K_DOWN = 2424832, 2490368, 2555904, 2621440
K_PGUP, K_PGDN = 2162688, 2228224


class Viewer:
    def __init__(self, args):
        self.args = args
        preset = dict(ddhf_config._PRESET)
        gates = {}
        if args.working_point:
            wp_preset, gates, wp = load_working_point(args.working_point)
            preset.update(wp_preset)
            print("[wp] %s (mode %s): %s" % (
                os.path.basename(args.working_point), wp.get("mode"),
                ", ".join("%s=%g" % kv for kv in sorted(gates.items()))))
        preset.update({k: v for k, v in
                       ddhf_config.cfg_overrides_from_args(args).items()
                       if v is not None})
        if getattr(args, "min_events", None) is None and not args.working_point:
            # The preset's 16 suits the dense cached recordings; a live rotor
            # this sparse leaves nothing to transform. A working point brings
            # its own value, so do not override that.
            preset["min_events"] = 8
        self.cfg = DDHFConfig(**preset)
        self.window_us = int(self.cfg.window_ms * 1000.0)

        self.offline = bool(args.input)
        self.device = studio.open_device(args.input, args.serial, False)
        self.width, self.height = studio.describe_device(self.device)

        # A single stuck pixel can own the stream - (20,420) was 54% of
        # rec_20260916_104120.raw. That matters beyond the event count: the
        # detector subsamples each window to --max-window-events uniformly, so
        # a defect at half the stream takes half of every window the NDFT ever
        # sees. Masking it roughly doubles the rotor's share for free.
        self.hot_mask = None
        hot_path = args.hot_pixels
        if hot_path is None:
            hot_path = next((p for p in ("cam.hotpx", os.path.join(
                _LAB, "data", "hot_pixels", "cam.hotpx")) if os.path.exists(p)), "")
        if hot_path:
            self.hot_mask = hot_pixels.Mask.load(hot_path, self.width, self.height)
            print("[hot] masking %d pixel(s) from %s"
                  % (len(self.hot_mask.pixels), os.path.basename(hot_path)))

        self.biases = studio.BiasControl(self.device, args.bypass_range_check)
        if args.nominal:
            self.biases.apply(NOMINAL, "nominal defaults")
        if args.bias and os.path.exists(args.bias):
            self.biases.apply(studio.read_bias_file(args.bias),
                              os.path.basename(args.bias))
        # After the file, so a flag wins over it for the bias it names.
        cli_biases = {n: getattr(args, n) for n in BIAS_NAMES
                      if getattr(args, n, None) is not None}
        if cli_biases and not self.biases.live:
            print("[bias] ignoring %s: no bias facility (replaying a file)"
                  % ", ".join(sorted(cli_biases)))
        elif cli_biases:
            self.biases.apply(cli_biases, "command line")
        for name in self.biases.names:
            print("[bias] %-14s %4d%s" % (
                name, self.biases.values[name],
                "   (outside recommended range)"
                if self.biases.out_of_recommended(name) else ""))
        warning = self.biases.consistency_warning()
        if warning:
            print("[bias] %s" % warning)

        self.det = Detector(self.width, self.height, self.cfg, args.pool,
                            args.hold, args.max_window_events, gates,
                            threaded=not args.sync_detector)
        self.accum = EventAccumulator(self.width, self.height,
                                      int(args.accumulation_time * 1000))
        self.palette_index = 0
        self.selected = 0
        self.paused = False
        self.show_det = True
        self.show_boxes = not args.no_boxes
        self.show_raw = not args.hide_unconfirmed
        self.pending = {}
        self._suspend = False
        self._drag = None                     # bias being dragged in the panel
        self._bias_panel_state = None         # last rendered panel's contents
        self._spec_open = False               # spectrum window opened on demand
        self.rate_hz = 0.0
        self.fps = 0.0
        self.message = ""
        self.message_until = 0.0
        self.was_detecting = False
        self.last_announce = 0.0
        self.running = True
        self.record_dir = args.record_dir or os.getcwd()
        self.recording = None
        # per-pixel counts over the first second, to find stuck pixels
        self.counts = np.zeros((self.height, self.width), np.int32)
        self.calibrating = True
        self.calib_deadline = time.time() + 6.0
        self.calib_events = 0
        self.reader = None
        self.buf = []
        self.buf_start = None
        studio.set_timer_resolution(True)
        self._build_windows()
        if args.record:
            self.toggle_recording()

    # -- windows ------------------------------------------------------------
    def _build_windows(self):
        cv2.namedWindow(VIEW_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(VIEW_WIN, self._on_view_mouse)
        cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
        self._bar("min_events", int(self.cfg.min_events), 64, self._on_min_events, 1)
        self._bar("tau_comb", int(self.cfg.tau_comb), 60, self._on_tau_comb, 1)
        self._bar("tau_sf_x100", int(self.cfg.tau_sf * 100), 100, self._on_tau_sf, 1)
        self._bar("pool", self.det.pool, 12, self._on_pool, 1)
        self._bar("hold", self.det.hold, 30, self._on_hold, 1)
        self._bar("accum_ms", int(self.accum.accumulation_us / 1000), 100,
                  self._on_accum, 1)
        if not self.biases.names:
            return
        # The biases get their own window: mixed in with the detector gates
        # they were seven more anonymous sliders in a stack of thirteen, and a
        # trackbar shows a number but not the recommended range, the hint, or
        # whether the sensor actually took the value.
        cv2.namedWindow(BIAS_WIN, cv2.WINDOW_AUTOSIZE)
        for name in self.biases.names:
            lo, hi = self.biases.limits[name]
            cv2.createTrackbar(name, BIAS_WIN, self.biases.values[name], hi,
                               self._make_bias_cb(name))
            try:
                cv2.setTrackbarMin(name, BIAS_WIN, lo)
            except cv2.error:
                pass
        cv2.setMouseCallback(BIAS_WIN, self._on_bias_mouse)

    def _bar(self, name, value, hi, cb, lo=0):
        cv2.createTrackbar(name, CTRL_WIN, int(value), hi, cb)
        try:
            cv2.setTrackbarMin(name, CTRL_WIN, lo)
        except cv2.error:
            pass

    def _make_bias_cb(self, name):
        def cb(value):
            if not self._suspend:
                self.pending[name] = value
        return cb

    def _on_min_events(self, v):
        self.cfg.min_events = max(1, int(v))

    def _on_tau_comb(self, v):
        self.cfg.tau_comb = float(max(1, v))

    def _on_tau_sf(self, v):
        self.cfg.tau_sf = max(0.01, v / 100.0)

    def _on_pool(self, v):
        if not self._suspend:
            self.det.set_pool(max(1, int(v)))
            self._recalibrate()

    def _on_hold(self, v):
        self.det.hold = max(1, int(v))

    def _on_accum(self, v):
        self.accum.accumulation_us = max(1, int(v)) * 1000

    def _recalibrate(self):
        self.counts[:] = 0
        self.calib_events = 0
        self.calibrating = True
        self.calib_deadline = time.time() + 6.0

    def _finish_calibration(self):
        """Mark the isolated always-on pixels and get out of the way.

        Also called on a timeout, because the 400k-event target is never
        reached on a quiet scene - and until this finishes the header shows a
        calibration percentage instead of the detection line, so a viewer that
        was working fine looked stuck at "finding stuck pixels ... 0%".
        """
        self.calibrating = False
        if self.calib_events < 20000:
            return                            # too little to tell a defect from noise
        stuck = find_stuck_pixels(self.counts, max(200, self.calib_events // 2000))
        ys, xs = np.nonzero(stuck)
        for x, y in zip(xs, ys):
            self.det.mark_stuck(int(x), int(y))
        if len(xs):
            self.notify("ignoring %d stuck pixel(s): %s" % (
                len(xs), ", ".join("(%d,%d)" % (a, b)
                                   for a, b in zip(xs[:3], ys[:3]))), 6.0)

    def notify(self, text, seconds=3.0):
        print("[ui] %s" % text)
        self.message = text
        self.message_until = time.time() + seconds

    def toggle_recording(self):
        """Start/stop a RAW recording, with the biases beside it.

        The .bias sidecar is what makes the recording replayable in context: a
        RAW file carries the events but not the sensor configuration that
        produced them, and the same scene looks completely different at
        different biases. Written at the moment recording starts, so it records
        the settings actually in force.
        """
        stream = self.device.get_i_events_stream()
        if stream is None:
            self.notify("no events stream to record")
            return
        if self.recording:
            stream.stop_log_raw_data()
            size = (os.path.getsize(self.recording) / 1e6
                    if os.path.exists(self.recording) else 0.0)
            self.notify("recorded %s (%.1f MB)"
                        % (os.path.basename(self.recording), size), 6.0)
            self.recording = None
            return
        if self.offline:
            self.notify("cannot record while replaying a file")
            return
        os.makedirs(self.record_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.record_dir, "rec_%s.raw" % stamp)
        stream.log_raw_data(path)
        self.recording = path
        if self.biases.live:
            studio.write_bias_file(os.path.splitext(path)[0] + ".bias",
                                   self.biases.values)
        self.notify("recording to %s (+ .bias)" % os.path.basename(path), 5.0)

    # -- main loop ----------------------------------------------------------
    def run(self):
        it = EventsIterator.from_device(device=self.device, mode="delta_t",
                                        delta_t=int(self.args.refresh_time * 1000))
        self.reader = EventReader(it).start()
        last_ts = 0
        rate_window = deque()
        frames = deque(maxlen=30)

        while self.running:
            if not self.paused:
                last_ts = self._pump(rate_window, last_ts)
            self._draw(last_ts)
            frames.append(time.time())
            if len(frames) > 1:
                span = frames[-1] - frames[0]
                self.fps = (len(frames) - 1) / span if span > 0 else 0.0
            self._keys(cv2.waitKeyEx(1))
            self._flush_pending()
            if cv2.getWindowProperty(VIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
        self.shutdown()

    def _pump(self, rate_window, last_ts):
        """Take every slice the reader thread has collected since last frame.

        Takes the whole backlog rather than one slice per frame: one per frame
        never works off a backlog, which showed up in the other viewer as a
        constant few-hundred-ms delay.
        """
        for e in self.reader.take():
            if self.hot_mask is not None:
                e = self.hot_mask(e)
                if not len(e):
                    continue
            last_ts = int(e["t"][-1])
            self.accum.add(e, last_ts)
            self._collect(e)
            rate_window.append((time.time(), len(e)))
        # Outside the loop so the rate decays to zero on a quiet scene instead
        # of freezing at whatever it last was - a stale number reads as a hung
        # viewer, which is the thing this is meant to tell you apart from.
        now = time.time()
        while rate_window and now - rate_window[0][0] > 1.0:
            rate_window.popleft()
        span = max(1e-3, now - rate_window[0][0]) if rate_window else 1.0
        self.rate_hz = sum(n for _, n in rate_window) / span
        if self.reader.finished and not self.reader.slices:
            if self.reader.error is not None:
                self.notify("event stream stopped: %s" % self.reader.error, 30.0)
            else:
                self.notify("end of %s" % (os.path.basename(self.args.input)
                                           if self.offline else "stream"), 30.0)
            self.paused = True
        return last_ts

    def _collect(self, e):
        """Buffer events and hand the detector each completed DDHF window."""
        if self.calibrating:
            # bincount, not np.add.at: the latter is orders of magnitude slower
            # and this runs on every slice, including at flood rates
            flat = (e["y"].astype(np.int64) * self.width
                    + e["x"].astype(np.int64))
            self.counts += np.bincount(
                flat, minlength=self.counts.size).reshape(self.counts.shape).astype(np.int32)
            self.calib_events += len(e)
            if self.calib_events > 400000:
                self._finish_calibration()
        self.buf.append(e)
        first = int(e["t"][0])
        if self.buf_start is None:
            self.buf_start = first
        if int(e["t"][-1]) - self.buf_start >= self.window_us:
            win = np.concatenate(self.buf)
            self.buf = []
            self.buf_start = None
            if self.show_det:
                # hands the window to the detector thread and returns; the
                # results appear in the maps a few frames later
                self.det.submit(win["x"].astype(np.int64),
                                win["y"].astype(np.int64),
                                win["p"], win["t"].astype(np.int64))

    def _announce(self):
        """Print the frequency when a detection appears, then once a second.

        Printed rather than only drawn because the interesting number is the
        one you want to keep: the overlay is gone in `hold` windows, and a
        rotor that is only intermittently confirmed would otherwise leave no
        record of what it measured.

        Called once per drawn frame, not per window: the detector publishes on
        its own thread, so the draw loop is the only place that sees every
        result exactly once it is complete.
        """
        # "still detecting" spans the hold, not just this window: the detector
        # drops windows when it falls behind, and a rotor is only confirmed in
        # a fraction of the windows anyway. Keying off the newest window alone
        # would call every gap a new detection and print on each one.
        with self.det.lock:
            fresh = self.det.freq_age == 0
            detecting = self.det.freq_age <= self.det.hold
        now = time.time()
        if fresh and (not self.was_detecting or now - self.last_announce > 1.0):
            rpm = ("  ~%.0f rpm" % (60.0 * self.det.last_freq / self.args.blades)
                   if self.args.blades > 0 else "")
            print("[det] %.0f Hz blade-pass (+-%.0f Hz bin)%s  peak snr %.1f  "
                  "%d cells  %d box(es)"
                  % (self.det.last_freq, self.det.bin_hz, rpm,
                     self.det.last_peak, self.det.last_count,
                     len(self.det.boxes)))
            if len(self.det.box_freq) > 1:
                print("      per box: %s" % ", ".join(
                    "%.0f Hz" % f for f in self.det.box_freq))
            self.last_announce = now
        self.was_detecting = detecting

    def _draw(self, last_ts):
        if self.calibrating and time.time() > self.calib_deadline:
            self._finish_calibration()        # here too: _collect never runs
        if self.show_det:                     #   when no events are arriving
            self._announce()
        palette = studio.PALETTES[self.palette_index]
        frame = self.accum.render(palette)
        shown, raw = (self.det.overlay(frame, self.show_raw)
                      if self.show_det else (0, 0))
        if self.show_det and self.show_boxes:
            self.det.draw_boxes(frame)
        self._draw_probe_marker(frame)        # before the scale, so it is 1 cell
        if self.args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.args.scale, fy=self.args.scale,
                               interpolation=cv2.INTER_NEAREST)
        rate = ("%.2f Mev/s" % (self.rate_hz / 1e6) if self.rate_hz >= 1e6
                else "%.0f kev/s" % (self.rate_hz / 1e3))
        lines = [
            ("%dx%d  %s  %.0f fps  accum %d ms%s" % (
                self.width, self.height, rate, self.fps,
                self.accum.accumulation_us // 1000,
                "  [PAUSED]" if self.paused else ""), (0, 255, 0)),
            ("detector: pool %d  min_ev %d  tau_comb %.0f  tau_sf %.2f  "
             "%.1f ms/window" % (self.det.pool, self.cfg.min_events,
                                 self.cfg.tau_comb, self.cfg.tau_sf,
                                 self.det.last_ms), (0, 255, 0)),
        ]
        analysed = self.det.windows - self.det.dropped
        if self.det.dropped and self.det.windows:
            # The detector is slower than real time and skipping windows to
            # stay out of the display's way. Amber past a third dropped: below
            # that the overlay still updates several times a second.
            miss = self.det.dropped / float(self.det.windows)
            lines.append(("analysing %d of %d windows (%.0f%% dropped) - lower "
                          "--max-window-events or raise --pool for more"
                          % (analysed, self.det.windows, 100.0 * miss),
                          (0, 140, 255) if miss > 0.33 else (140, 190, 140)))
        if self.det.overloaded:
            lines.append(("dense window: %d events subsampled to %d"
                          % (self.det.overloaded, self.det.max_window_events),
                          (0, 140, 255)))
        elif self.calibrating:
            lines.append(("finding stuck pixels ... %d%%"
                          % min(99, self.calib_events // 4000), (0, 220, 255)))
        elif self.show_det:
            lines.append(("confirmed %d cells, %d boxes, peak comb-SNR %.1f  "
                          "| %d unconfirmed (noise)"
                          % (shown, len(self.det.boxes), self.det.last_peak, raw),
                          (0, 255, 255) if shown else (160, 160, 160)))
            readout, fresh = self.det.freq_readout(self.args.blades)
            if readout:
                # dim once the detection stops refreshing, so a stale number is
                # never mistaken for something the camera is still seeing
                lines.append(("ROTOR  " + readout + ("" if fresh else "  (held)"),
                              (0, 255, 255) if fresh else (0, 150, 150)))
        else:
            lines.append(("detections off (x to re-enable)", (160, 160, 160)))
        if self.recording:
            lines.append(("REC -> %s" % os.path.basename(self.recording),
                          (0, 0, 255)))
        if time.time() < self.message_until:
            lines.append((self.message, (0, 220, 255)))
        cv2.imshow(VIEW_WIN, draw_text(frame, lines))

        panel = np.full((90, 430, 3), 32, np.uint8)
        cv2.putText(panel, "detector gates - biases are in the Biases window",
                    (10, 24), FONT, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        for i, txt in enumerate(("a z accum   p palette   x detections   b boxes",
                                 "u unconfirmed   c clear   SPACE freeze   q quit")):
            cv2.putText(panel, txt, (10, 50 + i * 18), FONT, 0.38,
                        (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(CTRL_WIN, panel)
        self._show_bias_panel()
        self._show_spectrum_panel()

    # -- pixel spectrum -----------------------------------------------------
    def _on_view_mouse(self, event, x, y, flags, _param):
        """Left-click a pixel to probe it, right-click anywhere to stop."""
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = self.args.scale or 1.0
            sx, sy = int(x / scale), int(y / scale)
            if 0 <= sx < self.width and 0 <= sy < self.height:
                self.det.probe = (sx, sy)
                if not self._spec_open:
                    cv2.namedWindow(SPEC_WIN, cv2.WINDOW_AUTOSIZE)
                    self._spec_open = True
                self.notify("probing pixel (%d,%d)" % (sx, sy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.det.probe = None
            with self.det.lock:
                self.det.probe_result = None
            self.notify("probe cleared")

    def _draw_probe_marker(self, frame):
        """Ring the pooled cell being probed, on the sensor-resolution frame."""
        probe = self.det.probe
        if probe is None:
            return
        pool = self.det.pool
        x0, y0 = (probe[0] // pool) * pool, (probe[1] // pool) * pool
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + pool, y0 + pool),
                      (255, 255, 255), 1)

    def _show_spectrum_panel(self):
        if not self._spec_open:
            return
        cv2.imshow(SPEC_WIN, self._draw_spectrum_panel())

    def _draw_spectrum_panel(self):
        """Power spectrum of the probed cell, with every gate drawn on it."""
        panel = np.full((SP_H, SP_W, 3), 22, np.uint8)
        with self.det.lock:
            r = dict(self.det.probe_result) if self.det.probe_result else None
        if r is None:
            cv2.putText(panel, "left-click a pixel in the view to probe it",
                        (SP_L, SP_H // 2), FONT, 0.5, (150, 150, 150), 1,
                        cv2.LINE_AA)
            return panel
        cfg = self.cfg
        x0, x1 = SP_L, SP_W - SP_R
        y0, y1 = SP_T, SP_H - SP_B
        head = ("pixel (%d,%d) -> cell (%d,%d) at pool %d      %d events "
                "(min_events %d)" % (r["pixel"] + r["cell"] + (r["pool"],
                                     r["n"], cfg.min_events)))
        cv2.putText(panel, head, (SP_L, 20), FONT, 0.42, (210, 210, 210), 1,
                    cv2.LINE_AA)
        if r["power"] is None:
            cv2.putText(panel, r["verdict"], (SP_L, 44), FONT, 0.44,
                        (150, 150, 150), 1, cv2.LINE_AA)
            return panel
        gates = ("flatness %.3f / tau_sf %.2f      comb SNR %.1f / tau_comb "
                 "%.1f      peak %.0f Hz" % (r["sf"], cfg.tau_sf, r["snr"],
                                             cfg.tau_comb, r["freq"]))
        cv2.putText(panel, gates, (SP_L, 40), FONT, 0.42, (210, 210, 210), 1,
                    cv2.LINE_AA)
        cv2.putText(panel, r["verdict"], (SP_L, 60), FONT, 0.44,
                    (120, 240, 120) if r["passed"] else (0, 170, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(panel, "shaded = rotor_bands_hz; only fundamentals in "
                    "there are ever tested", (SP_L, 78), FONT, 0.36,
                    (130, 130, 130), 1, cv2.LINE_AA)

        power = np.asarray(r["power"], dtype=np.float64)
        K = power.size
        hz_max = (K - 1) * self.det.bin_hz

        def px_of_hz(hz):
            return int(x0 + (hz / max(hz_max, 1e-9)) * (x1 - x0))

        # The searchable bands first, so the curve draws over them.
        for lo, hi in getattr(cfg, "rotor_bands_hz", ()):
            cv2.rectangle(panel, (px_of_hz(lo), y0), (px_of_hz(hi), y1),
                          (56, 44, 30), -1)
        # The winning comb: fundamental and its harmonics, each with the
        # +-harmonic_half_width window the peak was actually taken over.
        if r["omega"]:
            for m in range(1, cfg.num_harmonics + 1):
                k = m * r["omega"]
                if k >= K:
                    break
                lo = max(0, k - cfg.harmonic_half_width) * self.det.bin_hz
                hi = min(K - 1, k + cfg.harmonic_half_width) * self.det.bin_hz
                cv2.rectangle(panel, (px_of_hz(lo), y0), (px_of_hz(hi), y1),
                              (40, 70, 80), -1)
                cv2.line(panel, (px_of_hz(k * self.det.bin_hz), y0),
                         (px_of_hz(k * self.det.bin_hz), y1), (0, 200, 255), 1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)

        # log power: a comb sits orders of magnitude above its floor, and on a
        # linear axis everything but the tallest harmonic vanishes
        mag = np.log10(power + 1e-12)
        lo_m, hi_m = float(mag[1:].min()), float(mag.max())
        span = max(hi_m - lo_m, 1e-9)
        pts = np.empty((K - 1, 2), np.int32)
        for k in range(1, K):                      # skip DC: not part of the comb
            pts[k - 1] = (px_of_hz(k * self.det.bin_hz),
                          int(y1 - (mag[k] - lo_m) / span * (y1 - y0)))
        cv2.polylines(panel, [pts], False, (210, 210, 210), 1, cv2.LINE_AA)

        # the noise floor the comb SNR divides by
        med = float(np.median(power[1:]))
        ymed = int(y1 - (np.log10(med + 1e-12) - lo_m) / span * (y1 - y0))
        if y0 <= ymed <= y1:
            cv2.line(panel, (x0, ymed), (x1, ymed), (90, 90, 140), 1)
            cv2.putText(panel, "median", (x1 - 52, max(y0 + 10, ymed - 4)),
                        FONT, 0.32, (120, 120, 170), 1, cv2.LINE_AA)

        step = 500.0 if hz_max > 2000 else 200.0
        hz = 0.0
        while hz <= hz_max:
            xt = px_of_hz(hz)
            cv2.line(panel, (xt, y1), (xt, y1 + 4), (110, 110, 110), 1)
            cv2.putText(panel, "%d" % hz, (xt - 12, y1 + 18), FONT, 0.34,
                        (140, 140, 140), 1, cv2.LINE_AA)
            hz += step
        cv2.putText(panel, "Hz", (x1 - 18, y1 + 18), FONT, 0.34,
                    (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, "log P", (6, y0 + 10), FONT, 0.34,
                    (140, 140, 140), 1, cv2.LINE_AA)
        return panel

    # -- bias panel ---------------------------------------------------------
    def _bias_rows_bottom(self):
        return BP_TOP + len(self.biases.names) * BP_ROW_H

    def _bias_buttons(self):
        """(label, action, x0, y0, x1, y1) for the footer buttons."""
        y0 = self._bias_rows_bottom() + 30
        out = []
        for i, (label, action) in enumerate(BP_BUTTONS):
            x0 = BP_X_NAME + i * (BP_BTN_W + 8)
            if action == "bypass":
                label = "range: %s" % ("full" if self.biases.bypass else "rec")
            out.append((label, action, x0, y0, x0 + BP_BTN_W, y0 + BP_BTN_H))
        return out

    def _bias_row_at(self, y):
        row = (y - BP_TOP) // BP_ROW_H
        return int(row) if y >= BP_TOP and 0 <= row < len(self.biases.names) else None

    def _show_bias_panel(self):
        """Redraw the bias window only when something on it changed.

        Nothing on this panel moves on its own, and HighGUI repaints a window
        from the last image it was given, so re-rendering and pushing it every
        frame was pure overhead in the display loop.
        """
        if not self.biases.names:
            return
        state = (tuple(self.biases.values[n] for n in self.biases.names),
                 tuple(self.biases.limits[n] for n in self.biases.names),
                 self.selected, self.biases.bypass)
        if state == self._bias_panel_state:
            return
        self._bias_panel_state = state
        cv2.imshow(BIAS_WIN, self._draw_bias_panel())

    def _draw_bias_panel(self):
        names = self.biases.names
        height = self._bias_rows_bottom() + 30 + BP_BTN_H + 10
        panel = np.full((height, BP_W, 3), 28, np.uint8)
        cv2.putText(panel, "drag a bar, click -/+, right-click a row for its "
                    "nominal value", (BP_X_NAME, 19), FONT, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
        for i, name in enumerate(names):
            y = BP_TOP + i * BP_ROW_H
            sel = (i == self.selected)
            value = self.biases.values[name]
            lo, hi = self.biases.limits[name]
            if sel:
                cv2.rectangle(panel, (4, y + 1), (BP_W - 5, y + BP_ROW_H - 3),
                              (58, 52, 30), -1)
            if not self.biases.modifiable.get(name, False):
                colour = (105, 105, 105)      # the HAL refuses to move this one
            elif sel:
                colour = (0, 220, 255)
            else:
                colour = (205, 205, 205)
            cv2.putText(panel, name, (BP_X_NAME, y + 21), FONT, 0.44, colour,
                        1, cv2.LINE_AA)
            # amber value = outside the recommended range: applied, but the
            # sensor is not characterised there
            cv2.putText(panel, "%4d" % value, (BP_X_VALUE, y + 21), FONT, 0.46,
                        (0, 180, 255) if self.biases.out_of_recommended(name)
                        else colour, 1, cv2.LINE_AA)
            self._draw_button(panel, "-", BP_X_MINUS[0], y + 6,
                              BP_X_MINUS[1], y + BP_ROW_H - 8)
            self._draw_button(panel, "+", BP_X_PLUS[0], y + 6,
                              BP_X_PLUS[1], y + BP_ROW_H - 8)
            x0, x1 = BP_X_TRACK
            cv2.rectangle(panel, (x0, y + 8), (x1, y + BP_ROW_H - 10),
                          (72, 72, 72), 1)
            frac = 0.0 if hi <= lo else (value - lo) / float(hi - lo)
            fill = int(round(x0 + min(1.0, max(0.0, frac)) * (x1 - x0)))
            cv2.rectangle(panel, (x0 + 1, y + 9),
                          (max(x0 + 1, fill - 1), y + BP_ROW_H - 11),
                          (0, 190, 255) if sel else (0, 120, 175), -1)
            cv2.putText(panel, "%d-%d" % (lo, hi), (BP_X_RANGE, y + 21), FONT,
                        0.33, (125, 125, 125), 1, cv2.LINE_AA)
        if names:
            cv2.putText(panel, studio.HINTS.get(names[self.selected], ""),
                        (BP_X_NAME, self._bias_rows_bottom() + 19), FONT, 0.38,
                        (150, 195, 150), 1, cv2.LINE_AA)
        for label, _action, x0, y0, x1, y1 in self._bias_buttons():
            self._draw_button(panel, label, x0, y0, x1, y1, 0.36)
        return panel

    @staticmethod
    def _draw_button(panel, label, x0, y0, x1, y1, scale=0.5):
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        cv2.putText(panel, label, (x0 + (x1 - x0 - tw) // 2,
                                   y0 + (y1 - y0 + th) // 2), FONT, scale,
                    (190, 190, 190), 1, cv2.LINE_AA)

    def _on_bias_mouse(self, event, x, y, flags, _param):
        """Click/drag on the bias panel.

        Runs on the GUI thread inside waitKeyEx, the same place the trackbar
        callbacks run, so it can talk to the HAL directly instead of queueing
        into self.pending like the trackbars have to.
        """
        if not self.biases.names:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for _label, action, x0, y0, x1, y1 in self._bias_buttons():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._bias_action(action)
                    return
            row = self._bias_row_at(y)
            if row is None:
                return
            self.selected = row
            if BP_X_MINUS[0] <= x <= BP_X_MINUS[1]:
                self._bump(-1)
            elif BP_X_PLUS[0] <= x <= BP_X_PLUS[1]:
                self._bump(+1)
            elif BP_X_TRACK[0] <= x <= BP_X_TRACK[1]:
                self._drag = self.biases.names[row]
                self._set_from_track(self._drag, x)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag and (flags & cv2.EVENT_FLAG_LBUTTON):
                self._set_from_track(self._drag, x)
            else:
                self._drag = None
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            row = self._bias_row_at(y)
            if row is not None:
                self.selected = row
                name = self.biases.names[row]
                if name in NOMINAL:
                    self.biases.set(name, NOMINAL[name])
                    self._sync_bias_bar(name)
                    self.notify("%s -> nominal %d"
                                % (name, self.biases.values[name]))

    def _set_from_track(self, name, x):
        """Queue the dragged value; _flush_pending writes it once per frame.

        Setting it here would put a HAL write and its readback on every mouse
        move event, and those arrive far faster than frames - a quick drag
        across the bar became dozens of USB round trips and visibly stuttered.
        """
        lo, hi = self.biases.limits[name]
        x0, x1 = BP_X_TRACK
        frac = min(1.0, max(0.0, (x - x0) / float(x1 - x0)))
        self.pending[name] = int(round(lo + frac * (hi - lo)))

    def _bias_action(self, action):
        if action == "nominal":
            self.biases.apply(NOMINAL, "nominal defaults")
            self._sync_all()
        elif action == "startup":
            self.biases.apply(self.biases.startup, "startup values")
            self._sync_all()
        elif action == "load":
            self._load_bias_file()
        elif action == "save":
            self._save_biases()
        elif action == "bypass":
            self._toggle_bypass()

    def _toggle_bypass(self):
        """Swap the settable range between recommended and the full allowed one.

        Without the bypass the HAL refuses anything outside the recommended
        range (error 102001), so the panel must not offer it; with it the whole
        allowed range becomes reachable. Values already set are left alone -
        narrowing the range does not silently move the sensor, the panel just
        marks them amber.
        """
        self.biases.bypass = not self.biases.bypass
        for name in self.biases.names:
            self.biases.limits[name] = (self.biases.ranges[name]
                                        if self.biases.bypass
                                        else self.biases.recommended[name])
            lo, hi = self.biases.limits[name]
            try:                              # min first: it stays below max
                cv2.setTrackbarMin(name, BIAS_WIN, lo)
                cv2.setTrackbarMax(name, BIAS_WIN, hi)
            except (cv2.error, AttributeError):
                pass
        self.notify("bias range: %s" % ("full allowed range" if self.biases.bypass
                                        else "recommended"))

    def _load_bias_file(self):
        if self.args.bias and os.path.exists(self.args.bias):
            self.biases.apply(studio.read_bias_file(self.args.bias),
                              os.path.basename(self.args.bias))
            self._sync_all()
            self.notify("loaded %s" % os.path.basename(self.args.bias))
        else:
            self.notify("no --bias file to load")

    def _save_biases(self):
        if not self.biases.live:
            self.notify("no bias facility to save (replaying a file)")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.record_dir, "biases_%s.bias" % stamp)
        studio.write_bias_file(path, self.biases.values)
        self.notify("saved %s" % path)

    # -- input --------------------------------------------------------------
    def _flush_pending(self):
        if not self.pending:
            return
        pending, self.pending = self.pending, {}
        for name, value in pending.items():
            self.biases.set(name, value)      # readback differs: the DAC quantises
            self._sync_bias_bar(name)         # follow the drag with the trackbar

    def _sync_bias_bar(self, name):
        self._suspend = True
        try:
            cv2.setTrackbarPos(name, BIAS_WIN, self.biases.values[name])
        except cv2.error:
            pass                              # no bias window (replaying a file)
        self._suspend = False

    def _bump(self, delta):
        if not self.biases.names:
            return
        name = self.biases.names[self.selected]
        self.biases.set(name, self.biases.values[name] + delta)
        self._sync_bias_bar(name)

    def _keys(self, key):
        if key == -1:
            return
        code = key & 0xFF
        if key in (K_UP, K_DOWN) and self.biases.names:
            self.selected = (self.selected + (-1 if key == K_UP else 1)) % len(
                self.biases.names)
        elif key == K_LEFT:
            self._bump(-1)
        elif key == K_RIGHT:
            self._bump(+1)
        elif key == K_PGUP:
            self._bump(+10)
        elif key == K_PGDN:
            self._bump(-10)
        elif code == ord(","):
            self._bump(-50)
        elif code == ord("."):
            self._bump(+50)
        elif code in (ord("q"), 27):
            self.running = False
        elif code == ord(" "):
            self.paused = not self.paused
        elif code == ord("c"):
            self.accum.clear()
            self.det.reset()
        elif code == ord("p"):
            self.palette_index = (self.palette_index + 1) % len(studio.PALETTES)
        elif code == ord("x"):
            self.show_det = not self.show_det
            self.notify("detections %s" % ("on" if self.show_det else "off"))
        elif code == ord("r"):
            self.toggle_recording()
        elif code == ord("b"):
            self.show_boxes = not self.show_boxes
            self.notify("boxes %s" % ("on" if self.show_boxes else "off"))
        elif code == ord("u"):
            self.show_raw = not self.show_raw
            self.notify("unconfirmed cells %s" % ("shown" if self.show_raw else "hidden"))
        elif code == ord("a"):
            self._set_accum(self.accum.accumulation_us // 1000 - 2)
        elif code == ord("z"):
            self._set_accum(self.accum.accumulation_us // 1000 + 2)
        elif code == ord("d"):
            self.biases.apply(self.biases.startup, "startup values")
            self._sync_all()
        elif code == ord("n"):
            self.biases.apply(NOMINAL, "nominal defaults")
            self._sync_all()
        elif code == ord("l"):
            self._load_bias_file()
        elif code == ord("s"):
            self._save_biases()
        elif code == ord("g"):
            self._toggle_bypass()
        elif code == ord("h"):
            print(__doc__)

    def _set_accum(self, ms):
        ms = int(max(1, min(100, ms)))
        self.accum.accumulation_us = ms * 1000
        self._suspend = True
        cv2.setTrackbarPos("accum_ms", CTRL_WIN, ms)
        self._suspend = False

    def _sync_all(self):
        for name in self.biases.names:
            self._sync_bias_bar(name)

    def shutdown(self):
        if self.reader is not None:
            self.reader.close()
        self.det.close()
        if self.recording:
            try:
                self.device.get_i_events_stream().stop_log_raw_data()
                print("[rec] stopped: %s" % self.recording)
            except Exception:
                pass
            self.recording = None
        cv2.destroyAllWindows()
        studio.set_timer_resolution(False)
        print("[bias] final values:")
        for name in self.biases.names:
            print("  %-4d %% %s" % (self.biases.values[name], name))


def main(argv=None):
    args = parse_args(argv)
    if args.input and not os.path.exists(args.input):
        # FileNotFoundError stringifies to the bare path, which reads like a
        # truncated message rather than an explanation.
        print("error: no such RAW file: %s" % args.input, file=sys.stderr)
        folder = os.path.dirname(os.path.abspath(args.input)) or "."
        near = sorted(f for f in os.listdir(folder) if f.endswith(".raw"))             if os.path.isdir(folder) else []
        if near:
            print("       RAW files in %s: %s"
                  % (folder, ", ".join(near[:6])), file=sys.stderr)
        return 1
    try:
        viewer = Viewer(args)
    except (OSError, FileNotFoundError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    try:
        viewer.run()
    except KeyboardInterrupt:
        viewer.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
