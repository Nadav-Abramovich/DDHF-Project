#!/usr/bin/env python
"""
Interactive live viewer + bias tuner for the SilkyEvCam VGA (EvC3A, Gen3.1).

A replacement for metavision_viewer.exe that adds live, interactive control of
the seven sensor biases while you watch the event stream:

    bias_diff       reference level the ON/OFF thresholds are compared against
    bias_diff_off   OFF-event contrast threshold  (lower  => more OFF events)
    bias_diff_on    ON-event  contrast threshold  (higher => fewer ON events)
    bias_fo         pixel low-pass  cut-off frequency - INVERTED: a lower
                    value is a HIGHER cut-off. Measured 2026-09-09: fo 1420
                    floods a quiet scene at 10 Mev/s, fo 1741 is nearly silent.
                    Use it to buy bandwidth, never to quieten a scene - that is
                    what the thresholds are for.
    bias_hpf        pixel high-pass cut-off frequency (kills slow/DC light)
    bias_pr         photoreceptor bandwidth
    bias_refr       refractory period after each event

Two windows open:
  * "SilkyEvCam - live"  the event image (ON white / OFF black on gray, default)
  * "Bias control"       one trackbar per bias + a status panel

Keys (either window must have focus):
    up/down .......... select a bias           left/right ... selected bias -1 / +1
    pgup/pgdn ........ selected bias +10/-10   , / . ....... selected bias -50 / +50
    a / z ............ accumulation time -/+   p ........... cycle color palette
    SPACE ............ freeze display          c ........... clear the accumulator
    f ................ apply the rotor preset  e ........... rotor readout on/off
    d ................ restore the biases the camera had at startup
    n ................ restore Prophesee nominal defaults
    l ................ reload the --bias file  s ........... save biases to a .bias file
    r ................ start/stop RAW recording (writes a .bias sidecar too)
    h ................ print this help         q / ESC ..... quit

Ranges: the HAL refuses any bias outside its recommended window (error 102001,
e.g. bias_diff must stay in 200-400), so by default the sliders span exactly
that window and cannot select a value the camera would reject. Pass
--bypass-range-check to unlock the full 0-1800 allowed range instead.

Note on this sensor: the Gen3.1 bias DAC quantises and saturates. Asking for
409 reads back as 404, and with the check bypassed bias_hpf tops out at ~1525
and bias_refr at ~1652 even though both advertise 0-1800. The panel always
shows the value the sensor actually holds, so trust the panel rather than the
trackbar position.

Usage:
    python silky_bias_studio.py                        # live camera
    python silky_bias_studio.py --bias data.bias       # live, biases preloaded
    python silky_bias_studio.py --input data.raw       # offline replay (read-only)
    python silky_bias_studio.py --record-dir runs      # where s / r write files
"""
import os
import sys
import time
import argparse
import datetime
from collections import deque

# ---------------------------------------------------------------------------
# Prophesee / CenturyArks runtime setup - must precede the metavision imports.
# The SilkyEvCam HAL plugin lives under CenturyArks, the generic one under
# Prophesee; both folders have to stay on MV_HAL_PLUGIN_PATH or the camera is
# simply not discovered.
# ---------------------------------------------------------------------------
_PLUGIN_DIRS = [
    r"C:\Program Files\Prophesee\lib\metavision\hal\plugins",
    r"C:\Program Files\CenturyArks\plugins",
]
_DLL_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "_deps", "openeb_bin"),                     # lab/_deps
    r"C:\Program Files\Prophesee\bin",
    r"C:\Program Files\Prophesee\third_party\bin",
    r"C:\Program Files\CenturyArks\bin",
    r"C:\Program Files\Prophesee\lib\hdf5\plugin",
] + _PLUGIN_DIRS

for _d in _DLL_DIRS:
    if os.path.exists(_d):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

def ensure_plugin_path():
    """Put both HAL plugin folders back on MV_HAL_PLUGIN_PATH.

    Not a one-off: importing other modules in this project (frame_explorer, via
    bayes_working_point) rewrites MV_HAL_PLUGIN_PATH to the Prophesee folder
    alone, which drops the CenturyArks plugin and makes the SilkyEvCam vanish
    from DeviceDiscovery entirely. open_device calls this first so the camera is
    findable regardless of what has been imported since.
    """
    existing = [d for d in os.environ.get("MV_HAL_PLUGIN_PATH", "").split(os.pathsep) if d]
    merged = list(dict.fromkeys([d for d in _PLUGIN_DIRS if os.path.exists(d)] + existing))
    os.environ["MV_HAL_PLUGIN_PATH"] = os.pathsep.join(merged)
    os.environ.setdefault("HDF5_PLUGIN_PATH", r"C:\Program Files\Prophesee\lib\hdf5\plugin")
    return os.environ["MV_HAL_PLUGIN_PATH"]


ensure_plugin_path()

import numpy as np
import cv2
import metavision_hal as mh
from metavision_core.event_io import EventsIterator

# ---------------------------------------------------------------------------
# Bias metadata
# ---------------------------------------------------------------------------
BIAS_NAMES = [
    "bias_diff",
    "bias_diff_off",
    "bias_diff_on",
    "bias_fo",
    "bias_hpf",
    "bias_pr",
    "bias_refr",
]

# Prophesee nominal values for the Gen3.1 VGA sensor (restored with the n key).
NOMINAL = {
    "bias_diff": 299,
    "bias_diff_off": 221,
    "bias_diff_on": 384,
    "bias_fo": 1477,
    "bias_hpf": 1499,
    "bias_pr": 1250,
    "bias_refr": 1500,
}

# Opt-in only (--preset, or the f key). This was briefly the startup default on
# the strength of a 306 Hz line it resolved at SNR 10.1 - but that line was not
# the rotor, it was present with the sensor seeing nothing but noise, and this
# configuration also produces 0.0% ON events. Do not make it a default again
# without a rotor that is actually visible to verify against.
ROTOR = {
    "bias_diff": 299,
    "bias_diff_off": 221,
    "bias_diff_on": 496,
    "bias_fo": 1477,
    "bias_hpf": 1448,
    "bias_pr": 1450,
    "bias_refr": 1506,
}

HINTS = {
    "bias_diff": "reference level for the ON/OFF thresholds",
    "bias_diff_off": "OFF threshold: lower = more OFF events",
    "bias_diff_on": "ON threshold: higher = fewer ON events",
    "bias_fo": "low-pass cut-off: LOWER = wider bandwidth, far more noise",
    "bias_hpf": "high-pass cut-off: higher = kills slow changes",
    "bias_pr": "photoreceptor bandwidth: lower = slower, quieter",
    "bias_refr": "refractory period: higher = shorter dead time",
}

PALETTES = [
    # (name, background, ON colour, OFF colour) in BGR
    ("gray", (128, 128, 128), (255, 255, 255), (0, 0, 0)),
    ("dark", (30, 30, 30), (255, 255, 255), (0, 0, 255)),
    ("rb", (0, 0, 0), (0, 0, 255), (255, 80, 0)),
]

VIEW_WIN = "SilkyEvCam - live"
CTRL_WIN = "Bias control"


def set_timer_resolution(enable):
    """Ask Windows for a 1 ms scheduler tick while the viewer is running.

    The default granularity is 15.6 ms, and cv2.waitKey(1) sleeps for a whole
    tick - which measured 15.5 ms per frame here, capping the display and
    adding that much latency on its own. Requesting 1 ms takes the GUI pump
    from 11.4 ms to 4.0 ms per frame. Always paired with the matching release.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        winmm = ctypes.windll.winmm
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Bias file I/O (Prophesee .bias format: "<value> % <name>")
# ---------------------------------------------------------------------------
def read_bias_file(path):
    """Parse a .bias file into {name: value}. Tolerates blank/comment lines."""
    values = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "%" not in line:
                continue
            raw, name = line.split("%", 1)
            try:
                values[name.strip()] = int(float(raw.strip()))
            except ValueError:
                continue
    return values


def write_bias_file(path, values):
    """Write {name: value} as a Prophesee .bias file."""
    with open(path, "w") as fh:
        for name in BIAS_NAMES:
            if name in values:
                fh.write("%-4d %% %s\n" % (values[name], name))
    return path


# ---------------------------------------------------------------------------
# Camera / bias handling
# ---------------------------------------------------------------------------
class BiasControl:
    """Thin wrapper over I_LL_Biases with ranges, defaults and file I/O."""

    def __init__(self, device, bypass_range_check=False):
        self.facility = device.get_i_ll_biases() if device is not None else None
        self.bypass = bypass_range_check
        self.names = []
        self.ranges = {}
        self.limits = {}
        self.recommended = {}
        self.modifiable = {}
        self.descriptions = {}
        self.values = {}
        self.startup = {}

        if self.facility is None:
            return

        available = self.facility.get_all_biases()
        self.names = [n for n in BIAS_NAMES if n in available]
        for name in self.names:
            info = self.facility.get_bias_info(name)
            lo, hi = info.get_bias_allowed_range()
            self.ranges[name] = (int(lo), int(hi))
            self.recommended[name] = tuple(int(v) for v in info.get_bias_recommended_range())
            self.modifiable[name] = bool(info.is_modifiable())
            self.descriptions[name] = info.get_description()
            self.values[name] = int(available[name])
            # Without the bypass the HAL refuses anything outside the
            # recommended range (error 102001), so that is what the UI must
            # offer; with the bypass the full allowed range is settable.
            self.limits[name] = self.ranges[name] if bypass_range_check                 else self.recommended[name]
        self.startup = dict(self.values)

    @property
    def live(self):
        return self.facility is not None

    def refresh(self):
        if self.facility is not None:
            for name in self.names:
                self.values[name] = int(self.facility.get(name))
        return self.values

    def set(self, name, value):
        """Set one bias; returns the value the sensor actually holds afterwards."""
        if self.facility is None or not self.modifiable.get(name, False):
            return self.values.get(name)
        lo, hi = self.limits[name]
        value = int(max(lo, min(hi, value)))
        try:
            self.facility.set(name, value)
        except Exception as exc:              # HAL raises when a value is refused
            print("[bias] %s <- %d rejected: %s" % (name, value, exc))
        self.values[name] = int(self.facility.get(name))
        return self.values[name]

    def apply(self, values, label=""):
        """Set several biases at once, printing what changed."""
        for name in self.names:
            if name in values and values[name] != self.values[name]:
                before = self.values[name]
                after = self.set(name, values[name])
                print("[bias] %-14s %4d -> %4d" % (name, before, after))
        if label:
            print("[bias] applied %s" % label)

    def out_of_recommended(self, name):
        lo, hi = self.recommended.get(name, (None, None))
        if lo is None:
            return False
        return not (lo <= self.values[name] <= hi)

    def consistency_warning(self):
        """Gen3.1 expects bias_diff_off < bias_diff < bias_diff_on."""
        v = self.values
        if not {"bias_diff", "bias_diff_on", "bias_diff_off"} <= set(v):
            return ""
        if not (v["bias_diff_off"] < v["bias_diff"] < v["bias_diff_on"]):
            return "check: expected diff_off < diff < diff_on"
        return ""


def open_device(input_path, serial, bypass_range_check):
    """Open a live camera (input_path empty) or a RAW file."""
    ensure_plugin_path()
    if input_path:
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        cfg = mh.RawFileConfig()
        cfg.do_time_shifting = True
        cfg.n_events_to_read = 100000
        device = mh.DeviceDiscovery.open_raw_file(input_path, cfg)
        if device is None:
            raise OSError("Could not open RAW file: %s" % input_path)
        return device

    sources = mh.DeviceDiscovery.list()
    if not sources:
        raise OSError(
            "No event camera found.\n"
            "  - is the SilkyEvCam plugged in and not already open in "
            "metavision_viewer.exe / Metavision Studio?\n"
            "  - MV_HAL_PLUGIN_PATH=%s" % os.environ.get("MV_HAL_PLUGIN_PATH"))
    print("[cam] available sources: %s" % sources)

    cfg = mh.DeviceConfig()
    if bypass_range_check:
        cfg.enable_biases_range_check_bypass(True)
    device = mh.DeviceDiscovery.open(serial, cfg)
    if device is None:
        raise OSError("Failed to open camera %s" % (serial or "(first available)"))
    return device


def describe_device(device):
    hw = device.get_i_hw_identification()
    geometry = device.get_i_geometry()
    width, height = geometry.get_width(), geometry.get_height()
    info = hw.get_system_info()
    print("[cam] %s %s serial=%s %dx%d format=%s" % (
        hw.get_integrator(), info.get("Sensor Name", "?"), hw.get_serial(),
        width, height, info.get("Current Data Encoding Format", "?")))
    return width, height


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class EventAccumulator:
    """Keeps the event slices covering the last accumulation_us microseconds."""

    def __init__(self, width, height, accumulation_us):
        self.width = width
        self.height = height
        self.accumulation_us = accumulation_us
        self.slices = deque()        # (end_timestamp, events)
        self.count = 0

    def clear(self):
        self.slices.clear()
        self.count = 0

    def add(self, events, end_ts):
        if events is not None and len(events):
            self.slices.append((end_ts, events))
            self.count += len(events)
        while self.slices and end_ts - self.slices[0][0] > self.accumulation_us:
            _, dropped = self.slices.popleft()
            self.count -= len(dropped)

    def render(self, palette):
        _, bg, on_color, off_color = palette
        frame = np.full((self.height, self.width, 3), bg, dtype=np.uint8)
        for _, events in self.slices:
            x = events["x"].astype(np.int32)
            y = events["y"].astype(np.int32)
            off = events["p"] == 0
            on = ~off
            if off.any():
                frame[y[off], x[off]] = off_color
            if on.any():
                frame[y[on], x[on]] = on_color
        return frame


class RotorAnalyzer:
    """Rolling per-block spectrum of the event rate, to surface a rotor line.

    A spinning rotor modulates the event rate under its blades at the
    blade-pass frequency. The whole-sensor rate is useless for finding it - a
    fan covering a few percent of the pixels reached only SNR 4.0 that way,
    lost in the noise of every other pixel. Cutting the sensor into blocks and
    transforming each block's rate separately found the same fan at SNR 10.1,
    so that is what this does: keep a ring buffer of per-block event counts and
    report whichever block carries the strongest periodic component.
    """

    def __init__(self, width, height, block=32, bin_us=500, history_s=1.5,
                 fmin=20.0, fmax=900.0):
        self.block = block
        self.bin_us = bin_us
        self.nx = (width + block - 1) // block
        self.ny = (height + block - 1) // block
        self.nblocks = self.nx * self.ny
        self.nbins = int(history_s * 1e6 / bin_us)
        self.freq = np.fft.rfftfreq(self.nbins, bin_us / 1e6)
        self.band = (self.freq >= fmin) & (self.freq <= fmax)
        self.window = np.hanning(self.nbins).astype(np.float32)
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.nblocks, self.nbins), np.float32)
        self.origin = None
        self.last_g = -1
        self.filled = 0
        self.result = None          # (freq_hz, snr, bx, by)

    def add(self, events):
        if events is None or not len(events):
            return
        t = events["t"].astype(np.int64)
        if self.origin is None:
            self.origin = int(t.min())
        g = (t - self.origin) // self.bin_us
        gmax = int(g.max())
        if gmax < self.last_g:      # stream restarted, timestamps went back
            self.reset()
            return
        # A slice is not guaranteed to be sorted - this camera emits EVT3
        # protocol violations (NonMonotonicTimeHigh) that put a stray earlier
        # or wildly out-of-range timestamp in the middle of a buffer. Keep only
        # what lands inside the window the ring buffer can represent, which
        # bounds nloc by nbins and keeps every index non-negative.
        lo = max(0, gmax - self.nbins + 1, self.last_g - self.nbins + 1)
        keep = g >= lo
        if not keep.any():
            return
        if not keep.all():
            g = g[keep]
            events = events[keep]
        # bins entered for the first time must be cleared of their stale round
        if self.last_g >= 0 and gmax > self.last_g:
            span = min(gmax - self.last_g, self.nbins)
            stale = np.arange(self.last_g + 1, self.last_g + 1 + span) % self.nbins
            self.hist[:, stale] = 0.0
        gmin = int(g.min())
        nloc = gmax - gmin + 1
        blk = ((events["y"].astype(np.int64) // self.block) * self.nx
               + events["x"].astype(np.int64) // self.block)
        flat = blk * nloc + (g - gmin)
        counts = np.bincount(flat, minlength=self.nblocks * nloc)
        counts = counts[:self.nblocks * nloc].reshape(self.nblocks, nloc)
        self.hist[:, np.arange(gmin, gmax + 1) % self.nbins] += counts.astype(np.float32)
        self.last_g = max(self.last_g, gmax)
        self.filled = min(self.filled + nloc, self.nbins)

    def analyze(self, min_events=150):
        """Transform every block with enough events, not just the busiest.

        Shortlisting the busiest blocks does not work: the noisiest blocks are
        busier than the rotor, so a top-k shortlist locks onto sensor edge
        noise and never transforms the block the rotor is actually in.
        """
        if self.filled < self.nbins // 2:
            return None
        totals = self.hist.sum(axis=1)
        idx = np.flatnonzero(totals >= min_events)
        if idx.size == 0:
            idx = np.flatnonzero(totals > 0)
        if idx.size == 0:
            return None
        rows = np.roll(self.hist[idx], -((self.last_g + 1) % self.nbins), axis=1)
        rows = rows - rows.mean(axis=1, keepdims=True)
        mag = np.abs(np.fft.rfft(rows * self.window, axis=1))[:, self.band]
        f = self.freq[self.band]
        peak = np.argmax(mag, axis=1)
        snr = mag[np.arange(len(mag)), peak] / (np.median(mag, axis=1) + 1e-9)
        best = int(np.argmax(snr))
        bid = int(idx[best])
        self.result = (float(f[peak[best]]), float(snr[best]),
                       bid % self.nx, bid // self.nx)
        return self.result


def draw_overlay(frame, lines, scale):
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_NEAREST)
    y = 18
    for text, color in lines:
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += 18
    return frame


PANEL_W = 560
ROW_H = 30


def draw_panel(biases, selected, status_lines):
    """Status panel drawn under the trackbars of the control window."""
    rows = max(1, len(biases.names))
    height = 34 + rows * ROW_H + 20 + 18 * len(status_lines)
    panel = np.full((height, PANEL_W, 3), 32, dtype=np.uint8)

    cv2.putText(panel, "bias            value   settable     recommended",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1,
                cv2.LINE_AA)

    for i, name in enumerate(biases.names):
        top = 34 + i * ROW_H
        value = biases.values[name]
        lo, hi = biases.limits.get(name, biases.ranges[name])
        rlo, rhi = biases.recommended.get(name, (lo, hi))

        if i == selected:
            cv2.rectangle(panel, (4, top), (PANEL_W - 5, top + ROW_H - 4),
                          (70, 70, 70), -1)
        color = (90, 160, 255) if biases.out_of_recommended(name) else (220, 220, 220)
        if not biases.modifiable.get(name, True):
            color = (120, 120, 120)

        cv2.putText(panel, name, (10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        cv2.putText(panel, "%5d" % value, (150, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(panel, "%d-%d" % (lo, hi), (215, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, "%d-%d" % (rlo, rhi), (310, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

        bar_x0, bar_x1 = 395, PANEL_W - 12
        cv2.rectangle(panel, (bar_x0, top + 10), (bar_x1, top + 18), (60, 60, 60), -1)
        if hi > lo:
            filled = int(bar_x0 + (bar_x1 - bar_x0) * (value - lo) / float(hi - lo))
            cv2.rectangle(panel, (bar_x0, top + 10), (filled, top + 18), color, -1)

    y = 34 + rows * ROW_H + 16
    for text, color in status_lines:
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)
        y += 18
    return panel


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
# Windows waitKeyEx codes for the navigation keys.
K_LEFT, K_UP, K_RIGHT, K_DOWN = 2424832, 2490368, 2555904, 2621440
K_PGUP, K_PGDN = 2162688, 2228224


class Studio:
    def __init__(self, args):
        self.args = args
        self.offline = bool(args.input)
        self.device = open_device(args.input, args.serial, args.bypass_range_check)
        self.width, self.height = describe_device(self.device)
        self.biases = BiasControl(self.device, args.bypass_range_check)

        if not self.biases.live:
            print("[bias] this source exposes no bias facility (read-only view)")
        else:
            for name in self.biases.names:
                print("[bias] %-14s %4d  %s" % (name, self.biases.values[name],
                                                self.biases.descriptions[name]))

        self.bias_file = args.bias
        if args.bias:
            self.load_bias_file(args.bias)
        elif self.biases.live and args.preset:
            self.biases.apply(ROTOR, "rotor preset")

        self.record_dir = args.record_dir or os.getcwd()
        self.accumulation_us = int(args.accumulation_time * 1000)
        self.accumulator = EventAccumulator(self.width, self.height, self.accumulation_us)
        self.palette_index = 0
        self.selected = 0
        self.paused = False
        self.recording_path = None
        self.pending = {}            # trackbar callbacks drop values here
        self.bias_intent = {}        # what the keys have ASKED for, per bias
        self._suspend_callbacks = False
        self.rate_hz = 0.0
        self.on_frac = 0.0
        self.display_fps = 0.0
        self.drained = 1
        self.rotor = RotorAnalyzer(self.width, self.height,
                                   fmin=args.rotor_fmin, fmax=args.rotor_fmax)
        self.show_rotor = not args.no_rotor
        self.next_analysis = 0.0
        self.message = ""
        self.message_until = 0.0
        self.running = True

        set_timer_resolution(True)
        self._build_windows()

    # -- windows ------------------------------------------------------------
    def _build_windows(self):
        cv2.namedWindow(VIEW_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
        for name in self.biases.names:
            lo, hi = self._trackbar_range(name)
            start = min(max(self.biases.values[name], lo), hi)
            cv2.createTrackbar(name, CTRL_WIN, start, hi,
                               self._make_callback(name))
            try:
                cv2.setTrackbarMin(name, CTRL_WIN, lo)
            except cv2.error:
                pass
        cv2.createTrackbar("accum_ms", CTRL_WIN, int(self.accumulation_us / 1000),
                           200, self._on_accumulation)
        try:
            cv2.setTrackbarMin("accum_ms", CTRL_WIN, 1)
        except cv2.error:
            pass

    def _trackbar_range(self, name):
        """The span worth putting on a slider, which is not the reported range.

        The HAL advertises 0-1800 for bias_diff_on/off, but Gen3.1 only accepts
        diff_off <= diff-65 and diff_on >= diff+75, neither further than 200
        from diff. Spreading 1800 values over the ~300 pixels of a trackbar put
        roughly six counts under every pixel, so a single step was simply not
        selectable - which defeats the point of the panel. Offering the window
        the sensor will actually take puts one count per pixel and makes the
        illegal values unreachable instead of silently rejected.

        Computed from the CURRENT bias_diff. Move bias_diff and these two
        windows shift under you; restart to rebuild the sliders (bias_diff is
        the one Prophesee advise not to touch anyway).
        """
        lo, hi = self.biases.limits[name]
        diff = self.biases.values.get("bias_diff")
        if diff is not None:
            if name == "bias_diff_off":
                lo, hi = max(lo, 100), min(hi, diff - 65)
            elif name == "bias_diff_on":
                lo, hi = max(lo, diff + 75), min(hi, diff + 200)
        return (lo, hi) if lo < hi else self.biases.limits[name]

    def _make_callback(self, name):
        def callback(value):
            if not self._suspend_callbacks:
                self.pending[name] = value
        return callback

    def _on_accumulation(self, value):
        if self._suspend_callbacks:
            return
        self.accumulation_us = max(1, value) * 1000
        self.accumulator.accumulation_us = self.accumulation_us

    def _sync_trackbar(self, name):
        self._suspend_callbacks = True
        cv2.setTrackbarPos(name, CTRL_WIN, self.biases.values[name])
        self._suspend_callbacks = False

    def _sync_all_trackbars(self):
        for name in self.biases.names:
            self._sync_trackbar(name)

    def notify(self, text, seconds=3.0):
        print("[ui] %s" % text)
        self.message = text
        self.message_until = time.time() + seconds

    # -- bias actions -------------------------------------------------------
    def bump_selected(self, delta):
        """Nudge the selected bias, and say so when the sensor refuses.

        The constraint that stops you is not in `limits`, which is why this
        used to look like a dead key: Gen3.1 enforces diff_off <= diff-65 and
        diff_on >= diff+75 inside the HAL, so walking diff_on down lands on 374
        (at diff=299) and every further press is rejected in silence, with the
        panel and the sensor free to drift apart. Clamping to the real window
        first keeps them together; the notify turns a refusal into a message
        instead of a mystery.

        Steps accumulate against what the keys have ASKED for, not against what
        the sensor reads back. The Gen3.1 bias DAC quantises to several counts,
        so stepping from the readback deadlocks: +1 asks for 371, the DAC snaps
        it back to 370, the next +1 asks for 371 again, and the value can never
        climb. Tracking the request separately means repeated presses walk past
        the quantisation boundary and the reading jumps when it is crossed -
        the way an exposure dial behaves.

        bias_intent is re-seeded from the sensor whenever the two have drifted
        further apart than quantisation explains, so loading a .bias file or
        hitting a preset does not leave the keys pushing against a stale value.
        """
        if not self.biases.live or not self.biases.names:
            return
        name = self.biases.names[self.selected]
        lo, hi = self._trackbar_range(name)
        held = self.biases.values[name]
        before = self.bias_intent.get(name, held)
        if abs(before - held) > 8:          # something else moved the bias
            before = held
        before = min(max(before, lo), hi)
        target = min(max(before + delta, lo), hi)
        if target == before:
            why = ("Gen3.1 requires diff_off <= diff-65 and diff_on >= diff+75"
                   if name in ("bias_diff_off", "bias_diff_on")
                   else "end of the recommended range")
            self.notify("%s already at its %s (holding %d) - %s" % (
                name, "minimum" if delta < 0 else "maximum", held, why))
            return
        after = self.biases.set(name, target)
        self.bias_intent[name] = target
        self._sync_trackbar(name)
        if after != target:
            self.notify("%s: asked %d, sensor holds %d (DAC quantisation)"
                        % (name, target, after))

    def load_bias_file(self, path):
        if not path or not os.path.exists(path):
            self.notify("no bias file at %s" % path)
            return
        values = read_bias_file(path)
        if not self.biases.live:
            self.biases.values.update({k: v for k, v in values.items() if k in BIAS_NAMES})
            self.notify("showing biases from %s (read-only)" % os.path.basename(path))
            return
        self.biases.apply(values, label=os.path.basename(path))
        if hasattr(self, "pending"):
            self._sync_all_trackbars()
        self.notify("loaded %s" % os.path.basename(path))

    def save_bias_file(self, path=None):
        if path is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.record_dir, "biases_%s.bias" % stamp)
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        write_bias_file(path, self.biases.values)
        self.bias_file = path
        self.notify("saved %s" % path)
        return path

    # -- recording ----------------------------------------------------------
    def toggle_recording(self):
        stream = self.device.get_i_events_stream()
        if stream is None:
            self.notify("no events stream to record")
            return
        if self.recording_path:
            stream.stop_log_raw_data()
            self.notify("recording stopped: %s" % self.recording_path)
            self.recording_path = None
            return
        if self.offline:
            self.notify("cannot record while replaying a file")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.record_dir, exist_ok=True)
        path = os.path.join(self.record_dir, "rec_%s.raw" % stamp)
        stream.log_raw_data(path)
        self.recording_path = path
        # keep the exact biases next to the recording, like metavision_viewer does
        write_bias_file(os.path.splitext(path)[0] + ".bias", self.biases.values)
        self.notify("recording to %s" % path)

    # -- main loop ----------------------------------------------------------
    def run(self):
        iterator = EventsIterator.from_device(
            device=self.device, mode="delta_t",
            delta_t=int(self.args.refresh_time * 1000))
        stream = iter(iterator)
        last_ts = 0
        rate_window = deque()
        frame_times = deque(maxlen=30)
        next_deadline = time.time()

        while self.running:
            if not self.paused:
                last_ts = self._pump(stream, rate_window, last_ts)

                if self.offline and self.args.realtime:
                    next_deadline += self.args.refresh_time / 1000.0
                    delay = next_deadline - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_deadline = time.time()

            if self.show_rotor and time.time() >= self.next_analysis:
                self.rotor.analyze()
                self.next_analysis = time.time() + 0.25

            self._draw(last_ts)
            frame_times.append(time.time())
            if len(frame_times) > 1:
                span = frame_times[-1] - frame_times[0]
                self.display_fps = (len(frame_times) - 1) / span if span > 0 else 0.0

            self._handle_keys(cv2.waitKeyEx(1))
            self._flush_pending()

            if cv2.getWindowProperty(VIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False

        self.shutdown()

    def _pump(self, stream, rate_window, last_ts):
        """Consume every slice already buffered, not just one per frame.

        The camera emits one slice per refresh_time no matter how fast we
        draw. Taking exactly one slice per displayed frame means a backlog is
        never worked off - including the one that builds up between starting
        the stream and the first frame - which showed up as a constant ~260 ms
        delay between something happening and it appearing. A next() that
        returns in well under a slice duration came from that backlog, so keep
        pulling until one actually blocks (we are live again) or the frame
        budget runs out. Only the newest events survive in the accumulator, so
        catching up costs nothing visually.

        File replay never blocks, so there it stays at one slice per frame.
        """
        slice_s = self.args.refresh_time / 1000.0
        budget = time.time() + max(0.020, slice_s)
        drained = 0
        while True:
            started = time.time()
            try:
                events = next(stream)
            except StopIteration:
                self.notify("end of stream", 10.0)
                self.paused = True
                break
            took = time.time() - started
            if len(events):
                last_ts = int(events["t"][-1])
            self.accumulator.add(events, last_ts)
            if self.show_rotor:
                self.rotor.add(events)
            drained += 1

            now = time.time()
            # ON share, over the same one-second window as the rate. A dead
            # polarity is invisible in the event rate but halves a rotor: each
            # blade passage is one ON (leading edge) and one OFF (trailing),
            # so a stream at 5% ON is missing half of every cycle while still
            # looking busy. On this sensor bias_diff_on >= ~474 kills the ON
            # channel outright - measured 0.3% ON at 495 - and nothing else on
            # screen reveals it.
            n_on = int((events["p"] == 1).sum()) if len(events) else 0
            rate_window.append((now, len(events), n_on))
            while rate_window and now - rate_window[0][0] > 1.0:
                rate_window.popleft()
            span = max(1e-3, now - rate_window[0][0]) if rate_window else 1.0
            total = sum(n for _, n, _ in rate_window)
            self.rate_hz = total / span
            self.on_frac = (sum(k for _, _, k in rate_window) / float(total)
                            if total else 0.0)

            if self.offline or took > slice_s / 2.0 or now > budget:
                break
        self.drained = drained
        return last_ts

    def _flush_pending(self):
        """Apply trackbar moves collected since the last frame.

        The Gen3.1 bias DAC quantises: asking for 409 can read back as 404. That
        is normal, so the trackbar is only forced back into place when the value
        was genuinely clamped to the allowed range - otherwise setTrackbarPos
        would fight the mouse on every drag event.
        """
        if not self.pending:
            return
        pending, self.pending = self.pending, {}
        for name, value in pending.items():
            held = self.biases.set(name, value)
            lo, hi = self.biases.limits[name]
            # Sync back whenever the sensor did not take the request. The old
            # test only caught values outside `limits`, but the inter-bias
            # constraints (diff_off <= diff-65, diff_on >= diff+75) live in the
            # HAL and not in `limits` - so dragging past one left the slider
            # showing a value the sensor never held, and every later nudge from
            # there was clamped straight back to the same place and looked like
            # the panel had stopped responding. The 8-count margin is there to
            # let the DAC's own quantisation through (409 reads back as 404)
            # without setTrackbarPos fighting the mouse mid-drag.
            if not (lo <= value <= hi) or abs(held - value) > 8:
                self._sync_trackbar(name)

    def _draw(self, last_ts):
        palette = PALETTES[self.palette_index]
        frame = self.accumulator.render(palette)

        rotor_line = None
        if self.show_rotor and self.rotor.result:
            freq, snr, bx, by = self.rotor.result
            blk = self.rotor.block
            strong = snr >= self.args.rotor_snr
            if strong:
                cv2.rectangle(frame, (bx * blk, by * blk),
                              (bx * blk + blk - 1, by * blk + blk - 1),
                              (0, 255, 255), 1)
            rotor_line = ("rotor %6.1f Hz  SNR %4.1f  at (%d,%d)%s" % (
                freq, snr, bx * blk + blk // 2, by * blk + blk // 2,
                "" if strong else "  (weak)"),
                (0, 255, 255) if strong else (140, 140, 140))

        rate = (("%.2f Mev/s" % (self.rate_hz / 1e6)) if self.rate_hz >= 1e6
                else ("%.1f kev/s" % (self.rate_hz / 1e3)))
        lines = [
            ("%dx%d  %s  accum %d ms  %.0f fps" % (
                self.width, self.height, rate, self.accumulation_us // 1000,
                self.display_fps), (0, 255, 0)),
            ("t=%8.3f s  palette=%s%s" % (
                last_ts / 1e6, palette[0], "  [PAUSED]" if self.paused else ""),
             (0, 255, 0)),
        ]
        on_pct = 100.0 * self.on_frac
        balanced = 20.0 <= on_pct <= 80.0
        lines.append(("ON %4.1f%%  /  OFF %4.1f%%%s" % (
            on_pct, 100.0 - on_pct,
            "" if balanced else "   <- polarity starved: check diff_on / diff_off"),
            (0, 255, 0) if balanced else (0, 220, 255)))
        if self.drained > 2 and not self.offline:
            lines.append(("catching up: %d slices/frame - lower the event rate "
                          "or the accumulation time" % self.drained, (0, 220, 255)))
        if rotor_line:
            lines.append(rotor_line)
        if self.recording_path:
            lines.append(("REC -> %s" % os.path.basename(self.recording_path),
                          (0, 0, 255)))
        if time.time() < self.message_until:
            lines.append((self.message, (0, 220, 255)))
        cv2.imshow(VIEW_WIN, draw_overlay(frame, lines, self.args.scale))

        status = []
        if self.biases.names:
            name = self.biases.names[self.selected]
            status.append(("> %s: %s" % (name, HINTS.get(name, "")), (0, 220, 255)))
        warning = self.biases.consistency_warning()
        if warning:
            status.append((warning, (90, 160, 255)))
        if not self.biases.live:
            status.append(("read-only: no live camera", (120, 120, 120)))
        status.append(("arrows select/adjust   pgup/pgdn +-10   , . +-50",
                       (150, 150, 150)))
        status.append(("d startup  n nominal  l load  s save  r rec  q quit",
                       (150, 150, 150)))
        cv2.imshow(CTRL_WIN, draw_panel(self.biases, self.selected, status))

    def _handle_keys(self, key):
        if key == -1:
            return
        code = key & 0xFF

        if key in (K_UP, K_DOWN) and self.biases.names:
            step = -1 if key == K_UP else 1
            self.selected = (self.selected + step) % len(self.biases.names)
        elif key == K_LEFT:
            self.bump_selected(-1)
        elif key == K_RIGHT:
            self.bump_selected(+1)
        elif key == K_PGUP:
            self.bump_selected(+10)
        elif key == K_PGDN:
            self.bump_selected(-10)
        elif code == ord(","):
            self.bump_selected(-50)
        elif code == ord("."):
            self.bump_selected(+50)
        elif code in (ord("q"), 27):
            self.running = False
        elif code == ord(" "):
            self.paused = not self.paused
        elif code == ord("c"):
            self.accumulator.clear()
        elif code == ord("p"):
            self.palette_index = (self.palette_index + 1) % len(PALETTES)
        elif code == ord("a"):
            self._set_accumulation(self.accumulation_us // 1000 - 5)
        elif code == ord("z"):
            self._set_accumulation(self.accumulation_us // 1000 + 5)
        elif code == ord("d"):
            self.biases.apply(self.biases.startup, "startup values")
            self._sync_all_trackbars()
            self.notify("restored startup biases")
        elif code == ord("f"):
            self.biases.apply(ROTOR, "rotor preset")
            self._sync_all_trackbars()
            self.notify("applied rotor preset")
        elif code == ord("e"):
            self.show_rotor = not self.show_rotor
            self.rotor.reset()
            self.notify("rotor readout %s" % ("on" if self.show_rotor else "off"))
        elif code == ord("n"):
            self.biases.apply(NOMINAL, "nominal defaults")
            self._sync_all_trackbars()
            self.notify("restored nominal biases")
        elif code == ord("l"):
            self.load_bias_file(self.bias_file)
        elif code == ord("s"):
            self.save_bias_file()
        elif code == ord("r"):
            self.toggle_recording()
        elif code == ord("h"):
            print(__doc__)

    def _set_accumulation(self, milliseconds):
        milliseconds = int(max(1, min(200, milliseconds)))
        self.accumulation_us = milliseconds * 1000
        self.accumulator.accumulation_us = self.accumulation_us
        self._suspend_callbacks = True
        cv2.setTrackbarPos("accum_ms", CTRL_WIN, milliseconds)
        self._suspend_callbacks = False

    def shutdown(self):
        if self.recording_path:
            try:
                self.device.get_i_events_stream().stop_log_raw_data()
                print("[rec] stopped: %s" % self.recording_path)
            except Exception:
                pass
            self.recording_path = None
        cv2.destroyAllWindows()
        set_timer_resolution(False)
        if self.biases.live:
            print("[bias] final values:")
            for name in self.biases.names:
                print("  %-4d %% %s" % (self.biases.values[name], name))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live viewer and interactive bias tuner for the SilkyEvCam VGA.")
    parser.add_argument("-i", "--input", default="",
                        help="RAW file to replay instead of the live camera")
    parser.add_argument("-S", "--serial", default="",
                        help="camera serial (default: first available)")
    parser.add_argument("-b", "--bias", default="",
                        help=".bias file loaded at startup and reloaded with l")
    parser.add_argument("-o", "--record-dir", default="",
                        help="directory for r recordings and s bias files")
    parser.add_argument("--accumulation-time", type=float, default=33.0,
                        help="displayed event history, in ms (default 33, "
                             "matching the detector's window. Note a rotor "
                             "blurs into a solid disc once this exceeds one "
                             "blade-pass period - ~1.7 ms at the Tello's 590 Hz "
                             "- so drop it to 2-4 to see individual blades)")
    parser.add_argument("--refresh-time", type=float, default=10.0,
                        help="event slice length fetched per frame, in ms (default 10)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="display scale factor (default 1.0)")
    parser.add_argument("--preset", action="store_true",
                        help="apply the OFF-only rotor preset at startup "
                             "(default: leave the camera on its own biases)")
    parser.add_argument("--no-rotor", action="store_true",
                        help="disable the rotor frequency readout")
    parser.add_argument("--rotor-fmin", type=float, default=20.0,
                        help="lowest rotor frequency to look for, Hz (default 20)")
    parser.add_argument("--rotor-fmax", type=float, default=900.0,
                        help="highest rotor frequency to look for, Hz (default 900)")
    parser.add_argument("--rotor-snr", type=float, default=6.0,
                        help="SNR above which a detection is highlighted (default 6)")
    parser.add_argument("--realtime", action="store_true",
                        help="throttle RAW replay to real time")
    parser.add_argument("--bypass-range-check", action="store_true",
                        help="unlock the full 0-1800 range (default: the sliders "
                             "stay inside the range the HAL accepts)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        studio = Studio(args)
    except (OSError, FileNotFoundError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    try:
        studio.run()
    except KeyboardInterrupt:
        studio.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
