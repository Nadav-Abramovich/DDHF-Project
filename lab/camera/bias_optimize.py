#!/usr/bin/env python
"""
Bayesian optimisation of the SilkyEvCam biases for rotor detectability.

Scores candidate bias settings with THIS PROJECT'S rotor detector rather than
an ad-hoc metric: code/utils.py's _detect_rotor_pixels runs the DDHF pipeline
(per-pixel NDFT -> spectral flatness -> harmonic-comb SNR) and returns the
rotor pixel mask plus the comb-SNR of every pixel. The objective is built from
exactly those two things:

    score = w_snr * log10(1 + comb_snr) + w_events * log10(1 + rotor_pixels)

so "maximise the SNR and event count of the fan" is measured with the same
gates (min_events, tau_sf, tau_comb) the detector itself uses downstream. Both
terms are logarithmic so neither runs away with the objective.

The surrogate is utility/bayes_working_point.py's Gaussian process - Matern 5/2
with lengthscale and noise chosen by marginal likelihood - and its Expected
Improvement, imported rather than reimplemented. Candidates are proposed over
the seven biases, repaired to keep bias_diff_off < bias_diff < bias_diff_on,
and appended to a CSV after every trial, so a run resumes where it stopped.

Where the rotor is
    Located once at nominal biases by running the same detector over the whole
    sensor and keeping the pixels it flags; scoring is then restricted to that
    region so the optimiser cannot win by finding noise elsewhere. Override
    with --roi if you would rather fix it yourself.

Costs and hazards measured on this camera
    Some bias combinations flood the sensor - bias_fo near its lower bound hit
    49 Mev/s, which overruns USB and desynchronises the EVT3 decoder until the
    device is reopened. Every trial is therefore rate-probed first and rejected
    outright if it floods, and a stalled stream is detected and recovered by
    reopening the camera.

Usage
    python bias_optimize.py                          # locate rotor, 60 trials
    python bias_optimize.py --trials 120 --seconds 0.7
    python bias_optimize.py --roi 128 96 288 352     # fixed region
    python bias_optimize.py --csv fan_search.csv     # trials logged here
    python bias_optimize.py --apply                  # leave the best on the camera
"""
import os
import sys
import csv
import gc
import json
import time
import argparse
import datetime
import subprocess

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, uniform_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB  = os.path.dirname(_HERE)                       # lab/ (data/, runs/, compare_replay)
for _p in (_HERE, os.path.join(_LAB, "code", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import silky_bias_studio as studio           # DLL/plugin bootstrap + HAL helpers
from metavision_core.event_io import EventsIterator

from utils import (DDHFConfig, _detect_rotor_pixels, _normalise_polarity,
                   _normalise_timestamps)
import ddhf_config
import bayes_working_point as BWP             # gp_fit / gp_predict / EI

BIAS_NAMES = studio.BIAS_NAMES
NOMINAL = studio.NOMINAL


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bayesian optimisation of SilkyEvCam biases for rotor detectability.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--init", type=int, default=14,
                   help="random trials before the GP takes over (default 14)")
    p.add_argument("--seconds", type=float, default=0.5,
                   help="capture per trial; DDHF windows are 33.3 ms, so 0.5 s "
                        "is 15 windows (default 0.5)")
    p.add_argument("--settle", type=float, default=0.35,
                   help="seconds discarded after changing biases (default 0.35)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="rotor region in pixels; default is to locate it")
    p.add_argument("--pad", type=int, default=16,
                   help="pixels of margin around the located rotor (default 16)")
    p.add_argument("--w-snr", type=float, default=1.0)
    p.add_argument("--w-events", type=float, default=0.5)
    p.add_argument("--flood-rate", type=float, default=20e6,
                   help="events/s above which a candidate is rejected outright "
                        "(default 20e6), caught from a short probe because "
                        "flooding settings are the most expensive to measure")
    p.add_argument("--probe", type=float, default=0.15,
                   help="seconds used to estimate the rate first (default 0.15)")
    p.add_argument("--working-point", default="",
                   help="working-point JSON (utility/working_point_*.json). Its "
                        "params set the detector gates, including the component "
                        "gates DDHFConfig has no field for, and its objective "
                        "block supplies the default penalty and hit floor")
    p.add_argument("--objective", choices=("detect", "snr"), default="detect",
                   help="detect: precision on the rotor with a hit-rate floor "
                        "(default). snr: raw comb-SNR and event count, which a "
                        "flooding bias set can game")
    p.add_argument("--penalty", type=float, default=None,
                   help="penalty per unit of missing hit rate (default: from "
                        "the working point, else 3.0)")
    p.add_argument("--min-hit", type=float, default=None,
                   help="hit-rate floor below which the penalty bites "
                        "(default: from the working point, else 0.8)")
    p.add_argument("--trial-timeout", type=float, default=60.0,
                   help="seconds before a wedged trial is killed "
                        "(default 60)")
    p.add_argument("--pool", type=int, default=4,
                   help="BxB spatial pooling before the NDFT, the preset's "
                        "pixelize mode (default 3). A live rotor here is far "
                        "sparser than the cached recordings the preset was "
                        "tuned on - ~1-4 events per pixel per 33 ms window "
                        "against min_events=16 - so per-pixel grouping leaves "
                        "nothing to transform. Pooling BxB multiplies the "
                        "events per group by B^2. Use 1 for per-pixel.")
    p.add_argument("--csv", default=os.path.join(_LAB, "data", "bias_search", "bias_opt.csv"),
                   help="CSV of every trial and its score. Loaded at startup "
                        "and appended to after each trial, so a run resumes "
                        "where it stopped and survives a kill "
                        "(default: data/bias_search/bias_opt.csv)")
    p.add_argument("--reset", action="store_true",
                   help="start a fresh search, ignoring the existing CSV rows")
    p.add_argument("--apply", action="store_true",
                   help="leave the best configuration on the camera at exit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="")
    # The detector's own gates (--min-events, --tau-sf, --tau-comb, ...) come
    # from ddhf_config so they mean exactly what they mean everywhere else.
    ddhf_config.add_cfg_cli_args(p)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------
class _Stalled(Exception):
    pass


class Rig:
    """Owns the device and iterator; probes for floods and recovers stalls."""

    def __init__(self, args):
        self.args = args
        self.reopens = 0
        self.last_rate = 0.0
        self.open()

    def open(self, attempts=4):
        # Drop the old handle BEFORE opening a new one. Reassigning self.device
        # releases the previous device only after the new open has run, so the
        # old one still holds the USB claim and the open fails with
        # LIBUSB_ERROR_ACCESS - which turned every recoverable stall into a
        # hard failure.
        self.stream = None
        self.iterator = None
        self.device = None
        gc.collect()
        last = None
        for attempt in range(attempts):
            try:
                self.device = studio.open_device("", "", False)
                break
            except OSError as exc:
                last = exc
                time.sleep(0.6 * (attempt + 1))   # let the OS release the claim
        if self.device is None:
            raise last
        self.width, self.height = studio.describe_device(self.device)
        self.biases = studio.BiasControl(self.device)
        self.iterator = EventsIterator.from_device(
            device=self.device, mode="delta_t", delta_t=5000)
        self.stream = iter(self.iterator)
        self.first_call = True

    def reopen(self, why):
        self.reopens += 1
        print("    [rig] reopening camera (%s)" % why, flush=True)
        self.open()
        time.sleep(0.5)

    def _next(self):
        """One slice, with a stall guard. The first slice after opening carries
        the stream start-up cost, so it gets a longer budget."""
        budget = 8.0 if self.first_call else 2.0
        t0 = time.time()
        e = next(self.stream)
        self.first_call = False
        if time.time() - t0 > budget:
            raise _Stalled()
        return e

    def capture(self, seconds, roi=None, attempts=2, on_reopen=None):
        """Raw events, optionally clipped to `roi`. Returns (events, status).

        Clipping during capture is what makes a busy scene affordable: the rotor
        region is a few percent of the sensor, so a 20 Mev/s flood becomes a few
        hundred thousand events instead of tens of millions.
        """
        for attempt in range(attempts):
            ev, status = self._capture_once(seconds, roi)
            if status != "stall":
                return ev, status
            if attempt < attempts - 1:
                self.reopen("stall (attempt %d/%d)" % (attempt + 1, attempts))
                if on_reopen is not None:
                    on_reopen()          # the reopen reset the biases to factory
        return None, "stall"

    def _drain_to_live(self, cap=2.5):
        """Consume the backlog until next() actually waits for fresh data.

        Everything downstream depends on this. A slice that returns instantly
        came out of the HAL's buffer, not off the sensor, so measuring while
        draining reports a rate several times the real one - which is what made
        the nominal configuration look like a 33 Mev/s flood.
        """
        end = time.time() + cap
        while time.time() < end:
            t0 = time.time()
            self._next()
            if time.time() - t0 > 0.0025:        # waited for the sensor: live
                return True
        return False

    def _span(self, span_s, roi=None, keep=True, wall_cap=None):
        """Collect until the DEVICE clock has advanced span_s seconds.

        Wall-clock windows are not comparable between candidates when the
        stream runs behind; device timestamps are.
        Returns (chunks, n_events, span_seconds).
        """
        chunks, n, t_first, t_last = [], 0, None, None
        end = time.time() + (wall_cap or max(5.0, span_s * 6))
        while True:
            e = self._next()
            if len(e):
                if t_first is None:
                    t_first = int(e["t"][0])
                t_last = int(e["t"][-1])
                n += len(e)
                if keep:
                    c = self._clip(e, roi)
                    if len(c):
                        chunks.append(c)
                if (t_last - t_first) / 1e6 >= span_s:
                    break
            if time.time() > end:
                raise _Stalled()
        return chunks, n, max(1e-6, (t_last - t_first) / 1e6)

    def _capture_once(self, seconds, roi):
        try:
            self._drain_to_live()
            self._span(self.args.settle, keep=False)     # discard post-change data
            probe, n_probe, span = self._span(self.args.probe, roi)
            self.last_rate = n_probe / span
            if self.last_rate > self.args.flood_rate:
                return None, "flood"
            rest, n_rest, span2 = self._span(max(0.0, seconds - self.args.probe), roi)
            chunks = probe + rest
            self.last_rate = (n_probe + n_rest) / (span + span2)
        except (_Stalled, StopIteration):
            return None, "stall"
        chunks = [c for c in chunks if c is not None and len(c)]
        if not chunks:
            return None, "empty"
        return np.concatenate(chunks), "ok"

    @staticmethod
    def _clip(e, roi):
        if roi is None or not len(e):
            return e
        x0, y0, x1, y1 = roi
        x, y = e["x"], e["y"]
        keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        return e[keep] if not keep.all() else e


# ---------------------------------------------------------------------------
# Scoring with the project's DDHF detector
# ---------------------------------------------------------------------------
def ddhf_windows(events, height, width, cfg, pool=1):
    """Run _detect_rotor_pixels over each DDHF window of a capture.

    Yields (rotor_mask, score_map) per window - the detector's own output, so
    the gates applied here are the ones the rest of the pipeline applies.
    """
    if events is None or not len(events):
        return
    t = events["t"].astype(np.int64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    x = events["x"][order].astype(np.int64)
    y = events["y"][order].astype(np.int64)
    p = events["p"][order]
    if pool > 1:
        x = x // pool
        y = y // pool
        height = (height + pool - 1) // pool
        width = (width + pool - 1) // pool
    window_us = int(cfg.window_ms * 1000.0)
    start = int(t[0])
    while start + window_us <= int(t[-1]) + 1:
        lo = np.searchsorted(t, start, "left")
        hi = np.searchsorted(t, start + window_us, "left")
        start += window_us
        if hi - lo < cfg.min_events:
            continue
        tn = _normalise_timestamps(t[lo:hi], window_span=float(window_us))
        pol = _normalise_polarity(p[lo:hi])
        yield _detect_rotor_pixels(x[lo:hi], y[lo:hi], pol, tn, height, width, cfg)


def score_capture(events, height, width, cfg, roi, args, bad=None, pool=1):
    """Rotor pixel count and comb-SNR, averaged over the capture's windows."""
    out = dict(rotor_px=0.0, snr=0.0, windows=0, measured_px=0.0)
    if events is None or not len(events):
        return out
    x0, y0, x1, y1 = [v // pool for v in roi]      # maps are at pooled scale
    bad_sub = None if bad is None else bad[y0:y1, x0:x1]
    counts, snrs, measured = [], [], []
    for rotor, scores in ddhf_windows(events, height, width, cfg, pool):
        sub_r = rotor[y0:y1, x0:x1]
        sub_s = scores[y0:y1, x0:x1]
        if bad_sub is not None:
            sub_r = sub_r & ~bad_sub
            sub_s = np.where(bad_sub, 0.0, sub_s)
        n = int(sub_r.sum())
        counts.append(n)
        measured.append(int((sub_s > 0).sum()))
        if n:
            snrs.append(float(np.median(sub_s[sub_r])))
        elif (sub_s > 0).any():
            # Nothing passed the gate, but keep a gradient for the surrogate:
            # a hard pass/fail objective is flat everywhere it fails, which
            # leaves the GP nothing to follow out of a bad region.
            snrs.append(float(np.percentile(sub_s[sub_s > 0], 95)))
        else:
            snrs.append(0.0)
    if not counts:
        return out
    out["windows"] = len(counts)
    out["rotor_px"] = float(np.mean(counts))
    out["measured_px"] = float(np.mean(measured))
    out["snr"] = float(np.mean(snrs))
    return out


def objective(m, args):
    return (args.w_snr * np.log10(1.0 + m["snr"])
            + args.w_events * np.log10(1.0 + m["rotor_px"]))


def defective_pixels(events, height, width, min_count=200, isolation=0.10,
                     pool=1):
    """Pixels that fire constantly on their own - sensor defects, not rotors.

    This camera has a stuck pixel at (20,420) that emits ~30 kev/s, 14% of the
    whole sensor's output. It clears min_events in every window and its steady
    firing scores a comb-SNR above tau_comb, so the rotor locator picked it in
    preference to the fan. A rotor blade sweeps a neighbourhood; a broken pixel
    is loud while everything touching it stays quiet, which is the test used
    here: busy, but its eight neighbours carry less than `isolation` of its own
    count.
    """
    ex = events["x"].astype(np.int64) // pool if events is not None else None
    ey = events["y"].astype(np.int64) // pool if events is not None else None
    height = (height + pool - 1) // pool
    width = (width + pool - 1) // pool
    bad = np.zeros((height, width), bool)
    if events is None or not len(events):
        return bad
    cnt = np.bincount(ey * width + ex, minlength=height * width).reshape(height, width)
    busy = cnt > min_count
    if not busy.any():
        return bad
    neigh = np.zeros_like(cnt)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh += np.roll(np.roll(cnt, dy, 0), dx, 1)
    bad = busy & (neigh < isolation * cnt)
    return bad


def locate_rotor(rig, cfg, args):
    """Whole-sensor DDHF at nominal biases; the flagged pixels define the ROI.

    Cheap despite covering every pixel: the detector drops anything with fewer
    than cfg.min_events before the NDFT, and on a VGA sensor almost nothing but
    the rotor clears that bar.
    """
    print("locating the rotor at nominal biases ...", flush=True)
    rig.biases.apply(NOMINAL)
    events, status = rig.capture(max(0.5, args.seconds))
    if events is None:
        raise SystemExit("could not capture a baseline (%s at %.1f Mev/s)"
                         % (status, rig.last_rate / 1e6))
    bad = defective_pixels(events, rig.height, rig.width, pool=args.pool)
    if bad.any():
        ys_b, xs_b = np.nonzero(bad)
        print("  ignoring %d stuck pixel(s): %s" % (
            bad.sum(), ", ".join("(%d,%d)" % (x, y)
                                 for x, y in zip(xs_b[:5], ys_b[:5]))), flush=True)
    ph = (rig.height + args.pool - 1) // args.pool
    pw = (rig.width + args.pool - 1) // args.pool
    hits = np.zeros((ph, pw), np.int32)
    best = np.zeros((ph, pw), np.float32)
    n = 0
    for rotor, scores in ddhf_windows(events, rig.height, rig.width, cfg, args.pool):
        hits += rotor
        best = np.maximum(best, scores)
        n += 1
    if n == 0:
        raise SystemExit("no complete DDHF window captured - increase --seconds")
    hits[bad] = 0
    best[bad] = 0.0
    if not hits.any():
        raise SystemExit(
            "the detector flagged nothing at nominal biases. Give --roi "
            "explicitly, lower --min-events, or raise --pool.")
    # Cluster the hits rather than demanding per-cell persistence. Measured on
    # this rotor: each cell passed in only 2-5 of 60 windows, but every one of
    # the top cells sat within a few cells of the others - a marginal rotor
    # lights a different cell of the same small area from window to window, so
    # no cell is persistent while the area plainly is. Requiring per-cell
    # persistence found nothing and the run could not bootstrap.
    dens = uniform_filter(hits.astype(np.float32), size=5)
    pk = float(dens.max())
    region = dens >= 0.4 * pk
    lab, nlab = label(region)
    if nlab:
        top = np.unravel_index(int(np.argmax(dens)), dens.shape)
        region = lab == lab[top]
    ys, xs = np.nonzero(region)
    pad, P = args.pad, args.pool
    roi = (max(0, int(xs.min()) * P - pad), max(0, int(ys.min()) * P - pad),
           min(rig.width, (int(xs.max()) + 1) * P + pad),
           min(rig.height, (int(ys.max()) + 1) * P + pad))
    print("  %d windows, %d cell-hits, densest cluster %d cells, peak comb-SNR %.1f"
          % (n, int(hits.sum()), int(region.sum()), float(best.max())), flush=True)
    print("  roi x %d-%d  y %d-%d" % (roi[0], roi[2], roi[1], roi[3]), flush=True)
    return roi, bad


# ---------------------------------------------------------------------------
# Trials run in a child process
# ---------------------------------------------------------------------------
# A wedged EVT3 decoder blocks inside next() and never returns - measured at 0%
# CPU while the run sat there indefinitely. No in-process timeout can rescue
# that: the guard only runs once the call returns. So each measurement happens
# in a short-lived child that the parent can kill on a deadline. The child also
# opens the camera fresh, which resets the biases and leaves no backlog or
# decoder desync to carry into the next trial.

def _worker(payload):
    args = argparse.Namespace(**payload["args"])
    cfg = DDHFConfig(**payload["cfg"])
    rig = Rig(args)
    if payload["mode"] == "info":
        return dict(status="ok", width=rig.width, height=rig.height,
                    limits={n: list(rig.biases.limits[n]) for n in BIAS_NAMES})
    values = payload.get("values") or dict(NOMINAL)
    rig.biases.apply(values)
    applied = {n: rig.biases.values[n] for n in BIAS_NAMES}
    roi = tuple(payload["roi"]) if payload.get("roi") else None
    events, status = rig.capture(args.seconds, roi)
    out = dict(status=status, applied=applied, rate=rig.last_rate)

    if payload["mode"] == "locate":
        if events is None:
            out["error"] = "capture failed (%s)" % status
            return out
        bad = defective_pixels(events, rig.height, rig.width, pool=args.pool)
        ph = (rig.height + args.pool - 1) // args.pool
        pw = (rig.width + args.pool - 1) // args.pool
        hits = np.zeros((ph, pw), np.int32)
        best = np.zeros((ph, pw), np.float32)
        n = 0
        for rotor, scores in ddhf_windows(events, rig.height, rig.width, cfg, args.pool):
            hits += rotor
            best = np.maximum(best, scores)
            n += 1
        hits[bad] = 0
        best[bad] = 0.0
        ys_b, xs_b = np.nonzero(bad)
        out.update(windows=n, hits=hits.tolist(), peak=float(best.max()),
                   bad=[[int(a), int(b)] for a, b in zip(xs_b, ys_b)])
        return out

    bad = None
    if payload.get("bad"):
        ph = (rig.height + args.pool - 1) // args.pool
        pw = (rig.width + args.pool - 1) // args.pool
        bad = np.zeros((ph, pw), bool)
        for bx, by in payload["bad"]:
            bad[by, bx] = True
    if events is None:
        out.update(snr=0.0, rotor_px=0.0, windows=0, measured_px=0.0)
        return out
    if payload.get("objective") == "snr":
        m = score_capture(events, rig.height, rig.width, cfg, roi, args,
                          bad, args.pool)
    else:
        m = score_detect(events, rig.height, rig.width, cfg, roi, args,
                         payload.get("gates", {}), bad, args.pool)
    out.update(m)
    return out


def run_child(payload, timeout):
    """Run one measurement in a child process; kill it if it wedges."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker-stdin"]
    try:
        proc = subprocess.run(cmd, input=json.dumps(payload), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="timeout")
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return dict(status="crash", error=(proc.stderr or "")[-300:])


# ---------------------------------------------------------------------------
# Detection scoring against a saved working point
# ---------------------------------------------------------------------------
# DDHFConfig carries the pixel gates; a working point also carries component
# gates (min_rotor_pixels, min_fill_ratio, min_snr_sum, min_snr_avg) that the
# frame pipeline applies after connected-component labelling. Those are what
# suppress false positives - the pixel gates alone pass about 1% of pure-noise
# groups per window, which over thousands of groups is a steady rain of
# isolated specks - so they are applied here too.
COMPONENT_GATES = ("min_rotor_pixels", "min_fill_ratio", "min_snr_sum",
                   "min_snr_avg", "min_box_area")


def load_working_point(path):
    """Split a working-point JSON into DDHFConfig fields and component gates."""
    with open(path) as fh:
        wp = json.load(fh)
    params = wp.get("params", wp)
    cfg_fields = set(ddhf_config._PRESET)
    preset = {k: v for k, v in params.items() if k in cfg_fields}
    gates = {k: float(params[k]) for k in COMPONENT_GATES if k in params}
    obj = wp.get("objective", {})
    return preset, gates, obj, wp


def blobs(rotor, scores, gates):
    """Connected components surviving the working point's component gates.

    Returns dicts with the box, pixel count and comb-SNR statistics, in the
    pooled coordinates the detector ran in.
    """
    out = []
    if not rotor.any():
        return out
    lab, n = ndimage.label(rotor, structure=np.ones((3, 3)))
    if n == 0:
        return out
    for lid, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == lid
        size = int(sub.sum())
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        area = h * w
        vals = scores[sl][sub]
        snr_sum = float(vals.sum())
        snr_avg = float(vals.mean()) if size else 0.0
        fill = size / float(area) if area else 0.0
        if size < gates.get("min_rotor_pixels", 0):
            continue
        if area < gates.get("min_box_area", 0):
            continue
        if fill < gates.get("min_fill_ratio", 0.0):
            continue
        if snr_sum < gates.get("min_snr_sum", 0.0):
            continue
        if snr_avg < gates.get("min_snr_avg", 0.0):
            continue
        ys, xs = np.nonzero(sub)
        out.append(dict(cx=sl[1].start + float(xs.mean()),
                        cy=sl[0].start + float(ys.mean()),
                        size=size, area=area, snr_sum=snr_sum, snr_avg=snr_avg))
    return out


def score_detect(events, height, width, cfg, roi, args, gates, bad=None, pool=1):
    """Per-window detections split into hits on the rotor and false positives.

    A detection counts as a hit when its centroid falls inside the rotor
    region - the working point's own "match": "center" rule - and as a false
    positive anywhere else on the sensor. Everything outside the region is
    known not to be the rotor, so false positives are directly countable
    without any ground truth beyond the region itself.
    """
    out = dict(windows=0, hits=0, tp=0, fp=0, hit_rate=0.0, precision=0.0,
               fp_per_window=0.0, snr=0.0, rotor_px=0.0)
    if events is None or not len(events):
        return out
    x0, y0, x1, y1 = roi
    bad_sub = bad
    windows = hits = tp = fp = 0
    snr_best = 0.0
    px_total = 0
    for rotor, scores in ddhf_windows(events, height, width, cfg, pool):
        if bad_sub is not None:
            rotor = rotor & ~bad_sub
            scores = np.where(bad_sub, 0.0, scores)
        windows += 1
        if pool > 1:
            # Back to sensor resolution before the component gates. The
            # pipeline's pixelize mode broadcasts each block's value to every
            # pixel of the block, so its min_rotor_pixels / min_snr_sum are
            # counted in sensor pixels; leaving the mask pooled would make
            # min_rotor_pixels=6 mean 6*pool^2 pixels and min_snr_sum 1/pool^2
            # as easy to reach - neither gate would mean what it was tuned to.
            rotor = np.repeat(np.repeat(rotor, pool, 0), pool, 1)[:height, :width]
            scores = np.repeat(np.repeat(scores, pool, 0), pool, 1)[:height, :width]
        found = blobs(rotor, scores, gates)
        in_roi = [b for b in found
                  if x0 <= b["cx"] < x1 and y0 <= b["cy"] < y1]
        tp += len(in_roi)
        fp += len(found) - len(in_roi)
        if in_roi:
            hits += 1
            snr_best = max(snr_best, max(b["snr_avg"] for b in in_roi))
            px_total += sum(b["size"] for b in in_roi)
    if windows == 0:
        return out
    total = tp + fp
    out.update(windows=windows, hits=hits, tp=tp, fp=fp,
               hit_rate=hits / float(windows),
               precision=(tp / float(total)) if total else 0.0,
               fp_per_window=fp / float(windows),
               snr=snr_best, rotor_px=px_total / float(windows))
    return out


def objective_detect(m, args):
    """Precision, with a floor under the hit rate.

    Mirrors the working point's own objective (precision minus a penalty for
    missing the recall floor): precision alone is trivially maximised by
    detecting almost nothing, and a hit rate alone is maximised by flooding.
    """
    return (m["precision"]
            - args.penalty * max(0.0, args.min_hit - m["hit_rate"]))


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def bounds_of(biases):
    """Per-bias (lo, hi): the range the HAL actually accepts, not the nominal
    0-1800 - anything outside its recommended window is refused (error 102001)."""
    return [biases.limits[n] for n in BIAS_NAMES]


def decode(u, bounds):
    v = {n: int(round(lo + float(np.clip(ui, 0, 1)) * (hi - lo)))
         for n, ui, (lo, hi) in zip(BIAS_NAMES, u, bounds)}
    return repair(v, bounds)


def encode(v, bounds):
    return np.array([(v[n] - lo) / float(hi - lo) if hi > lo else 0.0
                     for n, (lo, hi) in zip(BIAS_NAMES, bounds)])


def repair(v, bounds):
    """Keep bias_diff_off < bias_diff < bias_diff_on; the comparator is
    meaningless otherwise."""
    lo_off, hi_off = bounds[BIAS_NAMES.index("bias_diff_off")]
    lo_on, hi_on = bounds[BIAS_NAMES.index("bias_diff_on")]
    d = v["bias_diff"]
    v["bias_diff_off"] = int(np.clip(min(v["bias_diff_off"], d - 10), lo_off, hi_off))
    v["bias_diff_on"] = int(np.clip(max(v["bias_diff_on"], d + 10), lo_on, hi_on))
    return v


def sample(rng, n, bounds):
    return np.array([encode(decode(u, bounds), bounds)
                     for u in rng.random((n, len(BIAS_NAMES)))])


def propose(X, Y, rng, bounds, xi=0.01, n_cand=4096):
    """Argmax Expected Improvement over random candidates plus jitters around
    the incumbents, using bayes_working_point's GP."""
    model = BWP.gp_fit(np.asarray(X), np.asarray(Y))
    cand = sample(rng, n_cand, bounds)
    top = np.asarray(X)[np.argsort(Y)[-5:]]
    jit = np.clip(np.repeat(top, 200, axis=0)
                  + rng.normal(0, 0.07, (len(top) * 200, len(BIAS_NAMES))), 0.0, 1.0)
    cand = np.vstack([cand, np.array([encode(decode(u, bounds), bounds) for u in jit])])
    mean, sd = BWP.gp_predict(model, cand)
    ei = BWP.expected_improvement(mean, sd, float(np.max(Y)), xi)
    return cand[int(np.argmax(ei))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CSV_FIELDS = (["timestamp", "trial", "phase", "objective", "status", "score",
               "hit_rate",
               "precision", "fp_per_window", "tp", "fp", "snr", "rotor_px",
               "windows", "rate", "pool", "min_events", "roi"] + BIAS_NAMES)


def load_csv(path, bounds, setup):
    """Rows from a previous run, as (X, Y, seen, best).

    Only rows measured under the same detector setup are reusable: a different
    pooling, min_events or ROI changes what the score means, and feeding those
    to the surrogate would have it fit points that were never comparable.
    Mismatched rows are kept in the file and reported, not silently used.
    """
    X, Y, seen = [], [], set()
    best = dict(score=-1e9, values=None, rec=None)
    skipped = 0
    if not os.path.exists(path):
        return X, Y, seen, best, skipped
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                score = float(row["score"])
                values = {n: int(row[n]) for n in BIAS_NAMES}
            except (KeyError, ValueError, TypeError):
                continue
            if (row.get("pool"), row.get("min_events"), row.get("roi"),
                    row.get("objective")) != setup:
                skipped += 1
                continue
            X.append(encode(values, bounds))
            Y.append(score)
            seen.add(tuple(values[n] for n in BIAS_NAMES))
            if score > best["score"]:
                best = dict(score=score, values=values, rec=row)
    return X, Y, seen, best, skipped


def open_csv(path):
    """Append handle plus writer, writing the header only for a new file."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()
    return fh, writer


def _worker_args(args):
    return dict(settle=args.settle, probe=args.probe, flood_rate=args.flood_rate,
                seconds=args.seconds, pool=args.pool, w_snr=args.w_snr,
                w_events=args.w_events, penalty=args.penalty,
                min_hit=args.min_hit)


def locate_from_hits(hits, bad_list, args, width, height, peak, windows):
    """Cluster the child's per-cell hit map into a rotor ROI."""
    hits = np.asarray(hits, np.int32)
    if not hits.any():
        raise SystemExit(
            "the detector flagged nothing at nominal biases. Give --roi "
            "explicitly, lower --min-events, or raise --pool.")
    # Cluster rather than demand per-cell persistence. Measured on this rotor:
    # each cell passed in only 2-5 of 60 windows, yet every top cell sat within
    # a few cells of the others - a marginal rotor lights a different cell of
    # the same small area each window. Requiring per-cell persistence found
    # nothing, and the run could not bootstrap.
    dens = uniform_filter(hits.astype(np.float32), size=5)
    region = dens >= 0.4 * float(dens.max())
    lab, nlab = label(region)
    if nlab:
        top = np.unravel_index(int(np.argmax(dens)), dens.shape)
        region = lab == lab[top]
    ys, xs = np.nonzero(region)
    pad, P = args.pad, args.pool
    roi = (max(0, int(xs.min()) * P - pad), max(0, int(ys.min()) * P - pad),
           min(width, (int(xs.max()) + 1) * P + pad),
           min(height, (int(ys.max()) + 1) * P + pad))
    print("  %d windows, %d cell-hits, densest cluster %d cells, peak comb-SNR %.1f"
          % (windows, int(hits.sum()), int(region.sum()), peak), flush=True)
    if bad_list:
        print("  ignoring %d stuck pixel(s) at pooled cells %s"
              % (len(bad_list), bad_list[:4]), flush=True)
    print("  roi x %d-%d  y %d-%d" % (roi[0], roi[2], roi[1], roi[3]), flush=True)
    return roi


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.csv
    bias_out = os.path.splitext(csv_path)[0] + ".bias"

    preset = dict(ddhf_config._PRESET)
    gates, wp_obj = {}, {}
    if args.working_point:
        wp_preset, gates, wp_obj, wp = load_working_point(args.working_point)
        preset.update(wp_preset)
        print("working point %s (mode %s, match %s)"
              % (os.path.basename(args.working_point), wp.get("mode"),
                 wp.get("match")))
        print("  gates: " + ", ".join("%s=%g" % (k, v)
                                      for k, v in sorted(gates.items())))
        met = wp.get("metrics", {})
        if met:
            print("  it scored precision %.3f / recall %.3f / %.2f fp per frame "
                  "offline" % (met.get("precision", 0), met.get("recall", 0),
                               met.get("fp_per_frame", 0)))
    if args.penalty is None:
        args.penalty = float(wp_obj.get("penalty", 3.0) or 3.0)
    if args.min_hit is None:
        args.min_hit = float(wp_obj.get("min_gtf", 0.8) or 0.8)
    preset.update({k: v for k, v in ddhf_config.cfg_overrides_from_args(args).items()
                   if v is not None})
    if getattr(args, "min_events", None) is None:
        # The preset's 16 suits the dense cached recordings; a live rotor this
        # sparse gives 1-4 events per pixel per window and nothing reaches the
        # NDFT. Measured here: 0 cells pass at 16, 27 at 8.
        preset["min_events"] = 8
    cfg = DDHFConfig(**preset)
    print("detector: window %.1f ms, min_events %d, tau_sf %.2f, tau_comb %.1f, "
          "%d bins, %dx%d pooling" % (cfg.window_ms, cfg.min_events, cfg.tau_sf,
                                      cfg.tau_comb, cfg.frequency_bins,
                                      args.pool, args.pool))
    if args.objective == "detect":
        print("objective: precision on the rotor, penalty %.1f below a hit rate "
              "of %.2f" % (args.penalty, args.min_hit))
    base = dict(args=_worker_args(args), cfg=preset, gates=gates,
                objective=args.objective)

    info = run_child(dict(base, mode="info"), args.trial_timeout)
    if "limits" not in info:
        raise SystemExit("could not query the camera: %s"
                         % (info.get("error") or info.get("status")))
    width, height = info["width"], info["height"]
    bounds = [tuple(info["limits"][n]) for n in BIAS_NAMES]
    print("search space (HAL-accepted ranges):")
    for n, (lo, hi) in zip(BIAS_NAMES, bounds):
        print("  %-14s %d - %d" % (n, lo, hi))

    bad_list = []
    if args.roi:
        roi = tuple(args.roi)
        print("using roi x %d-%d y %d-%d" % (roi[0], roi[2], roi[1], roi[3]))
    else:
        print("locating the rotor at nominal biases ...", flush=True)
        r = run_child(dict(base, mode="locate", values=dict(NOMINAL)),
                      args.trial_timeout)
        if "hits" not in r:
            raise SystemExit("rotor location failed: %s"
                             % (r.get("error") or r.get("status")))
        bad_list = r.get("bad", [])
        roi = locate_from_hits(r["hits"], bad_list, args, width, height,
                               r.get("peak", 0.0), r.get("windows", 0))

    # A score only means something alongside the gates, region and objective
    # that produced it - rows from a different setup are kept but not reused.
    setup = (str(args.pool), str(cfg.min_events), "%d %d %d %d" % roi,
             args.objective)
    X, Y, seen = [], [], set()
    best = dict(score=-1e9, values=None, rec=None)
    if args.reset:
        print("--reset: ignoring the rows already in %s" % csv_path)
    else:
        X, Y, seen, best, skipped = load_csv(csv_path, bounds, setup)
        if Y:
            print("resumed %d trials from %s (best score %.3f)"
                  % (len(Y), csv_path, best["score"]))
        if skipped:
            print("  %d row(s) came from a different setup (pool/min_events/"
                  "roi/objective) and were not reused" % skipped)
    fh, writer = open_csv(csv_path)

    def evaluate(values, trial, phase):
        t0 = time.time()
        r = run_child(dict(base, mode="score", values=values, roi=list(roi),
                           bad=bad_list), args.trial_timeout)
        status = r.get("status", "crash")
        applied = r.get("applied", values)
        m = dict(snr=r.get("snr", 0.0), rotor_px=r.get("rotor_px", 0.0),
                 windows=r.get("windows", 0), measured_px=r.get("measured_px", 0.0),
                 hit_rate=r.get("hit_rate", 0.0), precision=r.get("precision", 0.0),
                 fp_per_window=r.get("fp_per_window", 0.0), tp=r.get("tp", 0),
                 fp=r.get("fp", 0))
        if status == "ok":
            score = (objective_detect(m, args) if args.objective == "detect"
                     else objective(m, args))
        elif status == "flood":
            score = -1.0                     # a real observation: this floods
        else:
            score = -2.0                     # timeout / crash / stall
        rec = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                   trial=trial, phase=phase, status=status, score=round(score, 4),
                   snr=round(m["snr"], 3), rotor_px=round(m["rotor_px"], 2),
                   windows=m["windows"], rate=int(r.get("rate", 0)),
                   hit_rate=round(m["hit_rate"], 4),
                   precision=round(m["precision"], 4),
                   fp_per_window=round(m["fp_per_window"], 3),
                   tp=m["tp"], fp=m["fp"],
                   pool=setup[0], min_events=setup[1], roi=setup[2],
                   objective=setup[3])
        rec.update(applied)
        writer.writerow(rec)
        fh.flush()                      # a kill mid-run loses nothing
        marker = ""
        if status == "ok" and score > best["score"]:
            best.update(score=score, values=dict(applied), rec=rec)
            marker = "  <- best"
        if args.objective == "detect":
            detail = ("hits %4.0f%%  prec %4.0f%%  fp/win %5.2f  (tp %3d fp %4d)"
                      % (100 * m["hit_rate"], 100 * m["precision"],
                         m["fp_per_window"], m["tp"], m["fp"]))
        else:
            detail = "comb-SNR %6.2f  rotor px %5.2f" % (m["snr"], m["rotor_px"])
        print("  %3d/%-3d %-5s score %7.3f  %s  (%2d win, %.2f Mev/s, %.1fs) %s%s"
              % (trial, args.trials, phase, score, detail, m["windows"],
                 r.get("rate", 0) / 1e6, time.time() - t0, status, marker),
              flush=True)
        return score, applied

    trial = len(Y)
    seeds = [encode(dict(NOMINAL), bounds)]
    seeds += list(sample(rng, max(0, args.init - 1), bounds))
    print("\nseeding with %d configurations" % len(seeds), flush=True)
    for u in seeds:
        if trial >= args.trials:
            break
        v = decode(u, bounds)
        if tuple(v[n] for n in BIAS_NAMES) in seen:
            continue
        trial += 1
        score, applied = evaluate(v, trial, "seed")
        Y.append(score); X.append(encode(applied, bounds))
        seen.add(tuple(applied[n] for n in BIAS_NAMES))

    print("\nGP-guided search", flush=True)
    while trial < args.trials:
        trial += 1
        try:
            u = propose(X, Y, rng, bounds)
            phase = "gp"
        except Exception as exc:
            print("    [gp] %s: %s - random step" % (type(exc).__name__, exc))
            u = sample(rng, 1, bounds)[0]
            phase = "rand"
        v = decode(u, bounds)
        score, applied = evaluate(v, trial, phase)
        Y.append(score); X.append(encode(applied, bounds))
        seen.add(tuple(applied[n] for n in BIAS_NAMES))

    fh.close()
    print(chr(10) + "%d trials recorded in %s" % (len(Y), csv_path))
    if best["values"] is None:
        print("no successful trial")
        return 1
    print("best score %.3f  comb-SNR %s  rotor pixels %s"
          % (best["score"], best["rec"]["snr"], best["rec"]["rotor_px"]))
    for n in BIAS_NAMES:
        print("  %-14s %4d   (nominal %4d)" % (n, best["values"][n], NOMINAL[n]))
    studio.write_bias_file(bias_out, best["values"])
    print("wrote %s" % bias_out)
    return 0


if __name__ == "__main__":
    if "--worker-stdin" in sys.argv:
        _payload = json.loads(sys.stdin.read())
        print(json.dumps(_worker(_payload)))
        sys.exit(0)
    sys.exit(main())
