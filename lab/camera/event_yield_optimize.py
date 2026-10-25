#!/usr/bin/env python
"""
Bayesian search of the SilkyEvCam biases for the most events off a fixed fan.

A sibling of bias_optimize.py with a different objective. That one scores
candidates with the DDHF detector itself (comb-SNR x rotor pixels), which is
the right target once the rotor is being detected - but it is useless before
that, because a detector that finds nothing returns zero for every candidate
and the surrogate has a flat surface to search. This one scores raw event
yield off the fan, which is never zero, so it can start from a camera that is
showing you nothing and climb.

    Use this first to get events off the rotor.
    Use bias_optimize.py afterwards to turn those events into detections.

Why the objective is not simply "more events"
    Maximising the event count alone has one global optimum and it is a noise
    flood: bias_fo near its lower bound measured 49 Mev/s on this camera, which
    overruns USB and desynchronises the EVT3 decoder until the device is
    reopened. Every pixel fires, the fan is invisible in the wash, and the
    optimiser is delighted. So the score is the fan's per-pixel event rate
    penalised by the BACKGROUND's per-pixel event rate:

        score = log10(1 + roi_rate_px) - w_bg * log10(1 + bg_rate_px)

    Both are events per pixel per second, so the two regions are comparable
    despite their very different sizes. With the fan fixed in the ROI and the
    rest of the scene static, background events are by definition not signal -
    they are threshold noise, and this term is what stops the search walking
    into the flood. Trials that flood outright are rejected before they are
    measured and scored --penalty, so the surrogate learns the region is bad
    without paying for it.

    Both terms are logarithmic so neither runs away with the objective.

What is reported but not optimised
    Each trial also gets the ROI's dominant frequency and its FFT SNR, by the
    same 250 us binning flicker_check.py uses. These are diagnostics, in the
    CSV and on screen: more events is only progress if the periodicity survives
    them, and a configuration that doubles the yield while halving the peak SNR
    is a step backwards for detection even though it wins on this objective.
    Fold them in with --w-periodic if you would rather optimise both at once.

Getting the ROI
    --roi x0 y0 x1 y1, in sensor pixels, around the fan. Read the coordinates
    off live_detect_viewer.py: left-click the fan and the spectrum window's
    header names the pixel you hit.

Usage
    python event_yield_optimize.py --roi 240 160 400 320
    python event_yield_optimize.py --roi 240 160 400 320 --trials 120
    python event_yield_optimize.py --roi 240 160 400 320 --apply
    python event_yield_optimize.py --roi 240 160 400 320 --w-periodic 0.5
"""
import os
import sys
import csv
import time
import argparse
import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB  = os.path.dirname(_HERE)                       # lab/ (data/, runs/, compare_replay)
for _p in (_HERE, os.path.join(_LAB, "code", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import silky_bias_studio as studio             # DLL/plugin bootstrap + HAL
# Rig owns the camera: it probes each candidate's rate before committing to a
# full capture, rejects floods, and recovers a desynchronised stream by
# reopening the device. All of that was measured against this camera the hard
# way - reuse it rather than writing a second version that has to learn again.
from bias_optimize import (Rig, bounds_of, decode, encode, repair, sample,
                           propose)

BIAS_NAMES = studio.BIAS_NAMES
NOMINAL = studio.NOMINAL

BIN_US = 250                      # 4 kHz sampling -> resolves to 2 kHz
CSV_FIELDS = (["timestamp", "trial", "phase", "status", "score", "roi_rate_px",
               "bg_rate_px", "roi_events", "total_rate", "peak_hz", "peak_snr",
               "seconds", "w_bg", "w_periodic", "roi"] + BIAS_NAMES)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bayesian search of the biases for event yield off a fan.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roi", type=int, nargs=4, required=True,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="the fan's region in sensor pixels, x0 y0 x1 y1")
    p.add_argument("--trials", type=int, default=60,
                   help="candidates to measure in total (default 60)")
    p.add_argument("--init", type=int, default=12,
                   help="random trials before the surrogate takes over "
                        "(default 12). Too few and the GP extrapolates from a "
                        "corner it happened to land in")
    p.add_argument("--seconds", type=float, default=0.6,
                   help="capture length per trial (default 0.6)")
    p.add_argument("--settle", type=float, default=0.20,
                   help="data discarded after a bias change, seconds. The "
                        "analog front end needs time; measuring through the "
                        "transient scores the previous candidate (default 0.2)")
    p.add_argument("--probe", type=float, default=0.15,
                   help="short rate probe before the full capture, seconds; a "
                        "flooding candidate is rejected on this alone")
    p.add_argument("--flood-rate", type=float, default=20e6,
                   help="reject a candidate above this many events/s (default "
                        "20e6). Above ~40 Mev/s the decoder desynchronises and "
                        "the camera has to be reopened, which costs seconds")
    p.add_argument("--w-bg", type=float, default=1.0,
                   help="weight of the background-rate penalty (default 1.0). "
                        "0 maximises raw yield and WILL find the noise flood")
    p.add_argument("--w-periodic", type=float, default=0.0,
                   help="weight of the ROI's FFT peak SNR in the objective "
                        "(default 0 = reported only, not optimised)")
    p.add_argument("--penalty", type=float, default=-5.0,
                   help="score given to a flooded or stalled trial, so the "
                        "surrogate avoids that region (default -5)")
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=2000.0)
    p.add_argument("--csv", default="event_yield.csv",
                   help="trial log; an existing file is resumed from")
    p.add_argument("--out", default="event_yield.bias",
                   help="where the best biases are written")
    p.add_argument("--apply", action="store_true",
                   help="leave the best biases on the camera at the end "
                        "instead of restoring the ones it started with")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def periodicity(ev, fmin, fmax):
    """(peak_hz, peak_snr) of the ROI's aggregate event rate.

    The whole ROI is binned into one 4 kHz time series rather than analysed per
    pixel: this is a health check on the fan's modulation, not a detector, and
    summing over the region is both far cheaper and far less noisy than any
    per-pixel spectrum would be at these capture lengths.
    """
    if ev is None or len(ev) < 64:
        return 0.0, 0.0
    t = ev["t"].astype(np.int64)
    nb = int((t[-1] - t[0]) // BIN_US) + 1
    if nb < 64:
        return 0.0, 0.0
    hist = np.bincount((t - t[0]) // BIN_US, minlength=nb).astype(np.float64)
    mag = np.abs(np.fft.rfft((hist - hist.mean()) * np.hanning(nb)))
    freq = np.fft.rfftfreq(nb, BIN_US / 1e6)
    band = (freq >= fmin) & (freq <= fmax)
    if not band.any():
        return 0.0, 0.0
    m, f = mag[band], freq[band]
    k = int(np.argmax(m))
    return float(f[k]), float(m[k] / (np.median(m) + 1e-9))


def measure(rig, values, args, roi_px, total_px):
    """Apply one candidate and report what came off the sensor."""
    for name in rig.biases.names:
        if name in values:
            rig.biases.set(name, values[name])

    def restore():
        for name in rig.biases.names:            # a reopen resets to factory
            if name in values:
                rig.biases.set(name, values[name])

    ev, status = rig.capture(args.seconds, tuple(args.roi), on_reopen=restore)
    out = dict(status=status, roi_events=0, roi_rate_px=0.0, bg_rate_px=0.0,
               total_rate=float(rig.last_rate), peak_hz=0.0, peak_snr=0.0)
    if status != "ok" or ev is None or not len(ev):
        return out
    t = ev["t"].astype(np.int64)
    span = max((t[-1] - t[0]) / 1e6, 1e-6)
    roi_rate = len(ev) / span
    # rig.last_rate covers the WHOLE sensor - the clip happens after counting -
    # so the background rate is what is left once the ROI is taken out.
    bg_rate = max(0.0, float(rig.last_rate) - roi_rate)
    bg_px = max(1, total_px - roi_px)
    out.update(roi_events=int(len(ev)),
               roi_rate_px=roi_rate / roi_px,
               bg_rate_px=bg_rate / bg_px)
    out["peak_hz"], out["peak_snr"] = periodicity(ev, args.fmin, args.fmax)
    return out


def objective(m, args):
    if m["status"] != "ok":
        return args.penalty
    score = (np.log10(1.0 + m["roi_rate_px"])
             - args.w_bg * np.log10(1.0 + m["bg_rate_px"]))
    if args.w_periodic:
        score += args.w_periodic * np.log10(1.0 + m["peak_snr"])
    return float(score)


# ---------------------------------------------------------------------------
# Trial log
# ---------------------------------------------------------------------------
def setup_key(args):
    """What makes two rows comparable. A different ROI, capture length or
    weighting changes what the score means, so rows from another setup must not
    be fed to the surrogate as if they were the same measurement."""
    return ("%d %d %d %d" % tuple(args.roi), "%.3f" % args.seconds,
            "%.3f" % args.w_bg, "%.3f" % args.w_periodic)


def load_csv(path, bounds, key):
    X, Y, seen, skipped = [], [], set(), 0
    best = dict(score=-1e9, values=None)
    if not os.path.exists(path):
        return X, Y, seen, best, skipped
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                score = float(row["score"])
                values = {n: int(row[n]) for n in BIAS_NAMES}
            except (KeyError, ValueError, TypeError):
                continue
            if (row.get("roi"), row.get("seconds"), row.get("w_bg"),
                    row.get("w_periodic")) != key:
                skipped += 1
                continue
            X.append(encode(values, bounds))
            Y.append(score)
            seen.add(tuple(values[n] for n in BIAS_NAMES))
            if score > best["score"]:
                best = dict(score=score, values=values)
    return X, Y, seen, best, skipped


def open_csv(path):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    x0, y0, x1, y1 = args.roi
    if x1 <= x0 or y1 <= y0:
        print("error: --roi must be x0 y0 x1 y1 with x1>x0 and y1>y0",
              file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    rig = Rig(args)
    total_px = rig.width * rig.height
    if not (0 <= x0 < x1 <= rig.width and 0 <= y0 < y1 <= rig.height):
        print("error: --roi lies outside the %dx%d sensor"
              % (rig.width, rig.height), file=sys.stderr)
        return 1
    roi_px = (x1 - x0) * (y1 - y0)
    startup = dict(rig.biases.values)
    bounds = bounds_of(rig.biases)
    key = setup_key(args)

    X, Y, seen, best, skipped = load_csv(args.csv, bounds, key)
    print("sensor %dx%d, ROI %d x %d = %d px (%.1f%% of the sensor)"
          % (rig.width, rig.height, x1 - x0, y1 - y0, roi_px,
             100.0 * roi_px / total_px))
    if X:
        print("resuming from %s: %d comparable trials, best score %.4f"
              % (args.csv, len(X), best["score"]))
    if skipped:
        print("  (%d rows in the file came from a different setup and were "
              "left alone)" % skipped)

    fh, writer = open_csv(args.csv)
    # Nominal goes first so there is always a reference the rest can be read
    # against - "better than the camera's defaults" is the claim that matters.
    queue = [dict(NOMINAL)]
    todo = max(0, args.trials - len(X))
    try:
        for trial in range(todo):
            if queue:
                values, phase = queue.pop(0), "seed"
            elif len(X) < args.init:
                values = decode(sample(rng, 1, bounds)[0], bounds)
                phase = "random"
            else:
                values = decode(propose(X, Y, rng, bounds), bounds)
                phase = "bayes"
            values = repair(values, bounds)
            fingerprint = tuple(values[n] for n in BIAS_NAMES)
            if fingerprint in seen and phase != "seed":
                # The surrogate re-proposing its incumbent means EI has
                # collapsed there; jitter rather than spend a capture on a
                # point already measured.
                values = decode(np.clip(encode(values, bounds)
                                        + rng.normal(0, 0.05, len(BIAS_NAMES)),
                                        0, 1), bounds)
                fingerprint = tuple(values[n] for n in BIAS_NAMES)
            seen.add(fingerprint)

            m = measure(rig, values, args, roi_px, total_px)
            score = objective(m, args)
            X.append(encode(values, bounds))
            Y.append(score)
            row = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                       trial=len(X), phase=phase, score="%.6f" % score,
                       roi_rate_px="%.3f" % m["roi_rate_px"],
                       bg_rate_px="%.3f" % m["bg_rate_px"],
                       roi_events=m["roi_events"],
                       total_rate="%.0f" % m["total_rate"],
                       peak_hz="%.1f" % m["peak_hz"],
                       peak_snr="%.2f" % m["peak_snr"],
                       seconds="%.3f" % args.seconds, w_bg="%.3f" % args.w_bg,
                       w_periodic="%.3f" % args.w_periodic,
                       roi="%d %d %d %d" % tuple(args.roi),
                       status=m["status"], **values)
            writer.writerow(row)
            fh.flush()
            flag = " *" if score > best["score"] else ""
            if score > best["score"]:
                best = dict(score=score, values=dict(values))
            print("[%3d/%3d] %-6s %-5s score %7.4f  roi %8.1f ev/px/s  "
                  "bg %6.1f  peak %6.1f Hz snr %5.1f  total %5.2f Mev/s%s"
                  % (len(X), args.trials, phase, m["status"], score,
                     m["roi_rate_px"], m["bg_rate_px"], m["peak_hz"],
                     m["peak_snr"], m["total_rate"] / 1e6, flag), flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - the trials so far are in %s" % args.csv)
    finally:
        fh.close()

    if best["values"] is None:
        print("no trial produced a score", file=sys.stderr)
        return 1
    print("\nbest score %.4f" % best["score"])
    for name in BIAS_NAMES:
        print("  %-14s %4d   (nominal %4d)"
              % (name, best["values"][name], NOMINAL.get(name, 0)))
    studio.write_bias_file(args.out, best["values"])
    print("wrote %s" % args.out)

    final = best["values"] if args.apply else startup
    for name in rig.biases.names:
        if name in final:
            rig.biases.set(name, final[name])
    print("camera left at %s biases"
          % ("the best" if args.apply else "the ones it started with"))
    print("\nnext: check the fan is still periodic at these biases -\n"
          "  python live_detect_viewer.py --bias %s     (click the fan)\n"
          "  python flicker_check.py --bias %s\n"
          "then hand them to bias_optimize.py to optimise for detection."
          % (args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
