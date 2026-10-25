#!/usr/bin/env python
"""
Is the periodic signal the rotor, or the lamp lighting it?

A flickering light source (LED PWM, or mains ripple at 100/120 Hz) modulates
EVERYTHING it illuminates, so the comb detector reports it as happily as it
would a rotor - and a light aimed at a fan illuminates the fan most brightly,
which is exactly where you would expect a real rotor signal. The two are told
apart by where the frequency lives, not by how strong it is:

    a rotor    modulates only the swept disc - one compact region
    a lamp     modulates every lit surface, including static background

So this cuts the sensor into blocks, finds each block's dominant frequency, and
reports how widely the strongest one is shared. A frequency present across
widely separated blocks - especially over static scenery - is illumination. One
confined to a compact patch is a rotor.

It also flags a peak that is a harmonic of mains ripple (100 Hz on 50 Hz mains,
120 Hz on 60 Hz), which is a lamp beyond reasonable doubt.

Two manual checks settle any remaining doubt, and take seconds:
    stop the fan, leave the light on   - signal persists => it is the light
    change the fan speed               - frequency moves => it is the rotor

Usage
    python flicker_check.py
    python flicker_check.py --seconds 5 --bias wp_detect.bias
"""
import os
import sys
import datetime
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import silky_bias_studio as studio
from metavision_core.event_io import EventsIterator

BLK = 32
BIN_US = 250                      # 4 kHz sampling -> resolves up to 2 kHz


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tell a rotor apart from a flickering light source.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=3.0,
                   help="capture length; longer gives finer frequency "
                        "resolution (default 3)")
    p.add_argument("--bias", default="", help=".bias file to apply first")
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=2000.0)
    p.add_argument("--top", type=int, default=12,
                   help="how many blocks to list (default 12)")
    p.add_argument("--record", nargs="?", const="auto", default=None,
                   metavar="PATH",
                   help="save the analysed capture as a Metavision RAW plus a "
                        ".bias sidecar of the biases that produced it, so the "
                        "verdict can be re-checked offline. Bare --record "
                        "auto-names it flicker_<timestamp>.raw here")
    return p.parse_args(argv)


def capture(device, seconds, record=None, biases=None):
    it = EventsIterator.from_device(device=device, mode="delta_t", delta_t=10000)
    stream = iter(it)
    for _ in range(20):                       # drain the start-up backlog
        next(stream)
    if record:
        # Start logging only after the drain, so the file holds the window that
        # was actually analysed rather than the backlog in front of it.
        stream_facility = device.get_i_events_stream()
        stream_facility.log_raw_data(record)
        if biases is not None and biases.live:
            studio.write_bias_file(os.path.splitext(record)[0] + ".bias",
                                   biases.values)
    chunks, t0 = [], None
    while True:
        e = next(stream)
        if len(e):
            if t0 is None:
                t0 = int(e["t"][0])
            chunks.append(e.copy())
            if (int(e["t"][-1]) - t0) / 1e6 >= seconds:
                break
    if record:
        device.get_i_events_stream().stop_log_raw_data()
        print("recorded %s (%.1f MB) + %s"
              % (record, os.path.getsize(record) / 1e6,
                 os.path.basename(os.path.splitext(record)[0] + ".bias")))
    return np.concatenate(chunks)


def main(argv=None):
    args = parse_args(argv)
    device = studio.open_device("", "", False)
    width, height = studio.describe_device(device)
    biases = studio.BiasControl(device)
    if args.bias and os.path.exists(args.bias):
        biases.apply(studio.read_bias_file(args.bias), os.path.basename(args.bias))

    record = args.record
    if record == "auto":
        record = os.path.join(os.path.dirname(_HERE), "flicker_%s.raw"
                              % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if record:
        folder = os.path.dirname(os.path.abspath(record))
        if folder:
            os.makedirs(folder, exist_ok=True)
    ev = capture(device, args.seconds, record, biases)
    t = ev["t"].astype(np.int64)
    dur = (t[-1] - t[0]) / 1e6
    nx, ny = width // BLK, height // BLK
    nb = int((t[-1] - t[0]) // BIN_US) + 1
    bx = (ev["x"].astype(np.int64) // BLK).clip(0, nx - 1)
    by = (ev["y"].astype(np.int64) // BLK).clip(0, ny - 1)
    idx = (by * nx + bx) * nb + ((t - t[0]) // BIN_US)
    hist = np.bincount(idx, minlength=nx * ny * nb)[:nx * ny * nb]
    hist = hist.reshape(nx * ny, nb).astype(np.float32)
    print("%.2f s, %d events, %.2f Mev/s, %d x %d blocks"
          % (dur, len(t), len(t) / dur / 1e6, nx, ny))

    active = hist.sum(1) >= 100
    if not active.any():
        print("no active blocks - nothing to analyse")
        return 1
    rows = hist[active] - hist[active].mean(1, keepdims=True)
    mag = np.abs(np.fft.rfft(rows * np.hanning(nb), axis=1))
    freq = np.fft.rfftfreq(nb, BIN_US / 1e6)
    band = (freq >= args.fmin) & (freq <= args.fmax)
    m, f = mag[:, band], freq[band]
    pk = np.argmax(m, axis=1)
    snr = m[np.arange(len(m)), pk] / (np.median(m, axis=1) + 1e-9)
    ids = np.flatnonzero(active)
    peak = f[pk]

    order = np.argsort(snr)[::-1]
    print("\nstrongest periodic blocks:")
    for o in order[:args.top]:
        bid = ids[o]
        print("   x=%3d-%3d y=%3d-%3d  %7.1f Hz  SNR %6.1f  %6d ev"
              % ((bid % nx) * BLK, (bid % nx) * BLK + BLK - 1,
                 (bid // nx) * BLK, (bid // nx) * BLK + BLK - 1,
                 peak[o], snr[o], int(hist[bid].sum())))

    top = peak[order[0]]
    shared = (np.abs(peak - top) <= top * 0.02) & (snr >= 5.0)
    print("\ndominant %.1f Hz: %d of %d active blocks share it (SNR>=5)"
          % (top, int(shared.sum()), int(active.sum())))
    verdict = "inconclusive"
    if shared.any():
        xs = (ids[shared] % nx) * BLK
        ys = (ids[shared] // nx) * BLK
        spread_x = (xs.max() + BLK - xs.min()) / float(width)
        spread_y = (ys.max() + BLK - ys.min()) / float(height)
        frac = shared.sum() / float(active.sum())
        print("   spanning x %d-%d y %d-%d  = %.0f%% of width, %.0f%% of height"
              % (xs.min(), xs.max() + BLK - 1, ys.min(), ys.max() + BLK - 1,
                 100 * spread_x, 100 * spread_y))
        if frac > 0.25 or (spread_x > 0.6 and spread_y > 0.6):
            verdict = ("LIGHT - the frequency is spread over the scene, which "
                       "is what illumination does, not a rotor")
        elif frac < 0.10 and spread_x < 0.4 and spread_y < 0.4:
            verdict = "ROTOR - the frequency is confined to one compact patch"
    for label, base in (("50 Hz mains", 100.0), ("60 Hz mains", 120.0)):
        k = top / base
        if abs(k - round(k)) < 0.03 and round(k) >= 1:
            verdict = "LIGHT - %.1f Hz is %dx the %s ripple" % (top, round(k), label)
    print("\nverdict: %s" % verdict)
    print("confirm by stopping the fan with the light still on (signal stays "
          "=> light), or changing the fan speed (frequency moves => rotor).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
