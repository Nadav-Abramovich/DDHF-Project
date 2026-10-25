#!/usr/bin/env python
"""
Find and mask the sensor's hot pixels.

Most of what this finds is NOT a broken pixel - read this before masking
    A single pixel running at 50-120 kev/s looks like a defect and usually
    isn't. Measured on serial 00000341: the eight fastest pixels in
    rec_20260909_185230.raw (x=332 y=48 at 124,000 ev/s down to x=81 y=92 at
    27,000) all read EXACTLY 0 ev/s with the lens capped and that recording's
    own biases loaded. They are light-driven - bright points in the scene
    driving single pixels into continuous firing - not silicon.

    Nor are they flickering lamps: per-pixel signed spectra of all eight (20 us
    bins, out to 25 kHz) are broadband 1/f peaking at 20-40 Hz with no sharp
    line and no harmonic comb, which is what a pixel parked on a specular
    highlight in a dim scene does as it fires at its own bandwidth limit on
    shot noise.

    So their coordinates belong to the SCENE and move when the camera moves. A
    .hotpx saved from one session is not reusable in the next, and masking
    those pixels is masking scene content. Only a list built with the lens
    capped is a property of the sensor.

    The one genuine defect on this camera is x=20 y=420: 53,200 ev/s capped,
    reproducible across runs, at near-nominal biases. bias_refr 1415 silences
    it outright (a longer refractory dead time than its ~19 us period), which
    is why it does not appear in rec_20260909_185230.raw at all.

What masking buys you, which is less than it looks
    It does NOT reveal a weak rotor. These are isolated coordinates; a
    per-pixel or per-block spectrum computed anywhere else is untouched by
    them, and at ~1.3 Mev/s nothing is saturating the readout either. The
    ordinary-pixel background in these recordings is quiet - median 0.5 ev/s,
    p99 12 ev/s - so there is no noise floor being raised.

    What it does buy: any whole-frame or large-ROI spectrum stops being a
    measurement of a few runaway pixels, and the Mev/s readout starts meaning
    something. That is a diagnostics fix, not a detection fix.

This camera masks in software only
    Probed on serial 00000341: I_DigitalEventMask is None, and I_RoiPixelMask
    and I_EventTrailFilterModule are both absent. The CenturyArks Gen3.1 plugin
    exposes no in-sensor pixel masking at all, so apply_hardware() below finds
    nothing to program and Mask is the whole story. I_ROI is not a substitute -
    it works in whole rows and columns, so masking x=332 y=48 would cost 1,119
    pixels to remove one.

Why isolation, not rate, is the test
    Thresholding on rate alone cannot separate a hot pixel from a bright rotor
    pixel - FAN_RECORDING has ordinary pixels at 264 ev/s. But a rotor lights up
    a *patch*: its neighbours are busy too. A hot pixel sits alone in a quiet
    neighbourhood. So a pixel is hot when it is both fast in absolute terms and
    far faster than the busiest of its eight neighbours.

Usage
    python hot_pixels.py --probe                               # what the sensor offers
    python hot_pixels.py --scan rec_20260909_185230.raw        # list them
    python hot_pixels.py --scan rec_20260909_185230.raw -o cam.hotpx
    # then in your own code:
    #     from hot_pixels import Mask
    #     mask = Mask.load("cam.hotpx")
    #     e = mask(e)

Scanning the live camera: cap the lens first. With no scene there is nothing
but the sensor, so every pixel still firing is a real defect and the isolation
test has nothing to argue with.
"""
import os
import sys
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import silky_bias_studio as studio             # DLL/plugin bootstrap + HAL
from metavision_core.event_io import EventsIterator


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def rate_map(source, seconds=5.0, skip=1.0, serial=""):
    """Per-pixel event rate (ev/s) over `seconds` of a RAW or the live camera.

    `skip` drops the opening burst: the stream starts with whatever backlog the
    sensor had queued, which would read as every pixel being hot at once.
    """
    device = studio.open_device(source, serial, False)
    width, height = studio.describe_device(device)
    acc = np.zeros((height, width), np.int64)
    it = EventsIterator.from_device(device=device, mode="delta_t", delta_t=20000)
    t0 = None
    for e in it:
        if not len(e):
            continue
        if t0 is None:
            t0 = int(e["t"][0])
        rel = (int(e["t"][0]) - t0) / 1e6
        if rel < skip:
            continue
        if rel > skip + seconds:
            break
        np.add.at(acc, (e["y"].astype(np.int64), e["x"].astype(np.int64)), 1)
    return acc / float(seconds)


def find(rates, floor_hz=500.0, isolation=8.0):
    """Coordinates of the hot pixels in a rate map, hottest first.

    A pixel qualifies when it clears `floor_hz` AND beats the busiest of its
    eight neighbours by `isolation`x. The second test is what keeps a rotor -
    whose pixels are fast *together* - out of the list.

    What it cannot do is tell a broken pixel from a bright point source, since
    both are fast and both are alone. Only capping the lens separates those,
    so treat a list built from a scene as candidates, not as defects.
    """
    r = np.asarray(rates, np.float64)
    nb = np.zeros_like(r)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.maximum(nb, np.roll(np.roll(r, dy, 0), dx, 1))
    hot = (r >= floor_hz) & (r >= isolation * np.maximum(nb, 1.0))
    ys, xs = np.nonzero(hot)
    order = np.argsort(r[ys, xs])[::-1]
    return [(int(xs[i]), int(ys[i])) for i in order]


# ---------------------------------------------------------------------------
# Applying the mask
# ---------------------------------------------------------------------------
class Mask:
    """Drops the listed pixels from an event array.

    Held as a boolean image rather than a coordinate list so filtering is one
    fancy-index per slice - the per-slice cost has to stay negligible next to
    the detector or the live viewer loses frames to it.
    """

    def __init__(self, pixels, width=640, height=480):
        self.pixels = list(pixels)
        self.bad = np.zeros((height, width), bool)
        for x, y in self.pixels:
            if 0 <= x < width and 0 <= y < height:
                self.bad[y, x] = True

    def __call__(self, e):
        if not len(e) or not self.pixels:
            return e
        return e[~self.bad[e["y"], e["x"]]]

    def save(self, path):
        with open(path, "w") as fh:
            fh.write("# hot pixels: x y\n")
            for x, y in self.pixels:
                fh.write("%d %d\n" % (x, y))

    @classmethod
    def load(cls, path, width=640, height=480):
        px = []
        with open(path) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if line:
                    x, y = line.split()
                    px.append((int(x), int(y)))
        return cls(px, width, height)


def probe(device):
    """Report what this sensor offers for masking, and how much of it.

    The slot count is the number that decides the strategy - it is a small
    fixed budget in silicon, and anything past it has to fall back to the
    software Mask. It cannot be read from a RAW file, only from the camera.
    """
    out = {}
    dem = device.get_i_digital_event_mask()
    out["digital_event_mask"] = len(dem.get_pixel_masks()) if dem else 0
    for name, getter in (("roi_pixel_mask", "get_i_roi_pixel_mask"),
                         ("event_trail_filter", "get_i_event_trail_filter_module"),
                         ("erc", "get_i_erc_module"),
                         ("event_rate_activity_filter", "get_i_event_rate")):
        try:
            out[name] = getattr(device, getter)() is not None
        except Exception:
            out[name] = False
    return out


def apply_hardware(device, pixels):
    """Mask the pixels in the sensor itself, so they never reach the host.

    Kept for a camera that supports it - the events would die before readout,
    costing no bandwidth and no CPU - but THIS camera does not: the SilkyEvCam
    Gen3.1 returns no I_DigitalEventMask, so this returns 0 and every caller
    must software-mask the whole list. Check with --probe before relying on it.
    """
    facility = device.get_i_digital_event_mask()
    if facility is None:
        return 0
    slots = facility.get_pixel_masks()
    used = 0
    for slot, (x, y) in zip(slots, pixels):
        slot.set_mask(x, y, True)
        used += 1
    return used


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Find the sensor's hot pixels.")
    p.add_argument("--scan", default="",
                   help="a RAW to measure; omit to measure the live camera")
    p.add_argument("-o", "--out", default="",
                   help="write the list to this .hotpx file")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--skip", type=float, default=1.0)
    p.add_argument("--floor", type=float, default=500.0,
                   help="minimum ev/s to be considered at all (default 500)")
    p.add_argument("--isolation", type=float, default=8.0,
                   help="times faster than the busiest neighbour (default 8)")
    p.add_argument("--probe", action="store_true",
                   help="report the live camera's masking facilities and exit")
    p.add_argument("--serial", default="")
    a = p.parse_args(argv)

    if a.probe:
        device = studio.open_device("", a.serial, False)
        studio.describe_device(device)
        caps = probe(device)
        print("hardware mask slots : %d" % caps.pop("digital_event_mask"))
        for name, present in caps.items():
            print("%-20s: %s" % (name, "present" if present else "absent"))
        return 0

    rates = rate_map(a.scan, a.seconds, a.skip, a.serial)
    pixels = find(rates, a.floor, a.isolation)
    total = rates.sum()
    print("%d hot pixel(s) in %s" % (len(pixels), a.scan or "the live camera"))
    share = 0.0
    for x, y in pixels:
        share += rates[y, x]
        print("   x=%3d y=%3d  %9.0f ev/s  (%.2f%% of the stream)"
              % (x, y, rates[y, x], 100 * rates[y, x] / max(total, 1.0)))
    if pixels:
        print("   combined: %.1f%% of the stream" % (100 * share / max(total, 1.0)))
    if a.out:
        Mask(pixels, rates.shape[1], rates.shape[0]).save(a.out)
        print("written to %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
