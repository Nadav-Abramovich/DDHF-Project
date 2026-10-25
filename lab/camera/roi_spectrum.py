#!/usr/bin/env python
"""
Full spectral peak list for one region - signed and unsigned side by side.

Why this exists
    flicker_check.py reports ONE frequency per block, the argmax of its
    spectrum. That is the right summary for "is this a lamp or a rotor", but it
    cannot answer "is the peak I am looking at the fundamental or a harmonic":
    while the strongest line dominates every block, no weaker line - the shaft
    rate, a subharmonic - ever appears in the listing at all, at any --top or
    --fmin. This lists the top N peaks of one region instead, so the whole
    harmonic structure is visible at once.

The signed vs unsigned distinction, which is the point
    A blade passing a pixel makes TWO intensity transitions: the leading edge
    arrives (one polarity) and the trailing edge leaves (the other). Count how
    many events arrive per time bin, ignoring polarity, and you get two bursts
    per blade - so the strongest line sits at 2x the blade-pass rate. That is
    what flicker_check measures, and it is why a fan whose blade-pass is f can
    report 2f with a huge SNR and no obvious sign that anything is wrong.

    Sum the POLARITY per bin instead and the two bursts have opposite signs, so
    they no longer merge into one rate: the fundamental comes back at the true
    blade-pass. This is also what the DDHF detector actually transforms - its
    NDFT is over _normalise_polarity's +-1, not over event counts - so the
    signed column is the one that predicts what the detector will see.

    unsigned peak = 2x signed peak  =>  the signed one is the blade-pass.

With --blades the implied shaft rate is checked too: blade-pass must be
exactly blades x shaft, and finding that line confirms the reading outright.

Usage
    python roi_spectrum.py -i rec_low.raw --roi 416 64 640 224 --blades 7
    python roi_spectrum.py --roi 416 64 640 224 --blades 7 --seconds 5
    python roi_spectrum.py -i rec_high.raw --roi 416 64 640 224 --mains 50
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Top spectral peaks of a region, signed and unsigned.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="",
                   help="a Metavision RAW to analyse; omit to capture live")
    p.add_argument("--roi", type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="region to analyse (default: the whole sensor). Use "
                        "the patch flicker_check.py put the strongest blocks "
                        "in - a tight ROI is what keeps the rest of the scene "
                        "out of the spectrum")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="length to analyse; the frequency resolution is 1/this "
                        "(default 5 => 0.2 Hz)")
    p.add_argument("--bin-us", type=float, default=200.0,
                   help="time bin in microseconds (default 200 = 5 kHz "
                        "sampling, so frequencies up to 2500 Hz)")
    p.add_argument("--fmin", type=float, default=5.0)
    p.add_argument("--fmax", type=float, default=1000.0)
    p.add_argument("--top", type=int, default=10, help="peaks to list")
    p.add_argument("--blades", type=int, default=0,
                   help="blade count; checks blade-pass = blades x shaft")
    p.add_argument("--mains", type=float, default=50.0,
                   help="mains frequency, to report lamp ripple at 2x it and "
                        "its harmonics (default 50; 0 to skip)")
    p.add_argument("--bias", default="", help=".bias file to apply first (live)")
    p.add_argument("--serial", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def accumulate(args):
    """Bin the ROI's events into a count series and a signed-polarity series.

    Accumulated incrementally rather than by concatenating the events: a busy
    scene is ~20 Mev/s, so five seconds is 100M events and ~800 MB if held. The
    histograms are a few tens of thousands of bins whatever the rate.
    """
    device = studio.open_device(args.input, args.serial, False)
    width, height = studio.describe_device(device)
    if args.bias and os.path.exists(args.bias) and not args.input:
        biases = studio.BiasControl(device)
        biases.apply(studio.read_bias_file(args.bias),
                     os.path.basename(args.bias))
    roi = args.roi or (0, 0, width, height)
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise SystemExit("--roi lies outside the %dx%d sensor" % (width, height))

    nb = int(args.seconds * 1e6 / args.bin_us) + 1
    cnt = np.zeros(nb, np.float64)
    sgn = np.zeros(nb, np.float64)
    it = EventsIterator.from_device(device=device, mode="delta_t", delta_t=20000)
    stream = iter(it)
    if not args.input:
        for _ in range(20):               # drain the start-up backlog
            next(stream)
    t0, t_last, total, kept = None, None, 0, 0
    while True:
        try:
            e = next(stream)
        except StopIteration:
            break
        if not len(e):
            continue
        total += len(e)
        if t0 is None:
            t0 = int(e["t"][0])
        t_last = int(e["t"][-1])
        m = ((e["x"] >= x0) & (e["x"] < x1) & (e["y"] >= y0) & (e["y"] < y1))
        if m.any():
            sub = e[m]
            idx = ((sub["t"].astype(np.int64) - t0) // int(args.bin_us))
            ok = (idx >= 0) & (idx < nb)
            idx = idx[ok]
            if len(idx):
                kept += len(idx)
                cnt += np.bincount(idx, minlength=nb)
                # +1 for ON, -1 for OFF: the two edges of one blade passage
                # cancel into a single cycle instead of counting as two
                w = np.where(sub["p"][ok] == 1, 1.0, -1.0)
                sgn += np.bincount(idx, weights=w, minlength=nb)
        if (t_last - t0) / 1e6 >= args.seconds:
            break
    dur = max((t_last - t0) / 1e6, 1e-6) if t0 is not None else 0.0
    return cnt, sgn, dict(width=width, height=height, roi=roi, dur=dur,
                          total=total, kept=kept,
                          roi_px=(x1 - x0) * (y1 - y0))


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------
def spectrum(series, bin_us, fmin, fmax):
    nb = series.size
    mag = np.abs(np.fft.rfft((series - series.mean()) * np.hanning(nb)))
    freq = np.fft.rfftfreq(nb, bin_us / 1e6)
    band = (freq >= fmin) & (freq <= fmax)
    return freq[band], mag[band]


def peaks(freq, mag, top, min_sep_hz):
    """Local maxima by SNR against the median, thinned by a separation.

    Without the separation one broad peak fills the whole list with its own
    shoulders and the harmonics never appear - which is the failure this script
    exists to avoid.
    """
    med = float(np.median(mag)) + 1e-12
    inner = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
    if not inner.size:
        return []
    order = inner[np.argsort(mag[inner])[::-1]]
    out = []
    for i in order:
        if all(abs(freq[i] - f) >= min_sep_hz for f, _ in out):
            out.append((float(freq[i]), float(mag[i] / med)))
        if len(out) >= top:
            break
    return out


def amp_at(freq, mag, hz, tol_hz):
    """Peak SNR within +-tol of hz, or 0 - for testing a specific hypothesis."""
    if hz <= 0:
        return 0.0
    med = float(np.median(mag)) + 1e-12
    sel = np.abs(freq - hz) <= tol_hz
    return float(mag[sel].max() / med) if sel.any() else 0.0


def ladder(pk, res):
    """The fundamental that explains the most listed peaks as its harmonics.

    Rotating machinery puts energy at every multiple of the shaft rate, so the
    lines that matter are a comb, not a single tone - and the loudest line is
    usually NOT the fundamental (on a fan it is the blade-pass, the Nth
    harmonic for N blades). Reading the strongest peak alone therefore gets the
    shaft rate wrong by a factor of N, and its 2x harmonic wrong by 2N.

    Candidates are the peaks themselves plus every pairwise difference, since
    the spacing between adjacent harmonics IS the fundamental even when the
    fundamental itself is too weak to have made the list. Ties go to the larger
    candidate: a small enough number divides everything and explains nothing.
    """
    if len(pk) < 3:
        return None
    freqs = np.array([f for f, _ in pk])
    cands = list(freqs) + [abs(a - b) for a in freqs for b in freqs
                           if abs(a - b) > max(1.0, 4 * res)]
    best = None
    for f0 in sorted(set(round(c, 3) for c in cands)):
        if f0 < max(1.0, 4 * res):
            continue
        k = np.round(freqs / f0)
        tol = np.maximum(3 * res, 0.006 * freqs)
        ok = (k >= 1) & (np.abs(freqs - k * f0) <= tol)
        n = int(ok.sum())
        if n >= 3 and (best is None or n > best[1]
                       or (n == best[1] and f0 > best[0])):
            best = (f0, n, k, ok)
    return best


def report(label, freq, mag, args, res):
    print("\n%s" % label)
    pk = peaks(freq, mag, args.top, max(2.0, 4 * res))
    if not pk:
        print("   no peaks in %g-%g Hz" % (args.fmin, args.fmax))
        return pk
    f0 = pk[0][0]
    print("   %-9s %-9s %-8s %s" % ("freq", "SNR", "/ f0", "note"))
    for f, snr in pk:
        ratio = f / f0
        note = ""
        for k in (2, 3, 4, 5, 6, 7):
            if abs(ratio - k) < 0.02:
                note = "%dx strongest" % k
            elif abs(ratio - 1.0 / k) < 0.02 / k:
                note = "1/%d of strongest" % k
        print("   %8.2f  %8.1f  %7.3f  %s" % (f, snr, ratio, note))
    return pk


def main(argv=None):
    args = parse_args(argv)
    if args.input and not os.path.exists(args.input):
        print("error: no such RAW file: %s" % args.input, file=sys.stderr)
        return 1
    cnt, sgn, info = accumulate(args)
    if info["kept"] == 0:
        print("no events in the ROI", file=sys.stderr)
        return 1
    res = 1.0 / max(info["dur"], 1e-6)
    x0, y0, x1, y1 = info["roi"]
    print("%.2f s, %d events total, %d in ROI x=%d-%d y=%d-%d (%d px)"
          % (info["dur"], info["total"], info["kept"], x0, x1 - 1, y0, y1 - 1,
             info["roi_px"]))
    print("bin %.0f us -> up to %.0f Hz;  resolution %.3f Hz"
          % (args.bin_us, 0.5e6 / args.bin_us, res))

    f_u, m_u = spectrum(cnt, args.bin_us, args.fmin, args.fmax)
    f_s, m_s = spectrum(sgn, args.bin_us, args.fmin, args.fmax)
    pk_u = report("UNSIGNED - event counts per bin (what flicker_check "
                  "measures):", f_u, m_u, args, res)
    pk_s = report("SIGNED - polarity summed per bin (what the DDHF NDFT "
                  "transforms):", f_s, m_s, args, res)

    print("\nreading")
    if not pk_u or not pk_s:
        print("   not enough structure to interpret")
        return 0
    top_u, top_s = pk_u[0][0], pk_s[0][0]
    tol = max(2.0, 4 * res)
    if abs(top_u - 2 * top_s) <= tol:
        blade = top_s
        print("   unsigned %.1f Hz is 2x signed %.1f Hz - the two edges of each"
              % (top_u, top_s))
        print("   blade were being counted as two cycles.")
        print("   BLADE-PASS = %.1f Hz" % blade)
    elif abs(top_u - top_s) <= tol:
        blade = top_s
        print("   signed and unsigned agree, so each blade gives one cycle.")
        print("   BLADE-PASS = %.1f Hz" % blade)
    else:
        blade = top_s
        print("   signed %.1f Hz and unsigned %.1f Hz are not 1:1 or 1:2 - "
              "read the tables above rather than trusting one number."
              % (top_s, top_u))
    # The comb, not the loudest line: on an N-bladed fan the blade-pass is the
    # Nth harmonic of the shaft rate and dominates it, so "strongest peak" is
    # the wrong fundamental by a factor of N.
    lad = ladder(pk_s, res) or ladder(pk_u, res)
    if lad is not None:
        f0, n, k, ok = lad
        print("   harmonic comb: %.2f Hz explains %d of the peaks above "
              "(orders %s)" % (f0, n, ", ".join(
                  "%d" % int(v) for v, o in zip(k, ok) if o)))
        if args.blades > 0:
            bp = f0 * args.blades
            print("   at %d blades that makes shaft %.2f Hz = %.0f rpm, "
                  "blade-pass %.1f Hz" % (args.blades, f0, f0 * 60.0, bp))
            if abs(bp - blade) > tol:
                print("   NOTE: the comb says blade-pass %.1f Hz but the "
                      "strongest signed line is %.1f Hz (%.2fx) - trust the "
                      "comb" % (bp, blade, blade / max(bp, 1e-9)))
            blade = bp
    if args.blades > 0 and blade > 0:
        shaft = blade / args.blades
        # Search BOTH spectra: the shaft line is often stronger in the counts
        # than in the polarity sum, because a once-per-revolution imbalance is
        # not a paired ON/OFF edge and does not partly cancel when signed.
        snr = max(amp_at(f_s, m_s, shaft, tol), amp_at(f_u, m_u, shaft, tol))
        print("   %d blades -> shaft %.2f Hz = %.0f rpm, best SNR there %.1f%s"
              % (args.blades, shaft, shaft * 60.0, snr,
                 "  (present - confirms the reading)" if snr >= 5.0
                 else "  (not found; below --fmin, or lost in the noise)"))
    if args.mains > 0:
        # Tight: the grid holds to about +-0.05 Hz, so a lamp harmonic lands
        # within a few bins. A wide window here reports a nearby ROTOR line as
        # mains - at 0.2 Hz resolution a +-2 Hz window calls 198.2 Hz "200".
        mtol = max(0.5, 3 * res)
        ripple = 2 * args.mains
        print("   mains ripple would be at %.0f/%.0f/%.0f Hz (+-%.1f):  SNR "
              "%.1f / %.1f / %.1f"
              % (ripple, 2 * ripple, 3 * ripple, mtol,
                 amp_at(f_u, m_u, ripple, mtol),
                 amp_at(f_u, m_u, 2 * ripple, mtol),
                 amp_at(f_u, m_u, 3 * ripple, mtol)))
        print("   (a real lamp shows all three and none of them move when the "
              "fan speed changes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
