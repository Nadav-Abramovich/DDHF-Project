"""
pass_rate_vs_events.py — gate accuracy as a function of per-pixel event count.

Two panels over the cached frame_data features:
  LEFT  (inside GT):  fraction of in-GT pixels that PASS the spectral gates
                      (sf <= tau_sf AND snr >= tau_comb) per event-count bin —
                      "true positive rate" of the pixel test on drone pixels.
  RIGHT (outside GT): fraction of out-GT pixels that are REJECTED per bin —
                      "true negative rate" on background pixels.

Both curves rising with event count = direct evidence that the per-pixel
frequency analysis needs events to work; the faint bars show how many pixels
each bin actually contains (log scale), i.e. how much of the data lives in
each regime.

By default only the spectral gates are applied (applying min_events would
zero every bin below it and hide the region of interest); use
--include-min-events to see the full pixel-gate verdict instead.

Usage:
    python pass_rate_vs_events.py --seqs 3,10
    python pass_rate_vs_events.py --seqs 3,10 --tau-sf 0.7 --tau-comb 5
    python pass_rate_vs_events.py --seqs 3 --save fig.png
"""
import os
import sys
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_THIS_DIR)
_COMMON   = os.path.join(os.path.dirname(_REPO), "common")   # shared modules
for _p in (_THIS_DIR, os.path.join(_COMMON, "utility"), os.path.join(_COMMON, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import frame_explorer as FE

# event-count bin edges (right-open; last bin catches everything above)
BIN_EDGES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64, 10 ** 9]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", required=True,
                    help="Comma-separated case ids (aggregated together).")
    ap.add_argument("--mode", choices=("baseline", "pool", "pixelize", "components"),
                    default="baseline")
    ap.add_argument("--data-dir", default=os.path.join(_THIS_DIR, "frame_data"))
    ap.add_argument("--tau-sf", type=float, default=0.7)
    ap.add_argument("--tau-comb", type=float, default=5.0)
    ap.add_argument("--min-events", type=int, default=4)
    ap.add_argument("--include-min-events", action="store_true",
                    help="Also apply the min_events gate to the verdict "
                         "(zeroes all bins below it).")
    ap.add_argument("--wscale", type=int, default=1)
    ap.add_argument("--save", default=None, help="Save PNG instead of showing.")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    gc_all, sf_all, snr_all, ig_all = [], [], [], []
    for seq in seqs:
        print(f"loading case {seq} …")
        D = FE.load_D(seq, args.mode, args.data_dir, args.wscale)
        gc_all.append(D["gc"].astype(np.int64))
        sf_all.append(D["sf"].astype(np.float64))
        snr_all.append(D["snr"].astype(np.float64))
        ig_all.append(D["ig"].astype(bool))
        FE._D_CACHE.clear()                          # bound memory over many cases
    gc = np.concatenate(gc_all); sf = np.concatenate(sf_all)
    snr = np.concatenate(snr_all); ig = np.concatenate(ig_all)

    passes = (sf <= args.tau_sf) & (snr >= args.tau_comb)
    if args.include_min_events:
        passes &= gc >= args.min_events

    edges = BIN_EDGES
    n_bins = len(edges) - 1
    bin_of = np.digitize(gc, edges) - 1              # -1/-n outside handled below
    valid = (bin_of >= 0) & (bin_of < n_bins)

    rows = []
    for b in range(n_bins):
        m = valid & (bin_of == b)
        for region, rm in (("in", m & ig), ("out", m & ~ig)):
            tot = int(rm.sum())
            p = int((passes & rm).sum())
            rows.append((b, region, tot, p))
    lab = [f"{edges[i]}" if edges[i + 1] == edges[i] + 1 else
           (f"{edges[i]}+" if edges[i + 1] >= 10 ** 8 else f"{edges[i]}-{edges[i + 1] - 1}")
           for i in range(n_bins)]

    in_tot = np.array([r[2] for r in rows if r[1] == "in"], float)
    in_pass = np.array([r[3] for r in rows if r[1] == "in"], float)
    out_tot = np.array([r[2] for r in rows if r[1] == "out"], float)
    out_pass = np.array([r[3] for r in rows if r[1] == "out"], float)
    tpr = np.where(in_tot > 0, in_pass / np.maximum(1, in_tot), np.nan)
    tnr = np.where(out_tot > 0, 1.0 - out_pass / np.maximum(1, out_tot), np.nan)

    print(f"\ngates: sf<={args.tau_sf}  snr>={args.tau_comb}"
          + (f"  ev>={args.min_events}" if args.include_min_events else "")
          + f"   cases: {', '.join(seqs)}")
    print(f"{'events':>8} | {'in-GT n':>10} {'pass%':>7} | {'out-GT n':>10} {'reject%':>8}")
    for i in range(n_bins):
        if in_tot[i] + out_tot[i] == 0:
            continue
        print(f"{lab[i]:>8} | {int(in_tot[i]):>10} "
              f"{100 * tpr[i] if np.isfinite(tpr[i]) else float('nan'):>6.1f}% | "
              f"{int(out_tot[i]):>10} "
              f"{100 * tnr[i] if np.isfinite(tnr[i]) else float('nan'):>7.1f}%")

    x = np.arange(n_bins)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    fig.canvas.manager.set_window_title("gate accuracy vs events per pixel")
    for ax, rate, tot, name, color in (
            (axL, tpr, in_tot, "INSIDE GT: pass rate (TP rate)", "tab:green"),
            (axR, tnr, out_tot, "OUTSIDE GT: reject rate (TN rate)", "tab:red")):
        axc = ax.twinx()
        axc.bar(x, tot, color="0.85", zorder=0)
        axc.set_yscale("log")
        axc.set_ylabel("pixels in bin (log)", fontsize=8)
        axc.tick_params(labelsize=7)
        ax.plot(x, 100 * rate, "o-", color=color, lw=1.6, ms=4, zorder=3)
        ax.set_ylim(0, 105)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("events per pixel per window")
        ax.set_ylabel("percent [%]")
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=45, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_zorder(axc.get_zorder() + 1)          # lines above the count bars
        ax.patch.set_visible(False)
    fig.suptitle(f"cases {', '.join(seqs)}   gates: sf<={args.tau_sf} "
                 f"snr>={args.tau_comb}"
                 + (f" ev>={args.min_events}" if args.include_min_events else "")
                 + f"   [{args.mode}]", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved ->", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
