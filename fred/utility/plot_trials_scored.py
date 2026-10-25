"""
plot_trials_scored.py — "Trials, scored": precision vs recall of every trial in
a bayes_working_point journal, with the recall floor and the chosen trial.

Precision = mean per-case precision, recall = GT-frame hit rate (the measure
the recall floor is applied to). Score = precision - penalty * max(0, floor -
recall); the chosen trial is the highest-scoring one, as in bayes_working_point.

Usage:
    python plot_trials_scored.py                       # ss_r80 run, floor 0.8
    python plot_trials_scored.py --journal ..\\data\\working_points\\bayes_wp_pixelize_track_r90.jsonl --floor 0.9
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WP = os.path.join(_HERE, "..", "data", "working_points")

BG, PANEL, GRID, TEXT, MUTED = "#0f172a", "#111a2e", "#1e293b", "#f1f5f9", "#94a3b8"
REJ, VALID, CHOSEN = "#64748b", "#0ea5e9", "#fbad4b"


def trial_points(journal):
    pts = []
    for line in open(journal, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("kind") != "eval" or rec.get("error"):
            continue
        rows = rec["rows"]
        prec = np.mean([r["tp"] / max(1, r["tp"] + r["fp"]) for r in rows])
        rec_ = sum(r["gt_frames_hit"] for r in rows) / max(1, sum(r["gt_frames"] for r in rows))
        pts.append((rec_, prec))
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(_WP, "bayes_wp_pixelize_track_ss_r80.jsonl"))
    ap.add_argument("--floor", type=float, default=0.8)
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(_WP, "trials_scored.png"))
    args = ap.parse_args()

    pts = trial_points(args.journal)
    r, p = pts[:, 0], pts[:, 1]
    score = p - args.penalty * np.maximum(0.0, args.floor - r)
    best = int(np.argmax(score))
    valid = r >= args.floor
    rej = ~valid
    valid[best] = rej[best] = False

    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.scatter(r[rej], p[rej], s=46, c=REJ, zorder=3,
               label=f"Rejected (recall too low) — {rej.sum()}")
    ax.scatter(r[valid], p[valid], s=46, c=VALID, zorder=3,
               label=f"Valid trials — {valid.sum()}")
    ax.scatter(r[best], p[best], s=170, c=CHOSEN, zorder=4,
               label=f"Chosen (P={p[best]:.3f}, R={r[best]:.3f})")
    ax.axvline(args.floor, color=CHOSEN, ls=(0, (4, 3)), lw=1.6, zorder=2,
               label=f"Recall floor ({args.floor:g})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (GT-frame hit rate)", color=MUTED, fontweight="bold")
    ax.set_ylabel("Precision (mean per case)", color=MUTED, fontweight="bold")
    ax.set_title(f"Trials, scored  ({len(pts)} trials)", color=TEXT,
                 fontsize=15, fontweight="bold", pad=44)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#334155")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                    frameon=False, fontsize=9, labelcolor=MUTED,
                    handletextpad=0.4, columnspacing=1.6)
    for h in leg.legend_handles[:3]:
        h.set_sizes([40])

    fig.tight_layout()
    fig.savefig(args.out, facecolor=BG)
    print(f"{len(pts)} trials: {valid.sum()} valid, {rej.sum()} rejected; "
          f"chosen P={p[best]:.4f} R={r[best]:.4f}\nwrote {args.out}")


if __name__ == "__main__":
    main()
