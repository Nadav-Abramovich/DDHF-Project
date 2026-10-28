"""
show_case_graphs.py — per-case performance graphs in one window.

Top axes: selectable performance metrics per case id (checkboxes on the left) —
counts (TP / FP / FN / TN) on the left y-axis, rates (precision / recall /
gtf_hit_rate / FP-per-frame / median events-per-px-per-frame) on the right
y-axis. Test-split cases are shaded. Bottom axes: median_events_per_px_per_frame
per case (from case_statistics.csv), sharing the same x (case id).

Data source (per-case metrics): by default the WORKING-POINT BROWSER — every
working_point_eval_*.csv in fred/data/working_points (bayes_working_point
winners, plus the older find_working_point results). The button at the bottom cycles between the
working points; double-clicking a case column opens frame_explorer's GUI on
that case with the selected working point's FULL parameter set — an exact
replay of the pipeline the scores came from (cached features, same tracker).
--sweep-csv/--combo fall back to plotting a sweep CSV's best combo instead.

Usage:
    python show_case_graphs.py                       # working-point browser
    python show_case_graphs.py --eval-csv working_point_eval_baseline_track.csv
    python show_case_graphs.py --sweep-csv sweep_wp_baseline.csv
    python show_case_graphs.py --combo "me3_sf0.7_cb2.0_..."
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained layout: every module this tool needs sits next to this file,
# so "the repo" and "the code dir" are both just _THIS_DIR.
_REPO     = _THIS_DIR
_CODE_DIR = _THIS_DIR

# Case folders (events.raw / PADDED_RGB / coordinates.txt) are searched for
# UNDER this directory only, in this order: script/<seq>, script/cases/<seq>,
# fred/<seq> (one level up).
_CASE_ROOTS = ("", "cases", "..")

# The CSV/JSON inputs live in fred/data/<type>/ (one level up).
_DATA_DIR  = os.path.join(os.path.dirname(_THIS_DIR), "data")
_WP_DIR    = os.path.join(_DATA_DIR, "working_points")    # working points, sweeps, evals
_STATS_DIR = _DATA_DIR                                   # case_statistics.csv
_HF_DIR    = _DATA_DIR                                   # hf_index.json


def _resolve_data(p):
    """A bare filename passed to --eval-csv/--sweep-csv/--stats-csv is looked up
    in the data folders; absolute or already-existing paths are left alone."""
    if p and not os.path.isabs(p) and not os.path.exists(p):
        for d in (_WP_DIR, _STATS_DIR):
            cand = os.path.join(d, p)
            if os.path.exists(cand):
                return cand
    return p

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# All metrics are PERCENTAGES on one axis: TP/FN normalised by the number of
# GT frames, FP/TN by the total number of frames; precision/recall/hit-rate x100.
# (FP% of frames == FP-per-frame x100, so no separate FP/frame entry.)
METRICS = [("tp_pct", "TP (%GT)"), ("fn_pct", "FN (%GT)"),
           ("fp_pct", "FP boxes (%fr)"), ("fpfr_pct", "FP frames (%fr)"),
           ("tn_pct", "TN (%fr)"),
           ("precision_pct", "prec (boxes)"),
           ("recall_pct", "recall (boxes)"),
           ("f1_pct", "F1 (boxes)")]                 # harmonic mean of the two
MED_LABEL = "ev/px/frame (bars)"                     # background bars, own axis;
DENSITY_COLS = {"median": "median_events_per_px_per_frame",
                "avg": "avg_events_per_px_per_frame"}
DEFAULT_ON = {"prec (boxes)", "recall (boxes)"}
MODES_IN_TAG = ("baseline", "pool", "pixelize", "components")


def _f1(p, r):
    """Harmonic mean of a precision/recall pair (same units in, same out;
    0 when both are 0, NaN-safe for the columns an eval CSV may not carry)."""
    return 2.0 * p * r / (p + r).replace(0, np.nan) if hasattr(p, "replace") \
        else (2.0 * p * r / (p + r) if (p + r) else 0.0)


def _density_arrays(st, seqs):
    """{'median': arr, 'avg': arr} of events-per-GT-px-per-frame per case."""
    out = {}
    for key, col in DENSITY_COLS.items():
        arr = np.full(len(seqs), np.nan)
        if st is not None and col in st.columns:
            m = dict(zip(st["seq"], st[col]))
            arr = np.array([m.get(s, np.nan) for s in seqs], float)
        out[key] = arr
    return out


def _seq_key(s):
    s = str(s)
    return (0, int(s)) if s.isdigit() else (1, s)


PARAM_COLS = [("min_events", "min_events>="), ("tau_sf", "tau_sf<="),
              ("tau_comb", "tau_comb>="), ("min_rotor_pixels", "min_rotor_px>="),
              ("min_box_area", "min_box_area>="), ("alpha", "alpha="),
              ("min_fill_ratio", "fill>="), ("min_snr_sum", "snr_sum>="),
              ("min_snr_avg", "snr_avg>=")]


def _params_line(d, tracked=None, extra=""):
    """Human-readable threshold line for the title."""
    parts = [f"{lab}{d[k]}" for k, lab in PARAM_COLS if k in d and d[k] is not None]
    if "persist_m" in d and "persist_k" in d:
        parts.append(f"persist {d['persist_m']}/{d['persist_k']}")
    if tracked is not None:
        parts.append("tracking ON" if tracked else "tracking off")
    if extra:
        parts.append(extra)
    return "   ".join(str(p) for p in parts)


# ── working-point discovery + debug-viewer launch ────────────────────────────────

def _discover_wps(d, include_s4=False):
    """Every working point that has a per-case eval CSV next to this script:
    [{label, csv, meta}], meta from the companion working_point_<tag>.json.
    Stride-4 subset runs are hidden unless include_s4 (pass --all-wps)."""
    import json
    out = []
    for csv_path in sorted(glob.glob(os.path.join(d, "working_point_eval_*.csv"))):
        tag = os.path.basename(csv_path)[len("working_point_eval_"):-len(".csv")]
        if "_s4" in tag and not include_s4:
            continue
        jp = os.path.join(d, f"working_point_{tag}.json")
        meta = {}
        if os.path.exists(jp):
            try:
                with open(jp, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
        if "mode" not in meta:                       # older files lack it
            meta["mode"] = next((m for m in MODES_IN_TAG if m in tag), "baseline")
        if "track" not in meta:
            meta["track"] = "track" in tag
        out.append(dict(label=tag, csv=csv_path, json=jp, meta=meta))
    return out


def _case_dir(seq, download=True, progress=None):
    """The case folder the debug viewer needs, resolved relative to this
    script's own directory (see _CASE_ROOTS). When the case is not there it is
    downloaded from the FRED dataset on HuggingFace into cases/ (progress is
    passed to hf_cases.download_case)."""
    for root in _CASE_ROOTS:
        d = os.path.join(_THIS_DIR, root, str(seq))
        if os.path.isdir(d):
            return d
    if not download:
        return None
    try:
        import hf_cases
    except Exception as exc:
        print("case %s: not local and hf_cases unavailable (%s)" % (seq, exc))
        return None
    return hf_cases.ensure_case(seq, os.path.join(_THIS_DIR, "cases"),
                                data_dir=_HF_DIR, progress=progress)


def _wp_data_dir(wp_json):
    """The feature cache a working point was searched on, from its companion
    <wp>.search.json (feature_cache.dir, relative to the project root); None
    when absent, i.e. the default utility/frame_data."""
    if not wp_json:
        return None
    side = os.path.splitext(wp_json)[0] + ".search.json"
    if not os.path.exists(side):
        return None
    try:
        with open(side, encoding="utf-8") as fh:
            rel = (json.load(fh).get("feature_cache") or {}).get("dir")
    except Exception:
        return None
    if not rel:
        return None
    root = os.path.dirname(os.path.dirname(_THIS_DIR))
    d = os.path.normpath(rel if os.path.isabs(rel) else os.path.join(root, rel))
    default = os.path.normpath(os.path.join(os.path.dirname(_THIS_DIR), "utility", "frame_data"))
    return None if d == default else d


def _has_frame_cache(seq, mode, data_dir=None):
    """True when frame_features has cached this case+mode (what --wp replays from)."""
    base = os.path.join(data_dir or os.path.join(os.path.dirname(_THIS_DIR), "utility", "frame_data"),
                        f"frames_{seq}_{mode}")
    return os.path.exists(base + ".npz") or os.path.exists(base + ".csv.gz")


def _debug_viewer_cmd(seq, meta, wp_json=None):
    """debug_viewer_pooled command for `seq` at this working point — the
    raw-event debugger (click a pixel to see its DDHF transform). None when the
    raw case is absent.

    With wp_json (and the case's frame_data cache present) it passes --wp, so the
    viewer's boxes and green mask are the working point's OWN replayed detections
    — min_snr_sum / min_snr_avg / persistence / the coast-confirm-acquire-predict
    tracker included. Without it the viewer can only be given the thresholds its
    CLI exposes, which is an APPROXIMATION: the gates it drops are typically the
    ones removing most false positives."""
    p = meta.get("params", {})
    mode = meta.get("mode", "baseline")
    case = _case_dir(seq)
    if case is None:
        return None
    cmd = [sys.executable, os.path.join(_CODE_DIR, "debug_viewer_pooled.py"),
           "--seq", os.path.relpath(case, _CODE_DIR)]   # resolved vs this folder
    if wp_json and os.path.exists(wp_json) \
            and _has_frame_cache(seq, mode, _wp_data_dir(wp_json)):
        return cmd + ["--wp", os.path.abspath(wp_json)]  # faithful replay
    cmd += ["--mode", mode,
            "--min-events", str(int(round(float(p.get("min_events", 4))))),
            "--tau-sf", str(float(p.get("tau_sf", 0.7))),
            "--tau-comb", str(float(p.get("tau_comb", 3.0))),
            "--min-rotor-pixels", str(int(round(float(p.get("min_rotor_pixels", 1))))),
            "--min-box-area", str(int(round(float(p.get("min_box_area", 10))))),
            "--alpha", str(float(p.get("alpha", 1.5)))]
    return cmd


def _viewer_cmd(seq, meta, wp_json=None):
    """frame_explorer GUI command for `seq` at this working point — the EXACT
    pipeline the scores came from (cached features, full parameter set,
    including the tracking pass), so what it shows is a bit-true replay.
    A working point searched on another feature cache (e.g. the
    subharmonic-reject one) is replayed from that cache."""
    p = meta.get("params", {})

    def gi(k, dflt):
        return str(int(round(float(p.get(k, dflt)))))

    def gf(k, dflt):
        return str(float(p.get(k, dflt)))

    cmd = [sys.executable, os.path.join(_THIS_DIR, "frame_explorer.py"),
           "--seq", str(seq), "--mode", meta.get("mode", "baseline"),
           "--min-events", gi("min_events", 4),
           "--tau-sf", gf("tau_sf", 0.7),
           "--tau-comb", gf("tau_comb", 3.0),
           "--min-rotor-pixels", gi("min_rotor_pixels", 1),
           "--min-fill-ratio", gf("min_fill_ratio", 0.0),
           "--min-box-area", gi("min_box_area", 10),
           "--alpha", gf("alpha", 1.5),
           "--min-snr-sum", gf("min_snr_sum", 0.0),
           "--min-snr-avg", gf("min_snr_avg", 0.0),
           "--persist-m", gi("persist_m", 1),
           "--persist-k", gi("persist_k", 1),
           # mirror the metric the working point was scored with
           "--skip-start" if meta.get("skip_start") else "--no-skip-start"]
    if meta.get("match"):
        cmd += ["--match", str(meta["match"])]
    if meta.get("track"):
        cmd += ["--track",
                "--track-sigma", str(float(meta.get("track_sigma", 20.0))),
                "--track-coast", gi("track_coast", 0),
                "--track-confirm", gi("track_confirm", 1),
                "--track-acquire", gf("track_acquire", 1.0)]
        if p.get("track_predict"):
            cmd += ["--track-predict"]
    data_dir = _wp_data_dir(wp_json)
    if data_dir:
        cmd += ["--data-dir", data_dir]
    return cmd


def _percase_from_sweep(sweep_csv, combo, max_fppf):
    """Per-case metric rows for one combo of a sweep CSV (default: the best
    combo on the train split by gtf_hit_rate under the FP/frame budget)."""
    import pandas as pd
    df = pd.read_csv(sweep_csv, dtype={"seq": str})
    if combo is None:
        agg = (df.groupby(["combo", "split"], dropna=False)
                 .agg(frames=("frames", "sum"), gtn=("gt_frames", "sum"),
                      hit=("gt_frames_hit", "sum"), fp=("fp", "sum"))
                 .reset_index())
        agg["gtf"] = agg["hit"] / agg["gtn"].clip(lower=1)
        agg["fppf"] = agg["fp"] / agg["frames"].clip(lower=1)
        pick = agg[agg.split.astype(str) == "train"]
        if pick.empty:
            pick = agg
        ok = pick[pick.fppf <= max_fppf]
        if ok.empty:
            ok = pick.sort_values("fppf")
        combo = ok.sort_values("gtf", ascending=False).iloc[0]["combo"]
    ev = df[df.combo == combo].copy()
    ev["precision"] = ev.tp / (ev.tp + ev.fp).clip(lower=1)
    ev["recall"] = ev.tp / (ev.tp + ev.fn).clip(lower=1)
    ev["gtf_hit_rate"] = ev.gt_frames_hit / ev.gt_frames.clip(lower=1)
    ev["fp_per_frame"] = ev.fp / ev.frames.clip(lower=1)
    return ev, combo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", default=None,
                    help="Per-case metrics CSV to open first (default: the "
                         "first discovered working point).")
    ap.add_argument("--sweep-csv", default=None,
                    help="Sweep CSV to plot per-case instead of the "
                         "working-point browser.")
    ap.add_argument("--combo", default=None,
                    help="Combo id to plot from the sweep (default: best combo).")
    ap.add_argument("--max-fppf", type=float, default=0.2,
                    help="FP/frame budget for the best-combo pick (default 0.2).")
    ap.add_argument("--stats-csv",
                    default=os.path.join(_STATS_DIR, "case_statistics.csv"))
    ap.add_argument("--drone", default=None,
                    help="Pre-select a drone type filter (also switchable in the "
                         "viewer's radio panel).")
    ap.add_argument("--all-wps", action="store_true",
                    help="Include the stride-4 subset runs in the browser.")
    ap.add_argument("--save", default=None, help="Save PNG instead of showing.")
    args = ap.parse_args()

    # bare filenames resolve against data/
    args.eval_csv  = _resolve_data(args.eval_csv)
    args.sweep_csv = _resolve_data(args.sweep_csv)
    args.stats_csv = _resolve_data(args.stats_csv)

    import pandas as pd
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons, RadioButtons, Button

    def _load_eval_csv(path):
        ev = pd.read_csv(path, dtype={"seq": str})
        title = os.path.basename(path)
        # thresholds live in the companion working_point_*.json
        wp_json = path.replace("working_point_eval_", "working_point_") \
                      .replace(".csv", ".json")
        if os.path.exists(wp_json):
            import json
            meta = json.load(open(wp_json, encoding="utf-8"))
            extra = f"match={meta.get('match', '?')}"
            if meta.get("track"):
                extra += f"  sigma={meta.get('track_sigma')}"
            if meta.get("skip_start"):
                extra += "  [scored from 1st GT sighting]"
            title += "\n" + _params_line(meta.get("params", {}),
                                         tracked=meta.get("track"), extra=extra)
        return ev, title

    FULL_COLS = {"full_frames", "full_fp", "full_fp_frames", "full_tn_frames"}

    def _prep(ev, include_lead=False):
        """Sort by case + derive the percent columns; returns (ev, seqs, split).
        include_lead swaps in the lead-in-INCLUSIVE counts (full_*) when the
        eval CSV carries them (dual-metric evals); rate columns are recomputed
        from the selected counts either way."""
        ev = ev.sort_values("seq", key=lambda s: s.map(_seq_key)) \
               .reset_index(drop=True)
        if include_lead and FULL_COLS <= set(ev.columns):
            ev = ev.assign(frames=ev["full_frames"], fp=ev["full_fp"],
                           fp_frames=ev["full_fp_frames"],
                           tn_frames=ev["full_tn_frames"])
        ev["precision"] = ev["tp"] / (ev["tp"] + ev["fp"]).clip(lower=1)
        ev["recall"] = ev["tp"] / (ev["tp"] + ev["fn"]).clip(lower=1)
        ev["gtf_hit_rate"] = (ev["gt_frames_hit"]
                              / ev["gt_frames"].clip(lower=1))
        gtf_n = ev["gt_frames"].clip(lower=1)
        fr_n = ev["frames"].clip(lower=1)
        ev["tp_pct"] = 100.0 * ev["tp"] / gtf_n
        ev["fn_pct"] = 100.0 * ev["fn"] / gtf_n
        ev["fp_pct"] = 100.0 * ev["fp"] / fr_n
        ev["tn_pct"] = (100.0 * ev["tn_frames"] / fr_n
                        if "tn_frames" in ev.columns else np.nan)
        ev["fpfr_pct"] = (100.0 * ev["fp_frames"] / fr_n
                          if "fp_frames" in ev.columns else np.nan)
        # frame-level precision: of frames that hit vs frames that false-alarmed
        if "fp_frames" in ev.columns:
            hit = ev["gt_frames_hit"]
            ev["precfr_pct"] = 100.0 * hit / (hit + ev["fp_frames"]).clip(lower=1)
        else:
            ev["precfr_pct"] = np.nan
        ev["precision_pct"] = 100.0 * ev["precision"]
        ev["recall_pct"] = 100.0 * ev["recall"]
        ev["gtf_pct"] = 100.0 * ev["gtf_hit_rate"]
        ev["f1_pct"] = _f1(ev["precision_pct"], ev["recall_pct"])
        ev["f1fr_pct"] = _f1(ev["precfr_pct"], ev["gtf_pct"])
        seqs = ev["seq"].astype(str).tolist()
        split = (ev["split"].fillna("").astype(str).to_numpy()
                 if "split" in ev.columns else np.array([""] * len(seqs)))
        return ev, seqs, split

    # ── choose the data source ───────────────────────────────────────────────
    wps = _discover_wps(_WP_DIR, include_s4=args.all_wps)
    S = {"wi": -1}
    if args.sweep_csv or args.combo is not None or (not wps and not args.eval_csv):
        sweep_csv = args.sweep_csv                   # sweep fallback
        if sweep_csv is None:
            cands = sorted(glob.glob(os.path.join(_WP_DIR, "sweep*.csv")),
                           key=os.path.getmtime)
            cands = [c for c in cands if "_summary" not in c]
            if not cands:
                raise SystemExit("no data found — run bayes_working_point.bat "
                                 "first.")
            sweep_csv = cands[-1]
        ev, combo = _percase_from_sweep(sweep_csv, args.combo, args.max_fppf)
        row0 = ev.iloc[0].to_dict() if len(ev) else {}
        tracked = "track" in os.path.basename(sweep_csv)
        title = (f"{os.path.basename(sweep_csv)}   combo: {combo}\n"
                 + _params_line(row0, tracked=tracked))
        print(f"sweep: {sweep_csv}\ncombo: {combo}  ({len(ev)} cases)")
    else:
        first = args.eval_csv or wps[0]["csv"]
        for i, w in enumerate(wps):
            if os.path.abspath(w["csv"]) == os.path.abspath(first):
                S["wi"] = i
                break
        ev, title = _load_eval_csv(first)
        print("eval:", first)
        if wps:
            print(f"working points ({len(wps)}):",
                  ", ".join(w["label"] for w in wps))

    st = pd.read_csv(args.stats_csv, dtype={"seq": str}) \
        if os.path.exists(args.stats_csv) else None
    LI = {"include": False}                          # count the lead-in? (toggle)
    S["raw_ev"], S["raw_title"] = ev.copy(), title
    ev, seqs0, split0 = _prep(ev, LI["include"])
    S.update(ev=ev, seqs=seqs0, split=split0, title=title,
             med_map=_density_arrays(st, seqs0))
    Dsel = {"stat": "median"}

    # drone type per case (for the filter panel)
    drone_of = {}
    if st is not None and "drone_type" in st.columns:
        drone_of = dict(zip(st["seq"].astype(str), st["drone_type"].astype(str)))
    dtypes = ["all"] + sorted(set(drone_of.values()) | {"unknown"})

    # Radio labels: the full "A|B" drone names overflow the panel, so show a
    # short form and map it back to the full name for filtering.
    _SHORT = (("DarwinFPV cineape", "Darwin cine"), ("Betafpv ", "Beta "),
              ("DJI ", ""))

    def _short(d):
        parts = []
        for p in d.split("|"):
            for a, b in _SHORT:
                p = p.replace(a, b)
            parts.append(p)
        lab = " + ".join(parts)
        return lab if len(lab) <= 26 else lab[:25] + "…"

    def _totals(ev_):
        """Mean per-case box-level rates: each case's own precision / recall /
        F1, averaged over cases (every case weighs the same) - the
        precision_mean / f1_mean that bayes_working_point.py maximises."""
        pc = ev_["tp"] / (ev_["tp"] + ev_["fp"]).clip(lower=1)
        rc = ev_["tp"] / (ev_["tp"] + ev_["fn"]).clip(lower=1)
        fc = _f1(pc, rc).fillna(0.0)
        name = wps[S["wi"]]["label"] if S["wi"] >= 0 else "sweep"
        tot_head.set_text(f"{name} statistics")
        return ("mean per case\n"
                f"precision {pc.mean():.3f}\n"
                f"recall    {rc.mean():.3f}\n"
                f"F1        {fc.mean():.3f}")

    def _fmt_title(t):
        """Wrap each title line so long parameter lists stay above the plot."""
        import textwrap
        return "\n".join(textwrap.fill(line, 175, break_on_hyphens=False)
                         for line in t.split("\n"))

    fig = plt.figure(figsize=(13.5, 7.5))
    fig.canvas.manager.set_window_title("per-case performance")
    ax = fig.add_axes([0.20, 0.20, 0.755, 0.70])
    ax_den = fig.add_axes([0.012, 0.80, 0.145, 0.075])
    ax_den.set_title("density stat", fontsize=8)
    ax_chk = fig.add_axes([0.012, 0.40, 0.145, 0.36])
    ax_chk.set_title("metrics", fontsize=9)
    ax_dr = fig.add_axes([0.012, 0.015, 0.145, 0.345])
    ax_dr.set_title("drone type", fontsize=9)
    axr = ax.twinx()

    # status line + progress bar (top-left): downloads and viewer launches
    status_txt = fig.text(0.012, 0.972, "", fontsize=9, fontweight="bold",
                          color="crimson", va="center")
    ax_prog = fig.add_axes([0.012, 0.915, 0.145, 0.022])
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1)
    ax_prog.set_xticks([]); ax_prog.set_yticks([])
    prog_bar = ax_prog.barh([0.5], [0.0], height=1.0, color="#2a7ab0")[0]
    prog_txt = ax_prog.text(0.5, 0.5, "", ha="center", va="center", fontsize=7)
    ax_prog.set_visible(False)

    def _set_progress(frac, label=""):
        """frac in [0,1] shows the bar; None hides it."""
        if frac is None:
            ax_prog.set_visible(False)
            return
        ax_prog.set_visible(True)
        prog_bar.set_width(max(0.0, min(1.0, frac)))
        prog_txt.set_text(label)

    items = {}                                       # label -> [kind, y, set_vis, handle, get_vis]
    lab_of = {}                                      # metric column -> final label
    B = {"bars": None, "vis": MED_LABEL in DEFAULT_ON}
    SP = []                                          # test-split shading spans

    def _bar_vis(v):
        B["vis"] = v
        if B["bars"] is not None:
            for p_ in B["bars"].patches:
                p_.set_visible(v)
    items[MED_LABEL] = ["med", S["med_map"]["median"], _bar_vis, None,
                        lambda: B["vis"]]

    x = np.arange(len(S["seqs"]))
    labels = []
    for col, lab in METRICS:
        y = np.asarray(S["ev"][col], float) if col in S["ev"].columns \
            else np.full(len(x), np.nan)
        if not np.isfinite(y).any():                 # column absent in this source
            lab = lab + " (no data)"                 # (e.g. old-schema sweep files)
        (ln,) = ax.plot(x, y, marker="o", ms=3, lw=1.2, label=lab, zorder=3)
        items[lab] = ["pct", y, ln.set_visible, ln, ln.get_visible]
        lab_of[col] = lab
        labels.append(lab)
    labels.append(MED_LABEL)
    for lab in labels:
        base = lab.replace(" (no data)", "")
        items[lab][2](base in DEFAULT_ON and "(no data)" not in lab)

    F = {"idx": np.arange(len(S["seqs"])), "sel": "all"}
    H = {"xi": None, "tick": None}                   # hovered column
    HV = {"span": None, "ann": None}                 # hover artists (set below)

    # case-id colours: green = case folder on disk, red = needs a download;
    # the hovered id gets its own colour on a gold tag so it reads either way
    TICK_ON_DISK, TICK_MISSING, TICK_HOVER = "#1b8a3a", "#d32f2f", "#3b1fa8"

    def _on_disk(seq):
        return _case_dir(seq, download=False) is not None

    def _color_ticks():
        """Re-colour every case id by whether its case folder is on disk."""
        for t in ax.get_xticklabels():
            if t is not H.get("tick"):
                t.set_color(TICK_ON_DISK if _on_disk(t.get_text()) else TICK_MISSING)

    def _hover_draw():
        """Show the highlight / info box / enlarged tick for H['xi']."""
        if HV["span"] is None:
            return
        if H["tick"] is not None:                    # restore the old tick
            t = H["tick"]
            t.set_fontsize(7); t.set_fontweight("normal"); t.set_bbox(None)
            t.set_color(TICK_ON_DISK if _on_disk(t.get_text()) else TICK_MISSING)
            H["tick"] = None
        xi = H["xi"]
        if xi is None:
            HV["span"].set_visible(False); HV["ann"].set_visible(False)
            return
        i = int(F["idx"][xi])
        seq = S["seqs"][i]
        HV["span"].set_x(xi - 0.5); HV["span"].set_width(1.0)
        HV["span"].set_visible(True)
        row = S["ev"].iloc[i]

        def _v(col):
            v = row.get(col, np.nan)
            return "—" if not np.isfinite(v) else f"{v:.0f}%"
        split = S["split"][i]
        lines = [f"case {seq}" + (f"   ({split})" if split else ""),
                 drone_of.get(seq, "unknown drone"),
                 f"precision {_v('precision_pct')}  recall {_v('recall_pct')}  "
                 f"F1 {_v('f1_pct')}",
                 "downloaded" if _on_disk(seq)
                 else "not downloaded (opening it downloads it)",
                 "click the case id to open"]
        left = xi < F["idx"].size / 2
        HV["ann"].set_text("\n".join(lines))
        HV["ann"].set_x(xi + (0.8 if left else -0.8))
        HV["ann"].set_ha("left" if left else "right")
        HV["ann"].set_visible(True)
        ticks = ax.get_xticklabels()
        if xi < len(ticks):
            t = ticks[xi]
            t.set_fontsize(10); t.set_fontweight("bold"); t.set_color(TICK_HOVER)
            t.set_bbox(dict(boxstyle="round,pad=0.15", fc="gold", ec="none"))
            H["tick"] = t


    def _apply_filter(sel):
        seqs = S["seqs"]
        idx = np.array([i for i, s in enumerate(seqs)
                        if sel == "all" or drone_of.get(s, "unknown") == sel], int)
        if idx.size == 0:                            # e.g. --drone not in this WP
            if sel != "all":
                _apply_filter("all")
            return
        if status_txt.get_text().startswith("no "):  # stale WP-switch notice
            status_txt.set_text("")
        F["idx"] = idx; F["sel"] = sel
        cur = S["med_map"][Dsel["stat"]]
        items[MED_LABEL][1] = cur
        xs2 = np.arange(idx.size)
        for it in items.values():
            if it[0] == "pct":
                it[3].set_data(xs2, it[1][idx])
        if B["bars"] is not None:                    # rebuild the density bars
            for p_ in B["bars"].patches:
                p_.remove()
        B["bars"] = axr.bar(xs2, cur[idx], color="#5a7d9a", alpha=0.35,
                            zorder=1, label=f"{Dsel['stat']} ev/px/frame")
        items[MED_LABEL][3] = B["bars"]
        _bar_vis(B["vis"])
        for sp in SP:                                # redo test-split shading
            sp.remove()
        SP.clear()
        for xi, i in enumerate(idx):
            if S["split"][i] == "test":
                SP.append(ax.axvspan(xi - 0.5, xi + 0.5, color="0.92", zorder=0))
        ax.set_xticks(xs2)
        ax.set_xticklabels([seqs[i] for i in idx], rotation=90, fontsize=7)
        H["xi"] = None                               # columns changed: drop hover
        H["tick"] = None
        _color_ticks()                               # green on disk / red missing
        ax.set_xlim(-0.5, max(0.5, idx.size - 0.5))
        _hover_draw()
        _legend(); _rescale()

    def _rescale():
        idx = F["idx"]
        ys = [it[1][idx] for it in items.values()
              if it[0] == "pct" and it[4]() and np.isfinite(it[1][idx]).any()]
        hi = max([float(np.nanmax(y)) for y in ys] + [100.0]) if ys else 100.0
        ax.set_ylim(0, hi * 1.05 + 1e-9)
        cur = S["med_map"][Dsel["stat"]]
        if np.isfinite(cur[idx]).any():
            axr.set_ylim(0, float(np.nanmax(cur[idx])) * 1.08 + 1e-9)
        fig.canvas.draw_idle()

    def _legend():
        hs = [it[3] for it in items.values() if it[4]() and it[3] is not None]
        ax.legend(hs, [h.get_label() for h in hs], fontsize=8,
                  loc="upper right", ncols=2)

    def _toggle(label):
        it = items[label]
        it[2](not it[4]())
        _legend(); _rescale()

    checks = CheckButtons(ax_chk, labels, [lab in DEFAULT_ON for lab in labels])
    for t in checks.labels:                          # keep labels inside the box
        t.set_fontsize(8)
    checks.on_clicked(_toggle)

    DR = {"radio": None, "types": ["all"]}

    def _rebuild_drone_radio(keep):
        """(Re)build the drone-type radio with only the types the current WP
        has cases of, each with its case count. Selects `keep` when it is still
        there, else "all"; returns the selected type."""
        n = {}
        for s_ in S["seqs"]:
            d = drone_of.get(s_, "unknown")
            n[d] = n.get(d, 0) + 1
        types = ["all"] + [d for d in dtypes[1:] if d in n]
        labs = [f"all ({len(S['seqs'])})"] + [f"{_short(d)} ({n[d]})"
                                             for d in types[1:]]
        if DR["radio"] is not None:
            DR["radio"].disconnect_events()
        ax_dr.clear()
        ax_dr.set_title("drone type", fontsize=9)
        active = types.index(keep) if keep in types else 0
        r = RadioButtons(ax_dr, labs, active=active)
        for t in r.labels:
            t.set_fontsize(7)
        r.on_clicked(lambda _lab: _apply_filter(DR["types"][r.index_selected]))
        DR.update(radio=r, types=types)
        return types[active]

    def _set_density(lab):
        Dsel["stat"] = "avg" if "avg" in lab else "median"
        _apply_filter(F.get("sel", "all"))
    r_den = RadioButtons(ax_den, ["median ev/px/fr", "avg ev/px/fr"], active=0)
    for t in r_den.labels:
        t.set_fontsize(7)
    r_den.on_clicked(_set_density)

    # ── working-point browser: cycle button + double-click -> debug viewer ────
    def _show_ev(raw_ev, t2):
        """(Re)display a loaded eval under the current lead-in toggle."""
        has_full = FULL_COLS <= set(raw_ev.columns)
        if LI["include"]:
            t2 = t2 + ("   [incl. lead-in]" if has_full else
                       "   [no lead-in split stored — shown as scored]")
        ev2, seqs2, split2 = _prep(raw_ev, LI["include"])
        S.update(ev=ev2, seqs=seqs2, split=split2, title=t2,
                 med_map=_density_arrays(st, seqs2))
        for col, lab in lab_of.items():
            items[lab][1] = (np.asarray(ev2[col], float) if col in ev2.columns
                             else np.full(len(seqs2), np.nan))
        items[MED_LABEL][1] = S["med_map"][Dsel["stat"]]
        ax.set_title(_fmt_title(S["title"]), fontsize=8)
        tot_txt.set_text(_totals(ev2))
        want = F.get("sel", "all")
        sel = _rebuild_drone_radio(want)
        _apply_filter(sel)
        status_txt.set_text(f"no {_short(want)} cases in this working point - "
                            "showing all drones" if sel != want else "")

    def _select_wp(i):
        ev2, t2 = _load_eval_csv(wps[i]["csv"])
        S["raw_ev"], S["raw_title"], S["wi"] = ev2.copy(), t2, i
        if wp_btn is not None:
            wp_btn.label.set_text(f"WP {i + 1}/{len(wps)}: {wps[i]['label']}")
        _show_ev(ev2, t2)

    # header sits level with the other bottom-row titles; small so the
    # longest working-point name still fits before the viewer radio
    tot_head = fig.text(0.293, 0.088, "", ha="left", va="center", fontsize=7)
    tot_txt = fig.text(0.295, 0.040, _totals(S["ev"]), ha="left", va="center",
                       fontsize=8, family="monospace")
    wp_btn = None
    if wps:
        ax_wp = fig.add_axes([0.60, 0.012, 0.355, 0.045])
        start = S["wi"] if S["wi"] >= 0 else 0
        wp_btn = Button(ax_wp, f"WP {start + 1}/{len(wps)}: {wps[start]['label']}")
        wp_btn.label.set_fontsize(8)

        def _cycle_wp(_evt):
            _select_wp((S["wi"] + 1) % len(wps))
        wp_btn.on_clicked(_cycle_wp)

        ax_vw = fig.add_axes([0.448, 0.012, 0.142, 0.07])
        ax_vw.set_title("case click opens", fontsize=7)
        # "debug viewer" = the raw-event viewer replaying THIS working point
        # (--wp); "frame_explorer" = the same replay from the cache, with sliders.
        r_vw = RadioButtons(ax_vw, ("debug viewer", "frame_explorer"), active=0)
        for t in r_vw.labels:
            t.set_fontsize(7)

        def _set_viewer(lab):
            V["which"] = lab
        r_vw.on_clicked(_set_viewer)
        fig.text(0.60, 0.072, "button: next working point    click a case id "
                              "(or double-click its column) to open it",
                 fontsize=7, color="0.35")

    ax_li = fig.add_axes([0.20, 0.012, 0.085, 0.07])
    ax_li.set_title("lead-in frames", fontsize=7)
    r_li = RadioButtons(ax_li, ("exclude", "include"), active=0)
    for t in r_li.labels:
        t.set_fontsize(7)

    def _set_leadin(lab):
        LI["include"] = (lab == "include")
        if S.get("raw_ev") is not None:
            _show_ev(S["raw_ev"], S["raw_title"])
    r_li.on_clicked(_set_leadin)

    _last = {"seq": None, "t": 0.0}
    V = {"which": "debug viewer"}                    # click-launch target
    L = {"proc": None, "seq": None, "timer": None}   # in-flight viewer launch
    DL = {"thread": None, "seq": None, "phase": "", "done": 0, "total": 0,
          "result": None, "finished": False, "timer": None,
          "kind": "download", "proc": None, "rate0": None}   # download / cache build

    def _pid_has_window(pid):
        """True once the process shows a visible top-level window (= the
        viewer finished loading and called plt.show)."""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found = ctypes.c_bool(False)

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _lp):
            p = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value == pid and user32.IsWindowVisible(hwnd):
                found.value = True
                return False
            return True
        user32.EnumWindows(_cb, 0)
        return bool(found.value)

    def _poll_launch():
        """Unlock when the viewer's window is up, its process ended, or 90 s
        passed — fail-open: a detection hiccup must never wedge the clicks."""
        import time
        p = L["proc"]
        if p is None:
            return
        try:
            rc = p.poll()
            if rc is not None and rc != 0:
                status_txt.set_text(f"viewer for case {L['seq']} exited "
                                    f"(rc={rc}) — see console")
            elif (rc is None and not _pid_has_window(p.pid)
                    and time.monotonic() - L.get("t0", 0.0) < 90.0):
                secs = time.monotonic() - L.get("t0", 0.0)
                status_txt.set_text(f"opening case {L['seq']} … {secs:.0f} s")
                fig.canvas.draw_idle()
                return                               # still loading: stay locked
            else:
                status_txt.set_text("")
        except Exception as exc:
            print(f"launch-lock check failed ({exc!r}) — unlocking", flush=True)
            status_txt.set_text("")
        L["proc"] = None                             # unlock clicks
        if L["timer"] is not None:
            L["timer"].stop()
            L["timer"] = None
        fig.canvas.draw_idle()

    def _poll_download():
        """GUI-thread side of a background case download: mirror its progress
        into the status line + bar, then continue the launch when it ends."""
        if not DL["finished"]:
            ph, done, total = DL["phase"], DL["done"], DL["total"]
            if ph == "download" and total:
                frac = done / total
                lab = f"{100 * frac:.0f}%   {done / 1e6:.0f} / {total / 1e6:.0f} MB"
                msg = f"downloading case {DL['seq']}"
            elif ph == "download":
                frac, lab = 0.0, f"{done / 1e6:.0f} MB"
                msg = f"downloading case {DL['seq']}"
            elif ph == "extract":
                frac = done / max(1, total)
                lab = f"{100 * frac:.0f}%"
                msg = f"extracting case {DL['seq']}"
            elif ph == "compute":
                import time
                frac = done / max(1, total)
                now = time.monotonic()
                if DL["rate0"] is None or done < DL["rate0"][1]:
                    DL["rate0"] = (now, done)
                t0, d0 = DL["rate0"]
                left = ""
                if done > d0 and now > t0:
                    secs = (total - done) / ((done - d0) / (now - t0))
                    left = f"   ~{int(secs // 60)}:{int(secs % 60):02d} left"
                lab = (f"{100 * frac:.0f}%   {done}/{total} frames{left}"
                       if total > 1 else "loading events into the workers…")
                msg = (f"building {DL.get('mode')} feature cache for case "
                       f"{DL['seq']} (one-time)")
            elif ph == "write":
                frac, lab = 1.0, "writing"
                msg = f"saving feature cache for case {DL['seq']}"
            elif ph == "start":
                frac, lab = 0.0, ""
                msg = f"starting feature cache build for case {DL['seq']}"
            else:
                frac, lab = 0.0, ""
                msg = f"looking up case {DL['seq']} on HuggingFace"
            status_txt.set_text(msg + " … clicks locked")
            _set_progress(frac, lab)
            fig.canvas.draw_idle()
            return
        if DL["timer"] is not None:
            DL["timer"].stop()
            DL["timer"] = None
        DL["thread"] = None
        _set_progress(None)
        _color_ticks()                               # a download may have landed
        seq = DL["seq"]
        if DL["result"] is None:
            what = ("download" if DL["kind"] == "download"
                    else "building the feature cache")
            status_txt.set_text(f"{what} of case {seq} failed — see console")
            fig.canvas.draw_idle()
            return
        status_txt.set_text("")
        _last["seq"] = None                          # not a double-click repeat
        _launch(seq)
        fig.canvas.draw_idle()

    def _start_download(seq):
        """Fetch a missing case on a worker thread so the window stays live."""
        import threading
        DL.update(seq=seq, phase="lookup", done=0, total=0, result=None,
                  finished=False, kind="download")

        def _prog(phase, done, total):
            DL.update(phase=phase, done=done, total=total)

        def _work():
            try:
                DL["result"] = _case_dir(seq, progress=_prog)
            except Exception as exc:
                print(f"case {seq}: download failed ({exc!r})", flush=True)
                DL["result"] = None
            DL["finished"] = True

        DL["thread"] = threading.Thread(target=_work, daemon=True)
        DL["thread"].start()
        DL["timer"] = fig.canvas.new_timer(interval=250)
        DL["timer"].add_callback(_poll_download)
        DL["timer"].start()
        _poll_download()

    def _start_build(seq, mode):
        """Build the case's frame_data cache (downloading the case first if
        needed) with build_frame_cache.py in a child process, mirroring its
        progress lines into the status bar. Resumes _launch when it succeeds."""
        import re
        import subprocess
        import threading
        DL.update(seq=seq, mode=mode, phase="start", done=0, total=0,
                  result=None, finished=False, kind="build", rate0=None)
        cmd = [sys.executable, "-u", os.path.join(_THIS_DIR, "build_frame_cache.py"),
               str(seq), "--mode", mode]
        print("building cache:", " ".join(cmd), flush=True)
        prog_re = re.compile(r"\[progress\] windows (\d+)/(\d+)")

        def _work():
            try:
                p = subprocess.Popen(cmd, cwd=_THIS_DIR, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace")
                DL["proc"] = p
                for line in p.stdout:
                    line = line.rstrip()
                    if line.startswith("@@ "):
                        ph, d, t = line[3:].split()
                        DL.update(phase=ph, done=int(d), total=int(t))
                        continue
                    print(line, flush=True)          # child log -> our console
                    m = prog_re.search(line)
                    if m:
                        DL.update(phase="compute", done=int(m.group(1)),
                                  total=int(m.group(2)))
                    elif line.startswith("[precompute]") and DL["phase"] != "compute":
                        DL.update(phase="compute", done=0, total=1)
                DL["result"] = True if p.wait() == 0 else None
            except Exception as exc:
                print(f"case {seq}: cache build failed ({exc!r})", flush=True)
                DL["result"] = None
            DL["proc"] = None
            DL["finished"] = True

        DL["thread"] = threading.Thread(target=_work, daemon=True)
        DL["thread"].start()
        DL["timer"] = fig.canvas.new_timer(interval=500)
        DL["timer"].add_callback(_poll_download)
        DL["timer"].start()
        _poll_download()

    def _on_close(_evt):
        """Closing the window stops an in-flight cache build (resumable)."""
        p = DL.get("proc")
        if p is not None and p.poll() is None:
            print("window closed - stopping the cache build", flush=True)
            p.kill()
    fig.canvas.mpl_connect("close_event", _on_close)

    def _launch(seq):
        """Open `seq` with the selected working point in the chosen viewer.
        Locked while a previous launch is still loading (its window not up
        yet) or a case is downloading, so stray clicks can't pile up."""
        import time
        if not wps or S["wi"] < 0:
            return
        if DL["thread"] is not None:
            print(f"still preparing case {DL['seq']} — wait for it", flush=True)
            return
        if L["proc"] is not None:
            _poll_launch()                           # lazy re-check on click
        if L["proc"] is not None:
            print(f"still loading case {L['seq']} — wait for its window",
                  flush=True)
            return
        if seq == _last["seq"] and time.monotonic() - _last["t"] < 2.0:
            return
        _last["seq"], _last["t"] = seq, time.monotonic()
        w = wps[S["wi"]]
        mode = w["meta"].get("mode", "baseline")
        # frame_explorer and the exact (--wp) debug-viewer replay both read the
        # case's frame_data cache: build it first when it is missing.
        exact = (V["which"] == "frame_explorer"
                 or (w.get("json") and os.path.exists(w["json"])))
        wp_data = _wp_data_dir(w.get("json"))
        if exact and not _has_frame_cache(seq, mode, wp_data):
            if wp_data:                              # not the default cache: the
                print(f"case {seq}: not in {wp_data} (the cache "   # builder below
                      f"{w['label']} was searched on) — build it with "  # only makes
                      "frame_features.py and the options in its "        # default ones
                      ".search.json", flush=True)
                return
            _start_build(seq, mode)                  # resumes _launch when done
            return
        cmd = None
        if V["which"] == "debug viewer":
            if _case_dir(seq, download=False) is None:
                _start_download(seq)                 # resumes _launch when done
                return
            cmd = _debug_viewer_cmd(seq, w["meta"], wp_json=w.get("json"))
            if cmd is None:
                print(f"case {seq}: raw case folder not found — "
                      "opening frame_explorer instead", flush=True)
            elif "--wp" in cmd:
                print(f"faithful replay: the viewer shows {w['label']}'s own "
                      "detections (full parameter set)", flush=True)
            else:
                print(f"case {seq}: no frame_data cache for "
                      f"[{w['meta'].get('mode', 'baseline')}] — falling back to "
                      "the viewer's approximate thresholds (min_snr_sum / "
                      "min_snr_avg / persistence / the tracker are NOT applied)",
                      flush=True)
        if cmd is None:                              # frame_explorer GUI
            cmd = _viewer_cmd(seq, w["meta"], wp_json=w.get("json"))
        print("launching:", " ".join(cmd), flush=True)
        import subprocess
        try:
            L["proc"] = subprocess.Popen(cmd, cwd=_THIS_DIR)
            L["seq"] = seq
            L["t0"] = time.monotonic()
            status_txt.set_text(f"opening case {seq} … clicks locked")
            fig.canvas.draw_idle()
            L["timer"] = fig.canvas.new_timer(interval=500)
            L["timer"].add_callback(_poll_launch)
            L["timer"].start()
        except Exception as exc:
            print(f"launch failed: {exc!r}", flush=True)
            L["proc"] = None

    # ── hover: highlight the case column under the mouse ────────────────────
    from matplotlib.transforms import blended_transform_factory
    H["tick"] = None
    hl_span = ax.axvspan(-0.5, 0.5, color="gold", alpha=0.45, zorder=0.5,
                         visible=False)
    hover_ann = axr.text(0, 0.985, "", transform=blended_transform_factory(
                             ax.transData, ax.transAxes),
                         va="top", fontsize=9, zorder=20, visible=False,
                         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                                   ec="0.4", alpha=0.95))
    HV.update(span=hl_span, ann=hover_ann)

    def _column_at(event):
        """Case column under the mouse: inside the plot or in the tick-label
        strip below it (the click target). None elsewhere."""
        bb = ax.bbox
        strip = 0.09 * fig.bbox.height
        if not (bb.x0 <= event.x <= bb.x1 and bb.y0 - strip <= event.y <= bb.y1):
            return None
        xd = ax.transData.inverted().transform((event.x, event.y))[0]
        xi = int(round(xd))
        return xi if 0 <= xi < F["idx"].size else None

    def _on_move(event):
        xi = _column_at(event)
        if xi == H["xi"]:
            return
        H["xi"] = xi
        _hover_draw()
        try:
            from matplotlib.backend_tools import Cursors
            fig.canvas.set_cursor(Cursors.HAND if xi is not None
                                  else Cursors.POINTER)
        except Exception:
            pass
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    def _on_click(event):
        """Single click on the case-id strip under the plot, or double-click
        inside a case's column, opens that case."""
        if event.button != 1:
            return
        xi = _column_at(event)
        if xi is None:
            return
        below = event.y < ax.bbox.y0
        if below or getattr(event, "dblclick", False):
            _launch(S["seqs"][int(F["idx"][xi])])
    fig.canvas.mpl_connect("button_press_event", _on_click)

    ax.set_ylabel("percent [%]")
    axr.set_ylabel("events per GT px per frame (bars)")
    ax.set_title(_fmt_title(S["title"]), fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlabel("case id")

    _apply_filter(_rebuild_drone_radio(args.drone or "all"))
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved ->", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
