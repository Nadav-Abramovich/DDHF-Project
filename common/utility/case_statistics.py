"""
case_statistics.py — Per-case ground-truth / event statistics for the FRED
event-camera drone dataset.

For EVERY case (sequence) it writes one CSV row with:

  1. total_frames                   — number of event-camera frames
  2. gt_frames                      — frames that carry a GT box (drone present)
  3. total_events_in_gt             — raw events falling inside a GT box
  4. total_events_not_in_gt         — all other raw events (total - in_gt)
  5. avg_gt_size_px / median_...    — GT box area in pixels, over GT frames only
  6. avg_events_in_gt_per_frame / median_...        — events inside the GT box,
                                       per GT frame
  7. avg_events_per_px_per_frame / median_...       — (events in GT box) / (GT
                                       box area), per GT frame
  8. drone_type                     — from fred_drone_map.csv (or '?')

Data model:
  * A case folder holds the event stream (Event/events.hdf5 for HuggingFace
    cases, Event/events.raw for older local ones; x/y/p/t, t in us),
    Event/Frames/<...>_frame_<T>.png — one image PER event frame, named by that
    frame's timestamp T in microseconds (the full frame grid) — and the GROUND
    TRUTH: coordinates.txt (HuggingFace) or interpolated_coordinates.txt (older
    layout), the same interpolated boxes, one line `t_sec: x1,y1,x2,y2,...` per
    frame that contains a drone, absolute pixels in the event image.
  * total_frames = the number of event frames; gt_frames = the number of GT lines.
    Each GT frame owns the event window [T, T + step), step = the (constant)
    spacing between consecutive frame timestamps. Events inside that frame's GT
    box during the window are the frame's "events in GT".

Efficiency + robustness:
  * A case already on disk (fred/<seq> or fred/script/cases/<seq>) is measured in
    place and never modified. A missing case is downloaded from HuggingFace
    (hf_cases) into fred/script/cases and deleted again after it is measured
    (~1 GB each); --keep-downloads keeps it.
  * The per-case measurement runs in a CHILD process: Metavision keeps an OS lock
    on the event file until the owning process exits, so isolating it in a
    subprocess stops one bad case's native crash from aborting the whole run.
  * The CSV is appended + flushed after EVERY case, and the run is RESUMABLE: on
    start it reads the existing CSV and skips any case already present, so a
    killed/interrupted run simply continues where it left off.

Usage:
    python case_statistics.py                       # all cases, downloading as needed
    python case_statistics.py --no-download         # only cases already on disk
    python case_statistics.py --seqs 3,5,10         # just these cases
    python case_statistics.py --out my_stats.csv
"""
import os
import sys
import csv
import json
import time
import argparse
import subprocess
import re

import numpy as np

# Make the sibling `code/` package importable (utils.load_events + the battle-
# tested download / cleanup / logging helpers) regardless of the cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_THIS_DIR)
_FRED     = os.path.join(os.path.dirname(_REPO), "fred")   # default dataset + its data
_CODE_DIR = os.path.join(_REPO, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Default event image size, used only if a frame PNG can't be read to get the
# real dimensions the YOLO boxes are normalised against.
# Directories searched (in order) for cases that are ALREADY extracted on disk, so
# they're measured in place instead of being re-downloaded. A case found in any of
# these is used as-is and never deleted. --cases-dir adds more; the repo root is
# always included as well.
_DEFAULT_CASE_ROOTS = [
    _FRED,
    os.path.join(_FRED, "script", "cases"),   # HuggingFace downloads (hf_cases)
]


def _gt_file(case_dir):
    """The case's GT: interpolated_coordinates.txt (older local layout) or, in
    a HuggingFace case, coordinates.txt - the same interpolated boxes."""
    p = os.path.join(case_dir, "interpolated_coordinates.txt")
    if os.path.exists(p):
        return p
    p = os.path.join(case_dir, "coordinates.txt")
    return p if os.path.exists(os.path.join(case_dir, "Event", "events.hdf5")) else None


def _event_file(case_dir):
    """Event/events.raw (older local layout) or Event/events.hdf5 (HF), or None."""
    for n in ("events.raw", "events.hdf5"):
        p = os.path.join(case_dir, "Event", n)
        if os.path.exists(p):
            return p
    return None

# CSV schema. seq + drone_type are filled by the parent; the rest by the worker.
COLUMNS = [
    "seq",
    "drone_type",
    "total_frames",
    "gt_frames",
    "total_events",
    "total_events_in_gt",
    "total_events_not_in_gt",
    "avg_gt_size_px",
    "median_gt_size_px",
    "avg_events_in_gt_per_frame",
    "median_events_in_gt_per_frame",
    "avg_events_per_px_per_frame",
    "median_events_per_px_per_frame",
]


# ── Per-case measurement (runs in the child process) ────────────────────────────

def _frame_timestamp_us(fname):
    """Parse the frame timestamp (us) = the trailing integer of the filename. Works
    for BOTH dataset naming conventions: `Video_<N>_frame_<T>.png` and the
    `frame_`-less `Video_<N>_<T>.png`. Raises ValueError if there's no trailing
    number (so non-frame files are skipped by the caller)."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"no trailing timestamp in {fname}")
    return int(m.group(1))


def _frame_timestamps(case_dir):
    """Sorted event-frame timestamps (us) from Event/Frames/*.png filenames — the
    full frame grid (drone present or not), used for total_frames + the window
    step. Raises if the folder is empty/missing (an incomplete case, which the
    caller re-downloads)."""
    frames_dir = os.path.join(case_dir, "Event", "Frames")
    ts = []
    if os.path.isdir(frames_dir):
        for f in os.listdir(frames_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    ts.append(_frame_timestamp_us(f))
                except ValueError:
                    continue
    if not ts:
        raise FileNotFoundError(f"no event frames in {frames_dir}")
    return np.array(sorted(ts), dtype=np.int64)


def _parse_interpolated_gt(path):
    """Read interpolated_coordinates.txt -> {frame_ts_us: [(x1,y1,x2,y2), ...]}.

    Each line is `t_sec: x1, y1, x2, y2[, flag]`, absolute pixels in the event
    image (same space as the raw events). t_sec*1e6 is the frame timestamp;
    repeated timestamps (multiple drones) are grouped into one frame. The trailing
    flag column is ignored, matching debug_viewer_pooled._load_gt_boxes."""
    gt = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            t_str, sep, rest = line.partition(":")
            if not sep:
                continue
            try:
                t_us = int(round(float(t_str) * 1e6))
                vals = [float(v) for v in rest.replace(",", " ").split()[:4]]
            except ValueError:
                continue
            if len(vals) >= 4:
                gt.setdefault(t_us, []).append(tuple(vals[:4]))
    return gt


def _events_in_boxes(ex, ey, boxes):
    """Count events (ex, ey) inside the UNION of `boxes`, and the boxes' total
    pixel area. Inclusive edges, matching debug_viewer_pooled._events_in_gt_mask.
    Area is the sum over boxes (drones rarely overlap), so events/area is a
    faithful per-pixel density for the common single-box case."""
    if ex.size == 0 or not boxes:
        in_union = np.zeros(ex.size, dtype=bool)
    else:
        in_union = np.zeros(ex.size, dtype=bool)
        for (x1, y1, x2, y2) in boxes:
            in_union |= (ex >= x1) & (ex <= x2) & (ey >= y1) & (ey <= y2)
    area = float(sum(max(0.0, x2 - x1) * max(0.0, y2 - y1)
                     for (x1, y1, x2, y2) in boxes))
    return int(in_union.sum()), area


def compute_case_stats(case_dir):
    """Measure one unpacked case folder -> dict of the CSV metrics (minus seq /
    drone_type, which the parent fills in). Raises on missing inputs."""
    gt_path  = _gt_file(case_dir)
    raw_path = _event_file(case_dir)
    if gt_path is None or not os.path.exists(gt_path):
        raise FileNotFoundError(f"no interpolated_coordinates.txt / coordinates.txt "
                                f"in {case_dir}")
    if raw_path is None:
        raise FileNotFoundError(f"no Event/events.raw or events.hdf5 in {case_dir}")

    # total_frames = the full event-frame grid; gt = the interpolated GT boxes.
    frame_ts = _frame_timestamps(case_dir)
    total_frames = int(frame_ts.size)

    gt = _parse_interpolated_gt(gt_path)                      # {ts_us: [boxes]}
    gt_ts = sorted(gt)
    gt_frames = len(gt_ts)

    # Frame window length = the (constant) spacing between consecutive frames.
    if total_frames >= 2:
        step_us = int(np.median(np.diff(frame_ts)))
    else:
        step_us = 33333                                       # 33.333 ms fallback
    step_us = max(step_us, 1)

    # Load the raw event stream (Metavision) and sort by time for O(log N) windows.
    from utils import load_events
    events = load_events(raw_path)
    order = np.argsort(events["t"], kind="stable")
    t_all = events["t"][order].astype(np.int64)
    ex_all = events["x"][order].astype(np.int64)
    ey_all = events["y"][order].astype(np.int64)
    total_events = int(t_all.size)

    # Per GT-frame: count events inside its GT box within [T, T + step).
    gt_area, ev_in_gt, ev_per_px = [], [], []
    total_events_in_gt = 0
    for T in gt_ts:
        lo = int(np.searchsorted(t_all, T, side="left"))
        hi = int(np.searchsorted(t_all, T + step_us, side="left"))
        n_in, area = _events_in_boxes(ex_all[lo:hi], ey_all[lo:hi], gt[T])
        total_events_in_gt += n_in
        gt_area.append(area)
        ev_in_gt.append(n_in)
        ev_per_px.append((n_in / area) if area > 0 else 0.0)

    def _avg(a):
        return round(float(np.mean(a)), 4) if a else ""

    def _med(a):
        return round(float(np.median(a)), 4) if a else ""

    return {
        "total_frames": total_frames,
        "gt_frames": gt_frames,
        "total_events": total_events,
        "total_events_in_gt": total_events_in_gt,
        "total_events_not_in_gt": total_events - total_events_in_gt,
        "avg_gt_size_px": _avg(gt_area),
        "median_gt_size_px": _med(gt_area),
        "avg_events_in_gt_per_frame": _avg(ev_in_gt),
        "median_events_in_gt_per_frame": _med(ev_in_gt),
        "avg_events_per_px_per_frame": _avg(ev_per_px),
        "median_events_per_px_per_frame": _med(ev_per_px),
    }


# ── Case discovery / drone map (parent process) ─────────────────────────────────

def _load_drone_map():
    """sequence-name -> drone class string, from fred/data/fred_drone_map.csv
    (or utility/). {} if not found, so drone_type just shows '?'."""
    for cand in (os.path.join(_FRED, "data", "fred_drone_map.csv"),
                 os.path.join(_THIS_DIR, "fred_drone_map.csv")):
        if os.path.exists(cand):
            mp = {}
            with open(cand, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    seq = (row.get("sequence") or "").strip()
                    if seq:
                        mp[seq] = (row.get("drone_class(es)") or "").strip()
            return mp
    return {}


def _is_case_dir(path):
    """A COMPLETE case folder: GT (interpolated_coordinates.txt, or a HF case's
    coordinates.txt) + an event stream (Event/events.raw or events.hdf5) + a
    non-empty Event/Frames/ (the frame grid). Incomplete extractions (e.g. a
    folder missing Frames) fail this, so they're treated as absent and the case is
    re-downloaded clean instead of measured from a partial copy."""
    frames = os.path.join(path, "Event", "Frames")
    gt = _gt_file(path)
    return (gt is not None and os.path.exists(gt)
            and _event_file(path) is not None
            and os.path.isdir(frames) and len(os.listdir(frames)) > 0)


def _local_cases(roots):
    """{case_name: folder_path} for every already-extracted case found across the
    search roots. Earlier roots win when the same case name appears twice."""
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if name not in found and _is_case_dir(d):
                found[name] = d
    return found


def _print_log(log_path, msg):
    """Timestamped print + append (same format as batch's _log, without importing
    it here so the download helpers stay usable standalone)."""
    import datetime
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _seq_sort_key(name):
    """Sort numeric case names numerically, everything else after them by name."""
    return (0, int(name)) if str(name).isdigit() else (1, str(name))


def _load_existing(csv_path):
    """Set of seq names already in the CSV, so the run resumes and skips them."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s = (row.get("seq") or "").strip()
                if s:
                    done.add(s)
    except Exception:
        pass
    return done


def _append_row(csv_path, row):
    """Append one case row (header on first write), flushed + fsync'd so an
    interrupted run keeps every completed result."""
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(COLUMNS)
        w.writerow([row.get(c, "") for c in COLUMNS])
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def _run_worker(case_dir):
    """Measure one case in a CHILD process (releases the events.raw lock on exit)
    and return its metrics dict. Raises on failure."""
    out_json = os.path.join(_THIS_DIR, f"._case_stats_{os.getpid()}.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker", "--case-dir", case_dir, "--out-json", out_json]
    try:
        subprocess.run(cmd, check=True)
        if not os.path.exists(out_json):
            raise RuntimeError("worker produced no metrics JSON")
        with open(out_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        if os.path.exists(out_json):
            try:
                os.remove(out_json)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Per-case GT/event statistics CSV for the FRED dataset "
                    "(downloads missing cases from HuggingFace, resumable).")
    ap.add_argument("--out", default=os.path.join(_FRED, "data", "case_statistics.csv"),
                    help="Output CSV path (default fred/data/case_statistics.csv).")
    ap.add_argument("--seqs", default=None,
                    help="Comma-separated case names to process (default: every "
                         "case on HuggingFace, plus any local case folder).")
    ap.add_argument("--no-download", action="store_true",
                    help="Never download; only measure cases already on disk.")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="Keep cases this run downloaded (default: delete each one "
                         "after it is measured, ~1 GB per case).")
    ap.add_argument("--cases-dir", action="append", default=None,
                    help="Directory of already-extracted case folders to use in "
                         "place (repeatable). Defaults to fred/ and "
                         "fred/script/cases.")
    args = ap.parse_args()

    import batch_ddhf_eval_pooled as B      # cleanup / logging helpers
    import hf_cases                         # HuggingFace case list + download
    csv_path = os.path.abspath(args.out)
    log_path = os.path.splitext(csv_path)[0] + ".log"

    drone_map = _load_drone_map()

    # Cases already on disk are measured in place and never deleted.
    case_roots = list(args.cases_dir) if args.cases_dir else list(_DEFAULT_CASE_ROOTS)
    local = _local_cases(case_roots)      # {name: folder_path}

    index = {} if args.no_download else hf_cases.case_index()
    if args.seqs:
        seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        seqs = sorted(set(index) | set(local), key=_seq_sort_key)

    done = _load_existing(csv_path)
    B._log(log_path, f"START case statistics — {len(seqs)} cases, out={csv_path}")
    if done:
        B._log(log_path, f"Resuming: {len(done)} cases already in CSV, skipping them.")
    if not drone_map:
        B._log(log_path, "NOTE: fred_drone_map.csv not found — drone_type shown as '?'.")

    n_ok = n_err = 0
    for name in seqs:
        if name in done:
            continue
        t0 = time.perf_counter()
        B._log(log_path, f"-- Case {name} " + "-" * 30)
        case_dir, downloaded = local.get(name), False
        try:
            if case_dir:
                B._log(log_path, f"  using local case {case_dir}")
            elif args.no_download:
                raise FileNotFoundError(f"{name} not found locally and --no-download set.")
            else:
                partial = os.path.join(hf_cases.DEFAULT_CASES_DIR, name)
                if os.path.isdir(partial):          # incomplete earlier extract
                    B._safe_remove(partial)
                B._log(log_path, f"  downloading case {name} from HuggingFace ...")
                case_dir = hf_cases.ensure_case(name, hf_cases.DEFAULT_CASES_DIR)
                if case_dir is None:
                    raise FileNotFoundError(f"case {name}: HuggingFace download failed")
                downloaded = True

            stats = _run_worker(case_dir)
            row = {"seq": name, "drone_type": drone_map.get(name, "?")}
            row.update(stats)
            _append_row(csv_path, row)
            n_ok += 1
            dt = time.perf_counter() - t0
            B._log(log_path,
                   f"  DONE {name}: frames={stats['total_frames']} "
                   f"gt={stats['gt_frames']} ev_in_gt={stats['total_events_in_gt']} "
                   f"({dt / 60.0:.1f} min)")
        except Exception as exc:
            n_err += 1
            B._log(log_path, f"  SKIPPED {name}: {type(exc).__name__}: {exc}")
        finally:
            if downloaded and case_dir and not args.keep_downloads:
                B._safe_remove(case_dir)            # bound disk: ~1 GB per case

    B._log(log_path, f"END: {n_ok} cases measured, {n_err} skipped.  -> {csv_path}")


# ── Worker entry point ──────────────────────────────────────────────────────────

def _worker_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    stats = compute_case_stats(args.case_dir)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
