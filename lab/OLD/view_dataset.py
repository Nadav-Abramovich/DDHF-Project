#!/usr/bin/env python
"""
Pick a run exported by export_dataset.py and open it in lab_viewer_pooled.py
(the lab copy of code/debug_viewer_pooled.py) with the r80 tracked
working point.

    python view_dataset.py                          # choose from a list
    python view_dataset.py rec_20260916_194601
    python view_dataset.py rec_... --frame 300      # extra flags go to the viewer

The viewer joins --seq onto its own folder, and an absolute path wins that
join, so the run is passed as an absolute path and stays in runs/dataset.

--wp replays the working point through frame_explorer, which reads a cached
feature dump, utility/frame_data/frames_<run>_<mode>.npz (built by
utility/frame_features.py). The list marks runs without one, and picking such
a run offers to build it first - the full DDHF precompute, so minutes, but
only once per run. The run has no GT, so the viewer's scores are empty; the
boxes are what matters here.
"""
import os
import sys
import json
import subprocess

# (kept in lab/OLD - not used by any other script or launcher; it uses the
#  detector modules in ../code)
_HERE = os.path.dirname(os.path.abspath(__file__))           # lab/OLD
_LAB = os.path.dirname(_HERE)                                 # lab/
_CODE = os.path.join(_LAB, "code")                            # lab/code
DATASETS = os.path.join(_LAB, "runs", "dataset")
VIEWER = os.path.join(_CODE, "lab_viewer_pooled.py")
WP = os.path.join(_LAB, "data", "working_points", "working_point_bayes_pixelize_track_r80.json")
FRAME_DATA = os.path.join(_LAB, "frame_data")
FEATURES = os.path.join(_CODE, "common", "frame_features.py")


def list_datasets(root):
    """Folders under root that export_dataset.py produced."""
    if not os.path.isdir(root):
        return []
    return [name for name in sorted(os.listdir(root))
            if os.path.isfile(os.path.join(root, name, "Event", "events.raw"))]


def has_features(name, mode):
    return any(os.path.exists(os.path.join(FRAME_DATA, "frames_%s_%s.%s"
                                           % (name, mode, ext)))
               for ext in ("npz", "csv.gz"))


def build_features(seq, mode):
    """Run utility/frame_features.py on one exported run. Returns its exit code.

    frame_features only accepts a folder that case_statistics counts as a
    case, which needs interpolated_coordinates.txt; runs exported before
    export_dataset.py wrote one get it here, empty (no GT), the same thing
    old staged cases used. --cases-dir points it at runs/dataset,
    --no-download keeps it from fetching FRED cases, and --out-dir writes the
    cache into lab/frame_data, where this script and the lab viewer read it.
    """
    gt = os.path.join(seq, "interpolated_coordinates.txt")
    if not os.path.exists(gt):
        open(gt, "w").close()
        print("created empty %s" % gt)
    cmd = [sys.executable, FEATURES, "--mode", mode,
           "--seqs", os.path.basename(seq),
           "--cases-dir", os.path.dirname(seq), "--no-download",
           "--out-dir", FRAME_DATA]
    print(" ".join('"%s"' % c if " " in c else c for c in cmd))
    return subprocess.call(cmd, cwd=os.path.dirname(FEATURES))


def pick_dataset(root, mode):
    runs = list_datasets(root)
    if not runs:
        raise SystemExit("no exported runs in %s - run export_dataset.py first"
                         % root)
    print("exported runs in %s:" % root)
    for k, name in enumerate(runs):
        print("  %2d  %-34s%s" % (k + 1, name, "" if has_features(name, mode)
                                  else "   [no frame_data for --wp]"))
    while True:
        s = input("run number (enter to quit): ").strip()
        if not s:
            raise SystemExit(0)
        if s.isdigit() and 1 <= int(s) <= len(runs):
            return os.path.join(root, runs[int(s) - 1])
        if s in runs:
            return os.path.join(root, s)
        print("no run %r" % s)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # A leading bare name is the run; everything else is the viewer's.
    run = argv[0] if argv and not argv[0].startswith("-") else None
    extra = argv[1:] if run else argv

    for path in (VIEWER, WP):
        if not os.path.exists(path):
            raise SystemExit("missing %s" % path)
    with open(WP, encoding="utf-8") as fh:
        mode = json.load(fh).get("mode", "baseline")

    if run is None:
        seq = pick_dataset(DATASETS, mode)
    elif os.path.isdir(run):
        seq = run
    else:
        seq = os.path.join(DATASETS, run)
    seq = os.path.abspath(seq)
    if not os.path.isfile(os.path.join(seq, "Event", "events.raw")):
        raise SystemExit("%s is not an exported run (no Event/events.raw)" % seq)
    name = os.path.basename(seq)
    if not has_features(name, mode):
        ans = input("no frame_data for %s (%s) - build it now? this runs the "
                    "full DDHF precompute [Y/n]: " % (name, mode)).strip().lower()
        if ans in ("", "y", "yes"):
            if build_features(seq, mode) != 0 or not has_features(name, mode):
                raise SystemExit("frame_features.py did not produce frames_%s_%s"
                                 " - see %s" % (name, mode, os.path.join(
                                     FRAME_DATA, "frame_features_%s.log" % mode)))
        else:
            print("warning: launching without it - the --wp replay will fail")

    cmd = [sys.executable, VIEWER, "--seq", seq, "--wp", WP] + extra
    print(" ".join('"%s"' % c if " " in c else c for c in cmd))
    # code/ as cwd, as when the original is launched from there; the copy puts
    # code/ on sys.path itself, so this only matters for relative paths.
    return subprocess.call(cmd, cwd=_CODE)


if __name__ == "__main__":
    sys.exit(main())
