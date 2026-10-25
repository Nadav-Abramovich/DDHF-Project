#!/usr/bin/env python
"""
Add and edit GT boxes on a run exported by export_dataset.py.

    python gt_editor.py                          # choose a run from a list
    python gt_editor.py rec_20260927_151030_two_drones

One panel: the webcam frame with the events drawn over it through the run's
registration.npz, exactly as lab_viewer_pooled.py shows them, and the GT boxes
on top. You draw on that; GT is stored in event pixels, so each box is mapped
back through the registration when it is committed.

Keyframes
    Committing a box at frame X makes X a keyframe for that box: the box holds
    from X until the box's next keyframe, or the end of the recording if there
    is none. So set it once, step forward (+1 / +10) until the drone has
    moved, adjust it, commit - that frame onward follows, up to whatever you
    had already set further on. Going back to fix an earlier frame never
    touches a later keyframe. Deleting at X is a keyframe too ("absent from
    here"), so a drone that leaves the view and comes back is: delete where it
    leaves, commit a box where it returns.

Keys
    drag .................... draw a box / move or resize the selected one
    enter ................... commit it to the selected box (a new box if none)
    n ....................... commit it as a NEW box (another drone)
    tab / 1-9 ............... select the next box / box with that id
    x or delete ............. the selected box is absent from here (keyframe)
    r ....................... remove the selected box's keyframe at this frame,
                              so the previous one carries through again
    escape .................. drop the uncommitted drag
    u ....................... undo the last commit / delete / remove
    right / left ............ +1 / -1 frame
    shift+right / shift+left  +10 / -10   (up / down do the same)
    pageup / pagedown ....... +100 / -100     home / end: first / last frame
    [ / ] ................... previous / next keyframe of the selected box
    e ....................... events on/off
    q ....................... quit (every change is already saved)

Files
    Every change rewrites interpolated_coordinates.txt and coordinates.txt in
    the run - one line per frame per box, `t_sec: x1, y1, x2, y2, id` in event
    pixels, the format the viewers and utility/frame_features.py read - at
    t = frame * window. Their contents from before the first save are kept
    once as *.orig. Loading turns such a per-frame file back into keyframes
    (wherever a box changes, appears or disappears). The run's .iou_* caches
    and any gt_<run>.csv.gz under utility/frame_data* are refreshed so nothing
    downstream scores against the old boxes.
"""
import os
import sys
import csv
import glob
import gzip
import json
import shutil
import argparse
from bisect import bisect_right

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "common")):   # code/common: detector copies
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lab_viewer_pooled as LV        # Metavision bootstrap, StreamState, EvToRgb

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox
from matplotlib.patches import Rectangle

DATASETS = os.path.join(os.path.dirname(_HERE), "runs", "dataset")
UTILITY = _HERE                       # lab caches: frame_data, frame_data_variants
GT_FILES = ("interpolated_coordinates.txt", "coordinates.txt")

BOX_COLOR = "yellow"
SEL_COLOR = "cyan"
DRAG_COLOR = "lime"


# ---------------------------------------------------------------------------
# GT model: per box id, keyframes {frame: (x1, y1, x2, y2) or None}, event px
# ---------------------------------------------------------------------------
class GroundTruth:
    def __init__(self, folder, n_frames, window_us):
        self.folder = folder
        self.n = n_frames
        self.window_us = window_us
        self.kf = {}                  # id -> {frame: box, or None = absent}
        self._undo = []
        self.source = None

    # -- queries --------------------------------------------------------------
    def box(self, bid, f):
        """The box in force at frame f: the latest keyframe at or before f."""
        keys = sorted(self.kf.get(bid, {}))
        i = bisect_right(keys, f)
        return self.kf[bid][keys[i - 1]] if i else None

    def boxes(self, f):
        out = {}
        for bid in self.kf:
            b = self.box(bid, f)
            if b is not None:
                out[bid] = b
        return out

    def keyframes(self, bid):
        return sorted(self.kf.get(bid, {}))

    def new_id(self):
        return max(self.kf) + 1 if self.kf else 1

    # -- edits ----------------------------------------------------------------
    def _snapshot(self):
        self._undo.append({b: dict(k) for b, k in self.kf.items()})
        del self._undo[:-50]

    def set(self, f, bid, box):
        self._snapshot()
        self.kf.setdefault(bid, {})[f] = box
        self._tidy(bid)

    def delete(self, f, bid):
        self._snapshot()
        self.kf.setdefault(bid, {})[f] = None
        self._tidy(bid)

    def unkey(self, f, bid):
        if f not in self.kf.get(bid, {}):
            return False
        self._snapshot()
        del self.kf[bid][f]
        self._tidy(bid)
        return True

    def _tidy(self, bid):
        """Drop keyframes that change nothing (same as what precedes them), and
        the id itself once it never shows."""
        k = self.kf[bid]
        prev = None
        for f in sorted(k):
            if k[f] == prev:
                del k[f]
            else:
                prev = k[f]
        if all(v is None for v in k.values()):
            del self.kf[bid]

    def undo(self):
        if not self._undo:
            return False
        self.kf = self._undo.pop()
        return True

    # -- disk -----------------------------------------------------------------
    def load(self):
        """Read the first non-empty GT file (5th column = the box's id) and
        turn its per-frame lines into keyframes."""
        for name in GT_FILES:
            path = os.path.join(self.folder, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
        else:
            return 0
        self.source = name
        per_frame = {}
        n = 0
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                t_str, sep, rest = line.partition(":")
                if not sep:
                    continue
                try:
                    t = float(t_str)
                    vals = [float(v) for v in rest.replace(",", " ").split()]
                except ValueError:
                    continue
                if len(vals) < 4:
                    continue
                f = int(round(t * 1e6 / self.window_us))
                if not 0 <= f < self.n:
                    continue
                d = per_frame.setdefault(f, {})
                bid = int(vals[4]) if len(vals) > 4 else len(d) + 1
                while bid in d:                      # repeated id: keep both
                    bid += 1
                d[bid] = tuple(int(round(v)) for v in vals[:4])
                n += 1
        ids = sorted({b for d in per_frame.values() for b in d})
        for bid in ids:
            prev = None
            k = {}
            for f in range(self.n):
                cur = per_frame.get(f, {}).get(bid)
                if cur != prev:
                    k[f] = cur
                    prev = cur
            self.kf[bid] = k
        return n

    def save(self):
        lines = []
        for f in range(self.n):
            t = f * self.window_us / 1e6
            for bid, (x1, y1, x2, y2) in sorted(self.boxes(f).items()):
                lines.append("%.6f: %d, %d, %d, %d, %d\n" % (t, x1, y1, x2, y2, bid))
        for name in GT_FILES:
            path = os.path.join(self.folder, name)
            orig = path + ".orig"
            if (os.path.exists(path) and os.path.getsize(path) > 0
                    and not os.path.exists(orig)):
                shutil.copy2(path, orig)             # what was there before us, once
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            os.replace(tmp, path)
        self._refresh_caches()
        return len(lines)

    def _refresh_caches(self):
        """Stale copies of the GT: the viewer's IoU caches go; the frame_data
        GT tables are rewritten on their own frame grid (frame_features.py's
        rule: frame = round(t_us / window_us))."""
        for stale in glob.glob(os.path.join(self.folder, ".iou_*.npz")):
            os.remove(stale)
        name = os.path.basename(os.path.normpath(self.folder))
        for gt_csv in (glob.glob(os.path.join(UTILITY, "frame_data", "gt_%s.csv.gz" % name))
                       + glob.glob(os.path.join(UTILITY, "frame_data_variants", "*",
                                                "gt_%s.csv.gz" % name))):
            wus = self.window_us
            for meta in glob.glob(os.path.join(os.path.dirname(gt_csv),
                                               "meta_%s_*.json" % name)):
                try:
                    with open(meta, encoding="utf-8") as fh:
                        wus = int(json.load(fh).get("window_us", wus))
                    break
                except Exception:
                    pass
            tmp = gt_csv + ".tmp"
            with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "x1", "y1", "x2", "y2"])
                for f in range(self.n):
                    fidx = int(round(f * self.window_us / wus))
                    for bid, box in sorted(self.boxes(f).items()):
                        w.writerow([fidx] + list(box))
            os.replace(tmp, gt_csv)


# ---------------------------------------------------------------------------
# Choosing the run
# ---------------------------------------------------------------------------
def list_datasets(root):
    if not os.path.isdir(root):
        return []
    return [n for n in sorted(os.listdir(root))
            if os.path.isfile(os.path.join(root, n, "Event", "events.raw"))]


def gt_lines(folder):
    for name in GT_FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                return sum(1 for line in fh if ":" in line)
    return 0


def pick_dataset(root):
    runs = list_datasets(root)
    if not runs:
        raise SystemExit("no exported runs in %s - run export_dataset.py first" % root)
    print("exported runs in %s:" % root)
    for k, name in enumerate(runs):
        n = gt_lines(os.path.join(root, name))
        print("  %2d  %-40s %s" % (k + 1, name, "%d GT boxes" % n if n else "no GT"))
    while True:
        s = input("run number (enter to quit): ").strip()
        if not s:
            raise SystemExit(0)
        if s.isdigit() and 1 <= int(s) <= len(runs):
            return os.path.join(root, runs[int(s) - 1])
        if s in runs:
            return os.path.join(root, s)
        print("no run %r" % s)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run under runs/dataset, or a path")
    p.add_argument("--frame", type=int, default=0, help="frame to open at")
    p.add_argument("--window-ms", type=float, default=33.333,
                   help="frame length; the viewers' default (33.333)")
    p.add_argument("--rgb-offset-ms", type=float, default=None,
                   help="as lab_viewer_pooled.py: RGB starts this many ms after "
                        "the events (default: lab/data/rgb_offsets/rgb_offsets.json, else 0)")
    args = p.parse_args(argv)

    if args.run is None:
        folder = pick_dataset(DATASETS)
    elif os.path.isdir(args.run):
        folder = args.run
    else:
        folder = os.path.join(DATASETS, args.run)
    folder = os.path.abspath(folder)
    raw = os.path.join(folder, "Event", "events.raw")
    if not os.path.isfile(raw):
        raise SystemExit("%s is not an exported run (no Event/events.raw)" % folder)
    name = os.path.basename(folder)

    window_us = int(round(args.window_ms * 1000.0))
    rgb_off_us = (args.rgb_offset_ms * 1000.0 if args.rgb_offset_ms is not None
                  else LV._sidecar_rgb_offset_us(folder))
    stream = LV.StreamState(raw, os.path.join(folder, "PADDED_RGB"),
                            rgb_offset_us=rgb_off_us)
    ev_w, ev_h = stream.ev_width, stream.ev_height
    max_idx = stream.max_frame_idx(window_us)

    P, note = LV.EvToRgb.load(folder, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                               dtype=np.float32))
    print("[registration] " + note)
    fov_pts = P.outline(ev_w, ev_h)

    gt = GroundTruth(folder, max_idx + 1, window_us)
    n = gt.load()
    print("GT: %s" % ("%d boxes from %s -> %d keyframes"
                      % (n, gt.source, sum(len(k) for k in gt.kf.values()))
                      if n else "none yet"))

    # Our keys, not matplotlib's toolbar shortcuts (s = save figure, q = quit,
    # left/right = view history, ...). The toolbar buttons (zoom) still work.
    for k in list(plt.rcParams):
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    st = dict(frame=max(0, min(args.frame, max_idx)), sel=None, pending=None,
              events_on=True, base=None, evs=None)

    fig = plt.figure(figsize=(16, 9.5))
    ax = fig.add_axes([0.01, 0.08, 0.98, 0.86])
    ax.set_xticks([]); ax.set_yticks([])
    fig.text(
        0.5, 0.008,
        "drag: draw/adjust   enter: commit   n: new box   tab/1-9: select   "
        "x/del: absent from here   r: remove keyframe   esc: drop drag   u: undo\n"
        "left/right +-1   shift+arrow or up/down +-10   pgup/pgdn +-100   "
        "[ / ]: prev/next keyframe   e: events   q: quit",
        ha="center", va="bottom", family="monospace", fontsize=8)
    status = fig.text(0.01, 0.955, "", family="monospace", fontsize=10)
    tb_ax = fig.add_axes([0.905, 0.95, 0.05, 0.03])
    frame_box = TextBox(tb_ax, "frame (0..%d) " % max_idx, initial=str(st["frame"]))

    base, evs, _ = stream.load_window(st["frame"], window_us, verbose=False)
    H, W = base.shape[:2]
    im = ax.imshow(base)
    patches = []

    # -- webcam px <-> event px ------------------------------------------------
    def ev_to_ext(box):
        x1, y1, x2, y2 = LV._project_box_to_rgb(box, P, W, H)
        return (x1, x2, y1, y2)

    def ext_to_ev(x1, x2, y1, y2):
        """A rectangle on the webcam frame -> the event-pixel box enclosing it.
        Edges are sampled, since lens correction bends them; points off the
        event sensor are dropped."""
        t = np.linspace(0.0, 1.0, 5)
        pts = ([(x1 + (x2 - x1) * s, y1) for s in t] + [(x1 + (x2 - x1) * s, y2) for s in t]
               + [(x1, y1 + (y2 - y1) * s) for s in t] + [(x2, y1 + (y2 - y1) * s) for s in t])
        hits = [P.inv(int(round(x)), int(round(y))) for x, y in pts]
        hits = [h for h in hits if h is not None]
        if len(hits) < 2:
            return None
        xs = [h[0] for h in hits]; ys = [h[1] for h in hits]
        bx1, bx2 = max(0, min(xs)), min(ev_w - 1, max(xs))
        by1, by2 = max(0, min(ys)), min(ev_h - 1, max(ys))
        if bx2 <= bx1 or by2 <= by1:
            return None
        return int(bx1), int(by1), int(bx2), int(by2)

    def on_select(_press, _release):
        box = ext_to_ev(*selector.extents)
        if box is None:
            refresh_status("that box is outside the event camera's view")
            return
        st["pending"] = box
        selector.extents = ev_to_ext(box)       # snap to what will be stored
        refresh_status()

    selector = RectangleSelector(ax, on_select, interactive=True, button=[1],
                                 minspanx=4, minspany=4, spancoords="data",
                                 props=dict(edgecolor=DRAG_COLOR, fill=False,
                                            linewidth=1.5),
                                 useblit=False)

    def load_selection():
        """Put the selected box (if any) under the selector, ready to adjust."""
        st["pending"] = None
        box = gt.boxes(st["frame"]).get(st["sel"])
        selector.set_visible(box is not None)
        if box is not None:
            selector.extents = ev_to_ext(box)

    # -- drawing ----------------------------------------------------------------
    def refresh_status(msg=""):
        f = st["frame"]
        ids = sorted(gt.boxes(f))
        sel = st["sel"]
        key_s = ""
        if sel is not None:
            ks = gt.keyframes(sel)
            key_s = "   box %d keyframes: %s%s" % (
                sel, ",".join(map(str, ks[:12])), " ..." if len(ks) > 12 else "")
            if f in ks:
                key_s += "  (keyframe here)"
        status.set_text(
            "%s   frame %d/%d  t=%.3fs   boxes: %s   selected: %s%s%s%s"
            % (name, f, max_idx, f * window_us / 1e6,
               ",".join(map(str, ids)) or "none",
               sel if sel is not None else "-", key_s,
               "   [uncommitted - enter / n]" if st["pending"] is not None else "",
               ("   " + msg) if msg else ""))
        fig.canvas.draw_idle()

    def draw_boxes():
        for patch in patches:
            patch.remove()
        patches.clear()
        f = st["frame"]
        for bid, box in sorted(gt.boxes(f).items()):
            color = SEL_COLOR if bid == st["sel"] else BOX_COLOR
            x1, y1, x2, y2 = LV._project_box_to_rgb(box, P, W, H)
            dashed = f not in gt.kf.get(bid, {})      # carried from an earlier key
            patches.append(ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                linewidth=2.0 if bid == st["sel"] else 1.3,
                linestyle="--" if dashed else "-")))
            patches.append(ax.text(x1, y1 - 4, str(bid), color=color, fontsize=10,
                                   va="bottom", fontweight="bold"))

    def render():
        rgb = (LV.render_overlay(st["base"], st["evs"], P, fov_pts, None)
               if st["events_on"] else LV.render_plain(st["base"]))
        im.set_data(rgb)
        draw_boxes()

    def go(f, msg=""):
        f = max(0, min(int(f), max_idx))
        try:
            base, evs, _ = stream.load_window(f, window_us, verbose=False)
        except IndexError as exc:
            refresh_status(str(exc))
            return
        st.update(frame=f, base=base, evs=evs)
        ids = sorted(gt.boxes(f))
        if st["sel"] not in ids and st["sel"] not in gt.kf:
            st["sel"] = ids[0] if ids else None
        load_selection()
        render()
        frame_box.eventson = False
        frame_box.set_val(str(f))
        frame_box.eventson = True
        refresh_status(msg)

    def after_edit(msg):
        gt.save()
        if st["sel"] not in gt.kf:
            ids = sorted(gt.boxes(st["frame"]))
            st["sel"] = ids[0] if ids else None
        load_selection()
        draw_boxes()
        refresh_status(msg + " (saved)")

    # -- edits (each one saved at once) -----------------------------------------
    def commit(as_new):
        box = st["pending"]
        if box is None:
            refresh_status("nothing to commit - drag a box first")
            return
        f = st["frame"]
        bid = gt.new_id() if (as_new or st["sel"] is None) else st["sel"]
        gt.set(f, bid, box)
        st["sel"] = bid
        nxt = [k for k in gt.keyframes(bid) if k > f]
        after_edit("box %d keyed at %d, holds to %s"
                   % (bid, f, "frame %d" % (nxt[0] - 1) if nxt else "the end"))

    def delete():
        bid = st["sel"]
        if bid is None or gt.box(bid, st["frame"]) is None:
            refresh_status("no selected box in this frame")
            return
        gt.delete(st["frame"], bid)
        after_edit("box %d absent from frame %d" % (bid, st["frame"]))

    def unkey():
        bid = st["sel"]
        if bid is None or not gt.unkey(st["frame"], bid):
            refresh_status("the selected box has no keyframe at this frame")
            return
        after_edit("box %d keyframe at %d removed" % (bid, st["frame"]))

    def undo():
        if not gt.undo():
            refresh_status("nothing to undo")
            return
        after_edit("undone")

    def select(bid):
        if bid not in gt.boxes(st["frame"]):
            refresh_status("no box %s in this frame" % bid)
            return
        st["sel"] = bid
        load_selection()
        draw_boxes()
        refresh_status()

    def jump_key(direction):
        if st["sel"] is None:
            return
        ks = gt.keyframes(st["sel"])
        f = st["frame"]
        cand = [k for k in ks if k > f] if direction > 0 else [k for k in ks if k < f]
        if cand:
            go(cand[0] if direction > 0 else cand[-1])

    STEPS = {"right": 1, "left": -1, "shift+right": 10, "shift+left": -10,
             "up": 10, "down": -10, "pageup": 100, "pagedown": -100}

    def on_key(event):
        if frame_box.capturekeystrokes:            # typing a frame number
            return
        k = event.key
        if k in STEPS:
            go(st["frame"] + STEPS[k])
        elif k == "home":
            go(0)
        elif k == "end":
            go(max_idx)
        elif k == "enter":
            commit(as_new=False)
        elif k == "n":
            commit(as_new=True)
        elif k == "tab":
            ids = sorted(gt.boxes(st["frame"]))
            if ids:
                i = ids.index(st["sel"]) + 1 if st["sel"] in ids else 0
                select(ids[i % len(ids)])
        elif k is not None and len(k) == 1 and k.isdigit() and k != "0":
            select(int(k))
        elif k in ("x", "delete"):
            delete()
        elif k == "r":
            unkey()
        elif k == "escape":
            load_selection()
            refresh_status("drag dropped")
        elif k == "u":
            undo()
        elif k == "[":
            jump_key(-1)
        elif k == "]":
            jump_key(+1)
        elif k == "e":
            st["events_on"] = not st["events_on"]
            render()
            refresh_status()
        elif k == "q":
            plt.close(fig)

    def on_frame_submit(text):
        try:
            go(int(text.strip()))
        except ValueError:
            refresh_status("not a frame number: %r" % text)

    frame_box.on_submit(on_frame_submit)
    fig.canvas.mpl_connect("key_press_event", on_key)
    go(st["frame"])
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
