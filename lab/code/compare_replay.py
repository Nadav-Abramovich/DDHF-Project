#!/usr/bin/env python
"""
Show a dual_record.py folder three ways at once: the RGB video and the event
stream side by side on top, and the two composited into a single overlaid
frame below.

    python compare_replay.py runs/rec_20260916_194601
    python compare_replay.py runs/rec_... --window-ms 3 --panel-h 300

    +-----------------------+-----------------------+
    |      webcam (RGB)     |     event camera      |
    +-----------------------+-----------------------+
    |            events drawn over RGB              |
    +-----------------------------------------------+

Keys
    SPACE .... play / pause          left / right ... step 1 frame
    pgup/pgdn  step 1 second         home / end ..... first / last frame
    , / . .... event window -/+ 1 ms a / z .......... opacity -/+
    [ / ] .... playback speed -/+    s .............. save the canvas
    u / c .... undo / clear pairs    q / ESC ........ quit
    w ........ write pairs to disk   v / e .......... undistort webcam / events

Mouse - registering the two cameras
    Click the same physical point in both top panels (either order) and the
    two clicks become a correspondence pair. The mapping from event pixels to
    video pixels is refit after every pair and used for the bottom overlay,
    and the event sensor's outline transformed by it - the target area - is
    drawn on the bottom panel so you can see where the event frame is landing.
    Right-click cancels a half-finished pair; u drops the last pair; c clears.

    What gets fit depends on how many pairs there are, because a fit needs
    points to constrain it and a wrong fit off two clicks is worse than none:

        0 pairs   stretch the event frame onto the video frame per axis
                  (640x480 -> 640x360, so y by 0.75 and x untouched)
        1 pair    that stretch, shifted to put the pair on top of itself
        2 pairs   independent x and y scale plus shift. Not a similarity:
                  these sensors differ in aspect, and a uniform scale could
                  not express that
        3 pairs   full affine, least squares
        4+ pairs  homography, least squares over every pair (not RANSAC -
                  you placed these by hand, so none of them get discarded)

    The header prints the model in force and its RMS residual in video pixels.
    Spread the pairs over the frame and over depth; four points on one plane
    only register that plane.

Toolbar (top row of the window; each button also has a key)
    [x] undistort webcam (v)   [x] undistort events (e)
    save pairs (w)   load pairs   undo pair (u)   clear pairs (c)

    save writes the pairs to <folder>/pairs.csv, and they are loaded from
    there automatically the next time the folder is opened; load re-reads the
    file, discarding unsaved changes. "save pairs *" and the header's
    "*unsaved" mark pairs on screen that differ from the file. A checkbox is
    greyed out when its .calib is missing or not filled in.

LENS undistortion
    vis.calib and event.calib (in the recording's parent folder - runs/ - or
    --calib-dir) hold each camera's intrinsics and distortion; the templates
    there document the format. v and e toggle undistorting the webcam frame
    and the event coordinates independently (so do the toolbar checkboxes;
    --undistort starts with both on); a camera whose file is missing or not
    filled in just stays raw.

    Pairs are always stored in raw, distorted pixels and converted to
    whichever view is on screen, so the saved file does not depend on the
    toggles and a pair clicked in one view stays put in the other. The fit is
    redone in the current view: with both cameras undistorted a homography is
    the right model for a plane, where lens distortion would otherwise bend it.

TIME alignment
    Event timestamps are sensor-relative microseconds from the start of the
    RAW; webcam frames carry wall-clock times in camera_frames.csv; meta.txt
    records the wall clock at which each stream started. So for webcam frame i

        event_time_us = (csv_stamp[i] - meta["events started"]) * 1e6

    and the events shown are those within +-window/2 of it. The .mp4's nominal
    frame rate is a container field, not a measurement (30 claimed, ~142 real
    here), so frame_number / fps drifts badly and is only the fallback when the
    CSV or meta.txt is missing - the header says so when that happens.

Playback speed
    "1x" means the measured webcam rate from meta.txt, which at ~142 fps is
    faster than the decode-plus-composite loop can keep up with; the loop then
    simply runs as fast as it can. Drop the speed with [ to actually watch it.

The window is resizable, and resizing it does not spoil the picking: highgui
rescales mouse coordinates from window pixels back to image pixels before the
callback sees them, so a click still lands where you aimed it. What it does
cost is precision - one image pixel is a fraction of a screen pixel in a
shrunken window - so pick your points with the window at or above its natural
size. --panel-h and --max-width set that natural size.
"""
import os
import sys
import csv
import argparse

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "camera")):    # camera/: silky_bias_studio
    if _p not in sys.path:
        sys.path.insert(0, _p)
import silky_bias_studio as studio            # DLL/plugin bootstrap
from metavision_core.event_io import RawReader

WIN = "camera | events  /  overlay"
ON_COLOR = (255, 255, 255)      # BGR
OFF_COLOR = (0, 0, 255)
LABEL_COLOR = (0, 255, 0)
PAIR_COLOR = (0, 255, 255)      # accepted correspondences
PENDING_COLOR = (255, 0, 255)   # one half of a pair, waiting for the other
AREA_COLOR = (0, 255, 255)      # transformed event-sensor outline
SEP_COLOR = (60, 60, 60)
SEP_PX = 2
BAR_H = 34                      # toolbar row
BAR_BG = (40, 40, 40)
BTN_BG = (75, 75, 75)
BTN_FG = (235, 235, 235)
BTN_OFF_FG = (120, 120, 120)    # disabled button
CHECK_COLOR = (0, 255, 0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_events(path):
    """Whole RAW into memory - lab recordings are short and seeking wins."""
    reader = RawReader(path, max_events=200_000_000)
    height, width = reader.get_size()
    chunks = []
    while not reader.is_done():
        e = reader.load_delta_t(1_000_000)
        if e.size:
            chunks.append(e.copy())
    if not chunks:
        raise SystemExit("no events in %s" % path)
    evs = np.concatenate(chunks)
    return evs, evs["t"].astype(np.int64), width, height


def load_stamps(folder):
    """Wall-clock time of each webcam frame, or None if the CSV is absent."""
    path = os.path.join(folder, "camera_frames.csv")
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append(float(row["wall_clock_s"]))
    return np.array(out) if out else None


def load_meta(folder):
    """The few numbers we need out of meta.txt, keyed by their label."""
    path = os.path.join(folder, "meta.txt")
    meta = {}
    if not os.path.exists(path):
        return meta
    for line in open(path):
        for key in ("events started", "webcam started", "measured fps"):
            if line.startswith(key):
                try:
                    meta[key] = float(line.split()[-1])
                except ValueError:
                    pass
    return meta


PAIRS_FILE = "pairs.csv"
PAIR_FIELDS = ("ev_x", "ev_y", "vid_x", "vid_y")


def load_pairs(folder):
    """Saved correspondences, in raw (distorted) pixels, or [] if none."""
    path = os.path.join(folder, PAIRS_FILE)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            v = [float(row[k]) for k in PAIR_FIELDS]
            out.append(((v[0], v[1]), (v[2], v[3])))
    return out


def save_pairs(folder, pairs):
    path = os.path.join(folder, PAIRS_FILE)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(PAIR_FIELDS)
        for (ex, ey), (vx, vy) in pairs:
            wr.writerow(["%.3f" % v for v in (ex, ey, vx, vy)])
    return path


# ---------------------------------------------------------------------------
# Lens calibration
# ---------------------------------------------------------------------------
class Calib(object):
    """One camera's intrinsics and distortion, already scaled to its image.

    Undistorted pixels keep the same K, so the picture keeps its size and
    centre and only the bending comes out.
    """

    def __init__(self, K, D, fisheye):
        self.K = K
        self.D = D
        self.fisheye = fisheye

    def undistort(self, pts):
        """Raw pixels -> undistorted pixels, Nx2 in and out."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if not len(pts):
            return pts.reshape(-1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.K)
        elif hasattr(cv2, "undistortPointsIter"):
            # The plain call stops after 5 iterations, which leaves strong
            # barrel distortion visibly under-corrected at the edges.
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)
            out = cv2.undistortPointsIter(pts, self.K, self.D, None, self.K,
                                          crit)
        else:
            out = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        return out.reshape(-1, 2)

    def distort(self, pts):
        """Undistorted pixels -> raw pixels, the inverse of undistort."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if not len(pts):
            return pts
        K = self.K
        obj = np.column_stack([(pts[:, 0] - K[0, 2]) / K[0, 0],
                               (pts[:, 1] - K[1, 2]) / K[1, 1],
                               np.ones(len(pts))]).reshape(-1, 1, 3)
        zero = np.zeros((3, 1))
        if self.fisheye:
            out, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, self.D)
        else:
            out, _ = cv2.projectPoints(obj, zero, zero, K, self.D)
        return out.reshape(-1, 2)

    def maps(self, w, h):
        """remap() tables that undistort a whole w x h image."""
        if self.fisheye:
            return cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (w, h), cv2.CV_16SC2)
        return cv2.initUndistortRectifyMap(self.K, self.D, None, self.K,
                                           (w, h), cv2.CV_16SC2)


def load_calib(path, width, height):
    """Parse a .calib file for a width x height image.

    Returns (Calib or None, note); None whenever the file is missing, broken
    or still the unfilled template, with the note saying which.
    """
    if not os.path.exists(path):
        return None, "%s missing - stays raw" % path
    vals = {}
    for line in open(path):
        tok = line.split("#", 1)[0].split()
        if len(tok) >= 2:
            vals[tok[0].lower()] = tok[1:]
    try:
        fx, fy, cx, cy = (float(vals[k][0]) for k in ("fx", "fy", "cx", "cy"))
        cw = float(vals["width"][0]) if "width" in vals else float(width)
        ch = float(vals["height"][0]) if "height" in vals else float(height)
        dist = [float(v) for v in vals.get("dist", [])]
    except (KeyError, ValueError) as exc:
        return None, "%s: bad or missing field %s - stays raw" % (path, exc)
    model = vals.get("model", ["pinhole"])[0].lower()
    if fx <= 0 or fy <= 0:
        return None, "%s not filled in (fx/fy = 0) - stays raw" % path
    if model == "fisheye":
        if len(dist) != 4:
            return None, "%s: fisheye needs 4 dist values" % path
    elif model == "pinhole":
        if len(dist) not in (4, 5, 8, 12, 14):
            return None, "%s: pinhole needs 4/5/8/12/14 dist values" % path
    else:
        return None, "%s: unknown model %r" % (path, model)
    sx, sy = width / cw, height / ch
    K = np.array([[fx * sx, 0.0, cx * sx],
                  [0.0, fy * sy, cy * sy],
                  [0.0, 0.0, 1.0]])
    note = "%s: %s fx %.1f fy %.1f" % (path, model, K[0, 0], K[1, 1])
    if sx != 1.0 or sy != 1.0:
        note += "  (rescaled from %dx%d)" % (cw, ch)
    return Calib(K, np.array(dist, float).reshape(-1, 1), model == "fisheye"), note


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _affine_lstsq(src, dst):
    """Least-squares affine as a 3x3, or None if the points are degenerate."""
    design = np.column_stack([src, np.ones(len(src))])
    try:
        cx, _r, rank, _s = np.linalg.lstsq(design, dst[:, 0], rcond=None)
        cy, _r, _rk, _s = np.linalg.lstsq(design, dst[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:                      # collinear clicks: no unique affine
        return None
    return np.array([cx, cy, [0.0, 0.0, 1.0]], float)


def fit_transform(pairs, base_sx, base_sy):
    """Event-pixel -> video-pixel 3x3, degrading with the number of pairs.

    Returns (matrix, name). Each step up needs the points that make it
    identifiable; fitting a homography to two clicks would just be a confident
    way of being wrong, so the model grows only as the evidence does.
    """
    base = np.array([[base_sx, 0.0, 0.0],
                     [0.0, base_sy, 0.0],
                     [0.0, 0.0, 1.0]])
    n = len(pairs)
    if n == 0:
        return base, "aspect fit"

    src = np.array([p[0] for p in pairs], float)
    dst = np.array([p[1] for p in pairs], float)

    if n <= 2:
        out = base.copy()
        for ax in (0, 1):
            s = base[ax, ax]
            if n == 2:
                span = src[1, ax] - src[0, ax]
                if abs(span) > 1e-6:          # else that axis is unconstrained
                    s = (dst[1, ax] - dst[0, ax]) / span
            out[ax, ax] = s
            out[ax, 2] = dst[0, ax] - s * src[0, ax]
        return out, "1 pt shift" if n == 1 else "2 pt scale+shift"

    if n >= 4:
        try:
            H, _ = cv2.findHomography(src.astype(np.float32),
                                      dst.astype(np.float32), 0)
        except cv2.error:
            H = None
        if H is not None and np.all(np.isfinite(H)) and abs(H[2, 2]) > 1e-12:
            return H / H[2, 2], "homography %d pts" % n

    aff = _affine_lstsq(src, dst)
    if aff is not None:
        return aff, "affine %d pts" % n
    return base, "aspect fit (degenerate pts)"


def apply_xform(M, x, y):
    """Project arrays (or scalars) of event coordinates through M."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    den = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    return ((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / den,
            (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / den)


def rms_residual(M, pairs):
    if not pairs:
        return 0.0
    src = np.array([p[0] for p in pairs], float)
    dst = np.array([p[1] for p in pairs], float)
    px, py = apply_xform(M, src[:, 0], src[:, 1])
    return float(np.sqrt(np.mean((px - dst[:, 0]) ** 2 + (py - dst[:, 1]) ** 2)))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _paint(img, xs, ys, colour, dot, w, h):
    for oy in range(dot):
        for ox in range(dot):
            px, py = xs + ox, ys + oy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            img[py[ok], px[ok]] = colour


def event_image(ex, ey, on, ev_w, ev_h, bg):
    """The event slice on its own, at sensor resolution.

    ex/ey are the (possibly undistorted, so fractional) event coordinates;
    undistortion pushes some of them off the sensor, and those are dropped.
    """
    img = np.full((ev_h, ev_w, 3), bg, np.uint8)
    if len(ex):
        xs = np.rint(ex).astype(np.int32)
        ys = np.rint(ey).astype(np.int32)
        _paint(img, xs[on], ys[on], ON_COLOR, 1, ev_w, ev_h)
        _paint(img, xs[~on], ys[~on], OFF_COLOR, 1, ev_w, ev_h)
    return img


def draw_layer(frame, ex, ey, on, M, alpha, dot):
    """Paint events through M onto a copy of the frame, then blend it back.

    Blending the whole copy rather than building a mask is deliberate: pixels
    with no event are identical in both images, so addWeighted leaves them
    untouched and only the painted ones move toward the overlay colour.
    """
    h, w = frame.shape[:2]
    over = frame.copy()
    if len(ex):
        fx, fy = apply_xform(M, ex, ey)
        vx = fx.astype(np.int32)
        vy = fy.astype(np.int32)
        _paint(over, vx[on], vy[on], ON_COLOR, dot, w, h)
        _paint(over, vx[~on], vy[~on], OFF_COLOR, dot, w, h)
    return cv2.addWeighted(over, alpha, frame, 1.0 - alpha, 0.0)


def draw_target_area(img, M, ev_w, ev_h, scale):
    """Outline of the event sensor mapped through M, in panel coordinates."""
    corners = np.array([[0, 0], [ev_w - 1, 0], [ev_w - 1, ev_h - 1],
                        [0, ev_h - 1]], float)
    fx, fy = apply_xform(M, corners[:, 0], corners[:, 1])
    pts = np.column_stack([fx * scale, fy * scale]).astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, AREA_COLOR, 1, cv2.LINE_AA)
    corner = [(int(px), int(py)) for px, py in pts]   # cv2 rejects np scalars
    cv2.line(img, corner[0], corner[2], AREA_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, corner[1], corner[3], AREA_COLOR, 1, cv2.LINE_AA)


def draw_pair_marks(img, pts, scale, colour, tags=True):
    for k, (px, py) in enumerate(pts):
        c = (int(round(px * scale)), int(round(py * scale)))
        cv2.drawMarker(img, c, (0, 0, 0), cv2.MARKER_CROSS, 13, 3)
        cv2.drawMarker(img, c, colour, cv2.MARKER_CROSS, 11, 1)
        cv2.circle(img, c, 7, colour, 1, cv2.LINE_AA)
        if tags:
            cv2.putText(img, str(k + 1), (c[0] + 9, c[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(img, str(k + 1), (c[0] + 9, c[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def fit_height(img, height):
    """Resize to a target height, fattening single pixels when shrinking.

    A one-pixel event survives INTER_AREA as a faint smear and INTER_NEAREST
    not at all, so dilate first when the panel is smaller than the source -
    the point of this panel is to see that the events are there.
    """
    h, w = img.shape[:2]
    if h == height:
        return img
    ratio = height / float(h)
    if ratio < 0.9:
        k = int(np.ceil(1.0 / ratio))
        img = cv2.dilate(img, np.ones((k, k), np.uint8))
    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_NEAREST
    return cv2.resize(img, (max(1, int(round(w * ratio))), height),
                      interpolation=interp)


def draw_toolbar(width, items):
    """The button row, drawn by hand - highgui's own buttons need a Qt build.

    items are (id, text, is_checkbox, checked, enabled). Returns the image and
    [(id, x0, x1)] for the mouse callback to hit-test against.
    """
    bar = np.full((BAR_H, width, 3), BAR_BG, np.uint8)
    font, fs, box = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 14
    hits = []
    x = 6
    for ident, text, checkbox, checked, enabled in items:
        (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
        x0, x1 = x, x + tw + 16 + (box + 6 if checkbox else 0)
        fg = BTN_FG if enabled else BTN_OFF_FG
        cv2.rectangle(bar, (x0, 5), (x1, BAR_H - 6), BTN_BG, -1)
        cv2.rectangle(bar, (x0, 5), (x1, BAR_H - 6), fg, 1)
        tx = x0 + 8
        if checkbox:
            by = (BAR_H - box) // 2
            cv2.rectangle(bar, (tx, by), (tx + box, by + box), fg, 1)
            if checked:
                cv2.rectangle(bar, (tx + 3, by + 3),
                              (tx + box - 3, by + box - 3), CHECK_COLOR, -1)
            tx += box + 6
        cv2.putText(bar, text, (tx, (BAR_H + th) // 2), font, fs, fg, 1,
                    cv2.LINE_AA)
        if enabled:
            hits.append((ident, x0, x1))
        x = x1 + 8
    return bar, hits


def label(img, text, org=(8, 20)):
    for colour, thick in (((0, 0, 0), 3), (LABEL_COLOR, 1)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, thick, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="a rec_* folder written by dual_record.py")
    p.add_argument("--window-ms", type=float, default=10.0,
                   help="milliseconds of events to draw per frame (default 10; "
                        "2-3 resolves individual blades)")
    p.add_argument("--opacity", type=float, default=0.85,
                   help="overlay opacity 0-1 in the bottom panel (default 0.85)")
    p.add_argument("--dot", type=int, default=1,
                   help="event dot size in video pixels (default 1; use 2 when "
                        "the video is much larger than the event sensor)")
    p.add_argument("--panel-h", type=int, default=360,
                   help="height of each top panel in pixels (default 360)")
    p.add_argument("--max-width", type=int, default=1900,
                   help="shrink the whole canvas past this width (default 1900)")
    p.add_argument("--bg", type=int, default=0,
                   help="background grey of the event panel, 0-255 (default 0)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="rate that counts as 1x speed; 0 uses the measured rate "
                        "from meta.txt")
    p.add_argument("--speed", type=float, default=1.0,
                   help="initial playback speed multiplier (default 1.0)")
    p.add_argument("--calib-dir", default=None,
                   help="folder holding vis.calib and event.calib (default: "
                        "the recording's parent, i.e. runs/)")
    p.add_argument("--undistort", action="store_true",
                   help="start with both cameras undistorted (each only if "
                        "its .calib is filled in)")
    args = p.parse_args(argv)

    folder = args.folder
    raw = os.path.join(folder, "events.raw")
    vid = os.path.join(folder, "camera.mp4")
    for path in (raw, vid):
        if not os.path.exists(path):
            raise SystemExit("missing %s" % path)

    evs, ts, ev_w, ev_h = load_events(raw)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % vid)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stamps = load_stamps(folder)
    meta = load_meta(folder)
    t_events = meta.get("events started")
    synced = stamps is not None and t_events is not None
    if synced:
        frame_us = ((stamps - t_events) * 1e6).astype(np.int64) + int(ts[0])
        n_frames = min(n_frames, len(frame_us))
    else:
        frame_us = (int(ts[0])
                    + (np.arange(max(n_frames, 1)) / vid_fps * 1e6).astype(np.int64))
    if n_frames < 1:
        raise SystemExit("no video frames in %s" % vid)

    print("events %d over %.2f s (%dx%d) | video %d frames @ %.2f fps | %s"
          % (len(evs), (ts[-1] - ts[0]) / 1e6, ev_w, ev_h, n_frames, vid_fps,
             "CSV-synced" if synced else "NO SYNC DATA - assuming frame/fps"))
    print("click the same point in both top panels to add a correspondence; "
          "u undoes, c clears")

    # Per-axis, not one ratio: 640x480 of event sensor has to land on 640x360
    # of video, so y is squeezed by 0.75 where x is left alone. This is the
    # starting guess the clicked pairs then replace.
    base_sx = vid_w / float(ev_w)
    base_sy = vid_h / float(ev_h)
    vid_panel_scale = args.panel_h / float(vid_h)
    ev_panel_scale = args.panel_h / float(ev_h)

    calib_dir = args.calib_dir or os.path.dirname(os.path.abspath(folder))
    vis_cal, note = load_calib(os.path.join(calib_dir, "vis.calib"),
                               vid_w, vid_h)
    print("vis calib   " + note)
    ev_cal, note = load_calib(os.path.join(calib_dir, "event.calib"),
                              ev_w, ev_h)
    print("event calib " + note)
    vis_maps = vis_cal.maps(vid_w, vid_h) if vis_cal else None
    ev_lut = None
    if ev_cal:
        # Every sensor pixel undistorted once, so per-frame it is a lookup.
        gx, gy = np.meshgrid(np.arange(ev_w), np.arange(ev_h))
        und = ev_cal.undistort(np.column_stack([gx.ravel(), gy.ravel()]))
        ev_lut = (und[:, 0].reshape(ev_h, ev_w), und[:, 1].reshape(ev_h, ev_w))

    pairs = load_pairs(folder)
    if pairs:
        print("loaded %d pairs from %s"
              % (len(pairs), os.path.join(folder, PAIRS_FILE)))

    state = {"idx": 0, "shown": -1, "playing": False, "win": args.window_ms,
             "opacity": max(0.0, min(1.0, args.opacity)),
             "speed": max(0.05, args.speed),
             "pairs": pairs, "pend_ev": None, "pend_vid": None,
             "dirty": False,
             "und_vis": args.undistort and vis_cal is not None,
             "und_ev": args.undistort and ev_cal is not None,
             "geom": None, "sized": False}

    # Pairs and pending clicks live in raw (distorted) pixels; these map them
    # into, and clicks back out of, whatever the panels currently show.
    def vis_view(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return vis_cal.undistort(pts) if state["und_vis"] else pts

    def ev_view(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return ev_cal.undistort(pts) if state["und_ev"] else pts

    def view_pairs():
        if not state["pairs"]:
            return []
        ev = ev_view([q[0] for q in state["pairs"]])
        vid = vis_view([q[1] for q in state["pairs"]])
        return [(tuple(a), tuple(b)) for a, b in zip(ev, vid)]

    pairs_path = os.path.join(folder, PAIRS_FILE)

    def act(name):
        """Everything the toolbar can do; the keys call the same thing."""
        if name == "vis" and vis_cal is not None:
            state["und_vis"] = not state["und_vis"]
        elif name == "ev" and ev_cal is not None:
            state["und_ev"] = not state["und_ev"]
        elif name == "save":
            print("wrote %d pairs to %s"
                  % (len(state["pairs"]), save_pairs(folder, state["pairs"])))
            state["dirty"] = False
        elif name == "load":
            if os.path.exists(pairs_path):
                state["pairs"] = load_pairs(folder)
                state["pend_ev"] = state["pend_vid"] = None
                state["dirty"] = False
                print("loaded %d pairs from %s" % (len(state["pairs"]),
                                                   pairs_path))
            else:
                print("no %s yet" % pairs_path)
        elif name == "undo":
            state["pend_ev"] = state["pend_vid"] = None
            if state["pairs"]:
                state["pairs"].pop()
                state["dirty"] = True
        elif name == "clear":
            if state["pairs"]:
                state["dirty"] = True
            state["pairs"] = []
            state["pend_ev"] = state["pend_vid"] = None

    def toolbar_items():
        return [("vis", "undistort webcam (v)", True, state["und_vis"],
                 vis_cal is not None),
                ("ev", "undistort events (e)", True, state["und_ev"],
                 ev_cal is not None),
                ("save", "save pairs (w)" + (" *" if state["dirty"] else ""),
                 False, False, True),
                ("load", "load pairs", False, False,
                 os.path.exists(pairs_path)),
                ("undo", "undo pair (u)", False, False,
                 bool(state["pairs"]) or state["pend_ev"] is not None
                 or state["pend_vid"] is not None),
                ("clear", "clear pairs (c)", False, False,
                 bool(state["pairs"]))]

    def on_scrub(v):
        # Fires both for the user's drag and for our own setTrackbarPos after
        # advancing a frame; "shown" tells the two apart so playback survives.
        if v != state["shown"]:
            state["idx"] = v
            state["playing"] = False

    def on_mouse(event, x, y, flags, param):
        g = state["geom"]
        if g is None:
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            state["pend_ev"] = state["pend_vid"] = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cx, cy = x / g["disp"], y / g["disp"]      # undo the canvas shrink
        if cy < BAR_H:                             # toolbar
            for ident, x0, x1 in g["buttons"]:
                if x0 <= cx <= x1:
                    act(ident)
            return
        cy -= BAR_H
        if cy >= g["panel_h"]:                     # bottom panel: not a picker
            return
        if cx < g["left_w"]:
            pt = (cx / vid_panel_scale, cy / vid_panel_scale)
            if state["und_vis"]:
                pt = vis_cal.distort([pt])[0]
            state["pend_vid"] = (float(pt[0]), float(pt[1]))
        elif g["right_x0"] <= cx < g["right_x0"] + g["right_w"]:
            pt = ((cx - g["right_x0"]) / ev_panel_scale, cy / ev_panel_scale)
            if state["und_ev"]:
                pt = ev_cal.distort([pt])[0]
            state["pend_ev"] = (float(pt[0]), float(pt[1]))
        else:
            return                                  # the separator
        if state["pend_ev"] is not None and state["pend_vid"] is not None:
            state["pairs"].append((state["pend_ev"], state["pend_vid"]))
            state["pend_ev"] = state["pend_vid"] = None
            state["dirty"] = True
            print("pair %d: event %.1f,%.1f -> video %.1f,%.1f (raw px)"
                  % ((len(state["pairs"]),)
                     + state["pairs"][-1][0] + state["pairs"][-1][1]))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.createTrackbar("frame", WIN, 0, max(1, n_frames - 1), on_scrub)

    base_fps = args.fps or meta.get("measured fps") or vid_fps

    # Sequential reads are ~100x cheaper than a seek per frame, and playing
    # forward is the common case, so only seek when the target is not next.
    pos = {"next": 0}

    def read_frame(i):
        if i != pos["next"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            pos["next"] = i
        ok, fr = cap.read()
        pos["next"] = i + 1 if ok else -1
        return fr if ok else None

    step_1s = max(1, int(round(base_fps)))
    last = None

    while True:
        i = max(0, min(state["idx"], n_frames - 1))
        state["idx"] = i
        frame = read_frame(i)
        if frame is None:
            break
        if cv2.getTrackbarPos("frame", WIN) != i:
            state["shown"] = i
            cv2.setTrackbarPos("frame", WIN, i)
        state["shown"] = i

        centre = int(frame_us[i])
        half = int(state["win"] * 500)          # ms -> us, halved
        lo, hi = np.searchsorted(ts, (centre - half, centre + half))
        sub = evs[lo:hi]
        on = sub["p"] == 1
        if state["und_ev"]:
            ex = ev_lut[0][sub["y"], sub["x"]]
            ey = ev_lut[1][sub["y"], sub["x"]]
        else:
            ex = sub["x"].astype(np.float64)
            ey = sub["y"].astype(np.float64)
        if state["und_vis"]:
            frame = cv2.remap(frame, vis_maps[0], vis_maps[1],
                              cv2.INTER_LINEAR)
        pairs_v = view_pairs()

        M, model = fit_transform(pairs_v, base_sx, base_sy)

        left = fit_height(frame, args.panel_h).copy()
        right = fit_height(event_image(ex, ey, on, ev_w, ev_h, args.bg),
                           args.panel_h).copy()
        draw_pair_marks(left, [q[1] for q in pairs_v],
                        vid_panel_scale, PAIR_COLOR)
        draw_pair_marks(right, [q[0] for q in pairs_v],
                        ev_panel_scale, PAIR_COLOR)
        if state["pend_vid"] is not None:
            draw_pair_marks(left, vis_view([state["pend_vid"]]),
                            vid_panel_scale, PENDING_COLOR, tags=False)
        if state["pend_ev"] is not None:
            draw_pair_marks(right, ev_view([state["pend_ev"]]),
                            ev_panel_scale, PENDING_COLOR, tags=False)
        label(left, "webcam %dx%d%s  (click here)"
                    % (vid_w, vid_h, "  undistorted" if state["und_vis"] else ""))
        label(right, "events %dx%d%s  %d ev / %.1f ms  (click here)"
                     % (ev_w, ev_h, "  undistorted" if state["und_ev"] else "",
                        len(sub), state["win"]))
        top = np.hstack([left,
                         np.full((args.panel_h, SEP_PX, 3), SEP_COLOR, np.uint8),
                         right])

        over = draw_layer(frame, ex, ey, on, M, state["opacity"],
                          max(1, args.dot))
        bw = top.shape[1]
        bh = max(1, int(round(over.shape[0] * bw / float(over.shape[1]))))
        bottom = cv2.resize(over, (bw, bh),
                            interpolation=(cv2.INTER_AREA if bw < over.shape[1]
                                           else cv2.INTER_LINEAR))
        bot_scale = bw / float(vid_w)
        draw_target_area(bottom, M, ev_w, ev_h, bot_scale)
        # Where each pair's event point actually lands next to where it was
        # clicked in the video: the residual, seen rather than summarised.
        if pairs_v:
            src = np.array([q[0] for q in pairs_v], float)
            fx, fy = apply_xform(M, src[:, 0], src[:, 1])
            draw_pair_marks(bottom, list(zip(fx, fy)), bot_scale, AREA_COLOR)
            draw_pair_marks(bottom, [q[1] for q in pairs_v],
                            bot_scale, PAIR_COLOR, tags=False)
        label(bottom, "events over webcam  opacity %.2f  %s  rms %.1f px"
                      % (state["opacity"], model, rms_residual(M, pairs_v)))

        bar, buttons = draw_toolbar(bw, toolbar_items())
        canvas = np.vstack([bar, top,
                            np.full((SEP_PX, bw, 3), SEP_COLOR, np.uint8),
                            bottom])
        head = ("frame %d/%d  t=%.3fs  %s  %.2fx  %d pair%s%s  undist %s/%s%s"
                % (i, n_frames - 1, (centre - ts[0]) / 1e6,
                   "PLAY" if state["playing"] else "PAUSE", state["speed"],
                   len(pairs_v), "" if len(pairs_v) == 1 else "s",
                   " *unsaved" if state["dirty"] else "",
                   "vis" if state["und_vis"] else "-",
                   "ev" if state["und_ev"] else "-",
                   "" if synced else "  [no sync data - timing approximate]"))
        for colour, thick in (((0, 0, 0), 3), ((0, 200, 255), 1)):
            cv2.putText(canvas, head, (8, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, thick,
                        cv2.LINE_AA)

        disp = 1.0
        if args.max_width and canvas.shape[1] > args.max_width:
            disp = args.max_width / float(canvas.shape[1])
            canvas = cv2.resize(canvas, (args.max_width,
                                         int(round(canvas.shape[0] * disp))),
                                interpolation=cv2.INTER_AREA)
        # Published for the mouse callback: clicks arrive in canvas pixels and
        # have to be walked back through this shrink and the panel layout.
        state["geom"] = {"disp": disp, "panel_h": args.panel_h,
                         "buttons": buttons,
                         "left_w": left.shape[1], "right_w": right.shape[1],
                         "right_x0": left.shape[1] + SEP_PX}
        if not state["sized"]:
            # A NORMAL window opens at some default size; start it at 1:1 so
            # the first picks are full precision, then it is yours to drag.
            cv2.resizeWindow(WIN, canvas.shape[1], canvas.shape[0])
            state["sized"] = True
        cv2.imshow(WIN, canvas)
        last = canvas

        delay = max(1, int(round(1000.0 / max(base_fps * state["speed"], 1.0))))
        key_in = cv2.waitKeyEx(delay if state["playing"] else 20)
        if key_in in (27, ord("q")):
            break
        elif key_in == ord(" "):
            state["playing"] = not state["playing"]
        elif key_in in (2424832, 81):            # left
            state["playing"] = False; state["idx"] = i - 1
        elif key_in in (2555904, 83):            # right
            state["playing"] = False; state["idx"] = i + 1
        elif key_in in (2162688, 85):            # page up
            state["playing"] = False; state["idx"] = i - step_1s
        elif key_in in (2228224, 86):            # page down
            state["playing"] = False; state["idx"] = i + step_1s
        elif key_in in (2359296, 80):            # home
            state["playing"] = False; state["idx"] = 0
        elif key_in in (2293760, 87):            # end
            state["playing"] = False; state["idx"] = n_frames - 1
        elif key_in == ord(","):
            state["win"] = max(0.5, state["win"] - 1.0)
        elif key_in == ord("."):
            state["win"] = min(200.0, state["win"] + 1.0)
        elif key_in == ord("a"):
            state["opacity"] = max(0.0, state["opacity"] - 0.05)
        elif key_in == ord("z"):
            state["opacity"] = min(1.0, state["opacity"] + 0.05)
        elif key_in == ord("["):
            state["speed"] = max(0.05, state["speed"] / 1.5)
        elif key_in == ord("]"):
            state["speed"] = min(8.0, state["speed"] * 1.5)
        elif key_in == ord("u"):
            act("undo")
        elif key_in == ord("c"):
            act("clear")
        elif key_in == ord("w"):
            act("save")
        elif key_in == ord("v"):
            if vis_cal is None:
                print("no usable vis.calib in %s" % calib_dir)
            act("vis")
        elif key_in == ord("e"):
            if ev_cal is None:
                print("no usable event.calib in %s" % calib_dir)
            act("ev")
        elif key_in == ord("s") and last is not None:
            out_path = os.path.join(folder, "compare_%05d.png" % i)
            cv2.imwrite(out_path, last)
            print("wrote %s" % out_path)
        elif state["playing"]:
            state["idx"] = i + 1
            if state["idx"] >= n_frames:
                state["idx"] = n_frames - 1
                state["playing"] = False

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    pairs_v = view_pairs()
    if pairs_v:
        print("%d pairs, rms %.2f px"
              % (len(pairs_v),
                 rms_residual(fit_transform(pairs_v, base_sx, base_sy)[0],
                              pairs_v)))
    if state["dirty"]:
        print("pairs NOT saved (press w before quitting to keep them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
