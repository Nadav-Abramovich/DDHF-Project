#!/usr/bin/env python
"""
Play a dual_record.py folder with the event stream, and the DDHF detections,
drawn over the RGB video.

    python replay_overlay.py runs/rec_20260916_194601
    python replay_overlay.py runs/rec_... --window-ms 3 --drone tello
    python replay_overlay.py runs/rec_... --mode pixelize --tau-comb 4

Keys
    SPACE .... play / pause          left / right ... step 1 frame (Shift: 10)
    home/end . first / last frame    , / . .......... event window -/+ 1 ms
    a / z .... opacity -/+           e .............. detections on / off
    s ........ save the frame        q / ESC ........ quit

Note --opacity, not --alpha: the detector's own cfg.alpha already owns that
flag via add_cfg_cli_args.

Trackbars set the geometric alignment (scale, dx, dy) - see below.

How the two streams are lined up in TIME
    The event timestamps are sensor-relative microseconds from the start of the
    RAW; the webcam frames carry wall-clock times in camera_frames.csv; and
    meta.txt records the wall clock at which each stream started. So for webcam
    frame i:

        event_time_us = (csv_stamp[i] - meta["events started"]) * 1e6

    and the events drawn are those within +-window/2 of it. That is why the CSV
    exists: the .mp4's nominal frame rate is a container field, not a
    measurement, and using frame_number / fps instead will drift.

    With meta.txt or the CSV missing this falls back to frame_number / fps and
    says so in the header - usable for a quick look, not for timing.

How they are lined up in SPACE, which is to say: by hand
    The two cameras have different resolutions, lenses, and viewpoints, and
    nothing here calibrates them. The event image is simply scaled and shifted
    onto the video, and the scale/dx/dy trackbars are there for you to line the
    two up by eye on something both sensors can see. A proper job needs a
    homography from a moving blinking target, which this script deliberately
    does not pretend to do. Treat the overlay as an aid to seeing what the
    event camera saw, not as registered data.

Two different windows, deliberately
    --window-ms is how many milliseconds of events to DRAW (default 10; drop it
    to 2-3 to resolve individual blades, since a 590 Hz blade-pass is 1.7 ms per
    period). --detector-window-ms is the window the DDHF detector transforms
    (default 33.333, the value it was tuned at). Changing the display window
    must not silently retune the detector, so they are separate.

    Detections are cached per DETECTOR window, not per video frame - several
    video frames fall inside one detector window and share its result - and are
    only computed while paused. The NDFT costs a few hundred ms, which is fine
    when you stop on a frame and impossible at 30 fps.
"""
import os
import sys
import csv
import argparse
from collections import OrderedDict

import numpy as np
import cv2

# (kept in lab/OLD - not used by any other script or launcher; it uses the
#  detector modules in ../code)
_HERE = os.path.dirname(os.path.abspath(__file__))           # lab/OLD
_CODE = os.path.join(os.path.dirname(_HERE), "code", "common")   # lab/code/common
for _p in (_CODE, os.path.join(os.path.dirname(_HERE), "camera")):    # camera/: silky_bias_studio
    if _p not in sys.path:
        sys.path.insert(0, _p)
import silky_bias_studio as studio            # DLL/plugin bootstrap
from metavision_core.event_io import RawReader


def _find_code_dir():
    """Directory holding the shared detector modules (mirrors event_debug_viewer)."""
    candidates = [_CODE]                     # lab/code holds the detector copies
    for c in candidates:
        if os.path.exists(os.path.join(c, "debug_viewer_pooled.py")):
            return c
    raise FileNotFoundError("could not find debug_viewer_pooled.py in:\n  "
                            + "\n  ".join(candidates))


sys.path.insert(0, _find_code_dir())
import debug_viewer_pooled as dv
from ddhf_config import build_cfg, add_cfg_cli_args, cfg_overrides_from_args
from utils_pooled import _extract_boxes

WIN = "events over RGB"
ON_COLOR = (255, 255, 255)      # BGR
OFF_COLOR = (0, 0, 255)
MASK_COLOR = (0, 255, 0)
BOX_COLOR = (255, 0, 255)
DETECT_CACHE_MAX = 64


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


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def _paint(over, xs, ys, colour, dot, w, h):
    for oy in range(dot):
        for ox in range(dot):
            px, py = xs + ox, ys + oy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            over[py[ok], px[ok]] = colour


def draw_layer(frame, evs, mask, scale, dx, dy, alpha, dot):
    """Paint events (and the rotor mask) onto a copy, then blend it back.

    Blending the whole copy rather than masking is deliberate: pixels with no
    event are identical in both images, so addWeighted leaves them untouched
    and only the painted ones move toward the overlay colour. One pass, no mask
    image to build.
    """
    h, w = frame.shape[:2]
    over = frame.copy()
    if len(evs):
        vx = (evs["x"] * scale + dx).astype(np.int32)
        vy = (evs["y"] * scale + dy).astype(np.int32)
        on = evs["p"] == 1
        _paint(over, vx[on], vy[on], ON_COLOR, dot, w, h)
        _paint(over, vx[~on], vy[~on], OFF_COLOR, dot, w, h)
    if mask is not None and mask.any():
        my, mx = np.nonzero(mask)
        _paint(over, (mx * scale + dx).astype(np.int32),
               (my * scale + dy).astype(np.int32), MASK_COLOR, max(dot, 2), w, h)
    return cv2.addWeighted(over, alpha, frame, 1.0 - alpha, 0.0)


def draw_boxes(img, boxes, scale, dx, dy):
    for b in boxes:
        x1 = int(b.x1 * scale + dx); y1 = int(b.y1 * scale + dy)
        x2 = int(b.x2 * scale + dx); y2 = int(b.y2 * scale + dy)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        score = getattr(b, "score", None)
        if score is not None:
            cv2.putText(img, "snr=%.1f" % score, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="a rec_* folder written by dual_record.py")
    p.add_argument("--window-ms", type=float, default=10.0,
                   help="milliseconds of events to DRAW per frame (default 10; "
                        "2-3 resolves individual blades)")
    p.add_argument("--detector-window-ms", type=float, default=33.333,
                   help="window the DDHF detector transforms (default 33.333, "
                        "what it was tuned at). Separate from --window-ms on "
                        "purpose")
    # NOT --alpha: add_cfg_cli_args already owns that for the detector's
    # cfg.alpha, and argparse rejects the duplicate.
    p.add_argument("--opacity", type=float, default=0.85,
                   help="overlay opacity 0-1 (default 0.85)")
    p.add_argument("--dot", type=int, default=1,
                   help="event dot size in video pixels (default 1; use 2 when "
                        "the video is much larger than the event sensor)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="playback rate; 0 uses the measured rate from meta.txt")
    p.add_argument("--mode", choices=tuple(dv.MODE_FLAGS), default="pixelize",
                   help="detector spatial aggregation (default pixelize)")
    p.add_argument("--drone", choices=sorted(dv.DRONE_BANDS), default="all",
                   help="restrict the comb search to one drone's bands")
    p.add_argument("--no-detect", action="store_true",
                   help="start with detections off")
    p.add_argument("--gpu", action="store_true",
                   help="run the spectral stage on the GPU (CuPy) if available")
    add_cfg_cli_args(p)
    args = p.parse_args(argv)

    folder = args.folder
    raw = os.path.join(folder, "events.raw")
    vid = os.path.join(folder, "camera.mp4")
    for path in (raw, vid):
        if not os.path.exists(path):
            raise SystemExit("missing %s" % path)

    dv._USE_GPU = args.gpu
    dv._COMPONENTS, dv._PIXELIZE, dv._POOLING = dv.MODE_FLAGS[args.mode]
    cfg = build_cfg(**cfg_overrides_from_args(args))
    cfg.window_ms = cfg.step_ms = args.detector_window_ms
    cfg.rotor_bands_hz = dv.DRONE_BANDS[args.drone]
    det_us = int(cfg.window_ms * 1000)

    evs, ts, ev_w, ev_h = load_events(raw)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % vid)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

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

    print("events %d over %.2f s (%dx%d) | video %d frames @ %.2f fps | %s"
          % (len(evs), (ts[-1] - ts[0]) / 1e6, ev_w, ev_h, n_frames, vid_fps,
             "CSV-synced" if synced else "NO SYNC DATA - assuming frame/fps"))
    print("detector: %s  window=%gms  bands=%s" % (
        args.mode, cfg.window_ms,
        ",".join("%.0f-%.0f" % b for b in cfg.rotor_bands_hz) or "all"))

    detect_cache = OrderedDict()

    def detections(centre_us):
        """Mask and boxes for the detector window containing centre_us.

        Keyed by window index rather than video frame: several video frames land
        inside one detector window and must share a single NDFT, which is what
        makes stepping through the video bearable at all.
        """
        key = int((centre_us - int(ts[0])) // det_us)
        hit = detect_cache.get(key)
        if hit is not None:
            detect_cache.move_to_end(key)
            return hit
        lo_us = int(ts[0]) + key * det_us
        lo, hi = np.searchsorted(ts, (lo_us, lo_us + det_us))
        sub = evs[lo:hi]
        if not len(sub):
            out = (None, [])
        elif dv._COMPONENTS:
            rmask, boxes, _ = dv.detect_components_mask(sub, ev_h, ev_w, cfg)
            out = (rmask, boxes)
        else:
            _c, sfm, snrm, _f0 = dv.compute_pixel_stats(sub, ev_h, ev_w, cfg)
            rmask = (~np.isnan(sfm) & (sfm <= cfg.tau_sf)
                     & ~np.isnan(snrm) & (snrm >= cfg.tau_comb))
            out = (rmask, _extract_boxes(rmask, np.nan_to_num(snrm), cfg))
        detect_cache[key] = out
        if len(detect_cache) > DETECT_CACHE_MAX:
            detect_cache.popitem(last=False)
        return out

    base_scale = vid_w / float(ev_w)
    state = {"idx": 0, "playing": False, "win": args.window_ms,
             "opacity": max(0.0, min(1.0, args.opacity)),
             "detect": not args.no_detect}

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("scale %", WIN, 100, 300, lambda v: None)
    cv2.createTrackbar("dx +500", WIN, 500, 1000, lambda v: None)
    cv2.createTrackbar("dy +500", WIN, 500, 1000, lambda v: None)

    play_fps = args.fps or meta.get("measured fps") or vid_fps
    delay = max(1, int(round(1000.0 / max(play_fps, 1.0))))
    last = None

    while True:
        i = max(0, min(state["idx"], n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        centre = int(frame_us[i])
        half = int(state["win"] * 500)          # ms -> us, halved
        lo, hi = np.searchsorted(ts, (centre - half, centre + half))
        sub = evs[lo:hi]

        # Detections only while paused, or when this window is already cached -
        # the NDFT is a few hundred ms and would pin playback at ~2 fps.
        mask, boxes, stale = None, [], False
        if state["detect"]:
            key = int((centre - int(ts[0])) // det_us)
            if not state["playing"] or key in detect_cache:
                mask, boxes = detections(centre)
            else:
                stale = True

        scale = base_scale * cv2.getTrackbarPos("scale %", WIN) / 100.0
        dx = cv2.getTrackbarPos("dx +500", WIN) - 500
        dy = cv2.getTrackbarPos("dy +500", WIN) - 500
        out = draw_layer(frame, sub, mask, scale, dx, dy,
                         state["opacity"], max(1, args.dot))
        draw_boxes(out, boxes, scale, dx, dy)

        head = ("frame %d/%d  t=%.3fs  %d ev in %.0f ms  opac %.2f  %s  %s"
                % (i, n_frames - 1, (centre - ts[0]) / 1e6, len(sub),
                   state["win"], state["opacity"],
                   "PLAY" if state["playing"] else "PAUSE",
                   ("det off" if not state["detect"] else
                    ("det skipped (playing)" if stale
                     else "det: %d boxes" % len(boxes)))))
        for colour, thick in (((0, 0, 0), 3), ((0, 255, 0), 1)):
            cv2.putText(out, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        colour, thick, cv2.LINE_AA)
        if not synced:
            warn = "no camera_frames.csv / meta.txt - timing is approximate"
            for colour, thick in (((0, 0, 0), 3), ((0, 200, 255), 1)):
                cv2.putText(out, warn, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colour, thick, cv2.LINE_AA)
        cv2.imshow(WIN, out)
        last = out

        key_in = cv2.waitKeyEx(delay if state["playing"] else 20)
        if key_in in (27, ord("q")):
            break
        elif key_in == ord(" "):
            state["playing"] = not state["playing"]
        elif key_in in (2424832, 81):            # left
            state["playing"] = False; state["idx"] = i - 1
        elif key_in in (2555904, 83):            # right
            state["playing"] = False; state["idx"] = i + 1
        elif key_in == ord(","):
            state["win"] = max(0.5, state["win"] - 1.0)
        elif key_in == ord("."):
            state["win"] = min(200.0, state["win"] + 1.0)
        elif key_in == ord("a"):
            state["opacity"] = max(0.0, state["opacity"] - 0.05)
        elif key_in == ord("z"):
            state["opacity"] = min(1.0, state["opacity"] + 0.05)
        elif key_in == ord("e"):
            state["detect"] = not state["detect"]
        elif key_in in (2359296, 80):            # home
            state["playing"] = False; state["idx"] = 0
        elif key_in in (2293760, 87):            # end
            state["playing"] = False; state["idx"] = n_frames - 1
        elif key_in == ord("s") and last is not None:
            out_path = os.path.join(folder, "overlay_%05d.png" % i)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
