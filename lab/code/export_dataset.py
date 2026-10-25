#!/usr/bin/env python
"""
Turn a dual_record.py run into a FRED-style sequence folder that
lab_viewer_pooled.py (and show_lab_case_graphs.bat) can open, using the lens calibrations
(runs/vis.calib, runs/event.calib) and the pairs picked in compare_replay.py.

    python export_dataset.py                        # choose a run from a list
    python export_dataset.py rec_20260916_194601
    python debug_viewer.py --seq C:\\...\\lab\\runs\\dataset\\rec_20260916_194601

debug_viewer joins --seq onto its own folder, and an absolute path wins that
join, so the dataset does not have to be moved next to it.

Output, runs/dataset/<run>/
    Event/events.raw        the recording's events, untouched
    Event/Frames/Video_0_frame_<us>.png
                            event frames in the dataset's style for view 5:
                            events of [us - 33333, us), ON white, OFF blue
    PADDED_RGB/Video_0_HH_MM_SS.ffffff.jpg
                            every webcam frame, whole, undistorted with
                            vis.calib (same size and K, only the lens bending
                            removed; --raw-vis keeps it raw). Where the event
                            sensor lands is in registration.npz, which
                            lab_viewer_pooled.py draws through. --warp writes
                            the old output instead: each frame warped into
                            event-sensor pixels, so the viewer overlays events
                            at identity. The filename time IS the event
                            timestamp either way
    coordinates.txt         ground truth - created empty, never overwritten
    interpolated_coordinates.txt
                            likewise; its presence is what makes the folder
                            count as a case for utility/frame_features.py
    registration.txt        what was used: calibs, model, matrix, rms, timing
    registration.npz        the same mapping for lab_viewer_pooled.py: for
                            every raw event pixel, where it lands in a
                            PADDED_RGB frame (lens calibs included)

Geometry
    The dataset lives in RAW event pixels - events.raw is copied as is, and
    DDHF and the GT boxes both work there. So each PADDED_RGB pixel is sampled
    through the whole chain for its raw event pixel,

        raw event px -> undistort (event.calib) -> fitted transform
                     -> distort (vis.calib) -> raw webcam px

    as a single remap, and the webcam frame is resampled exactly once (with
    --warp). By default the frame is only undistorted, and the chain minus
    its last step (so landing in undistorted webcam px) is saved as
    registration.npz instead. The
    transform is refit from <run>/pairs.csv exactly as compare_replay.py fits
    it, in the view where every camera with a filled-in .calib is undistorted.
    --raw-vis / --raw-events leave that camera raw instead, to match what you
    saw in compare_replay with its checkbox off.

Time
    Webcam frame i sits at event time
        (csv_stamp[i] - meta["events started"]) * 1e6 + first event t
    as in compare_replay.py. debug_viewer takes the first PADDED_RGB file as
    event time 0, so when the first webcam frame lands after 0 a copy of it is
    also written at 00_00_00.000000 to hold that offset.
"""
import os
import sys
import glob
import shutil
import argparse
import datetime

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "common")):   # code/common: detector copies
    if _p not in sys.path:
        sys.path.insert(0, _p)
import compare_replay as cr                   # also the Metavision bootstrap

RUNS = os.path.join(os.path.dirname(_HERE), "runs")
SEQ = "Video_0"             # filename prefix; debug_viewer only parses the tail
WINDOW_US = 33333           # the dataset's 30 fps grid (debug_viewer's default)
FRAME_BG = (52, 37, 30)     # BGR of the dataset's event frames, measured off
FRAME_ON = (255, 255, 255)  # FRED seq 0: ON white, OFF blue, last event wins
FRAME_OFF = (200, 126, 64)
REGISTRATION_FILE = "registration.npz"   # read by lab_viewer_pooled.py


# ---------------------------------------------------------------------------
# Choosing the run
# ---------------------------------------------------------------------------
def list_runs(root):
    """Folders under root that hold a dual_record.py recording."""
    out = []
    if not os.path.isdir(root):                  # nothing recorded yet
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isfile(os.path.join(d, "events.raw"))
                and os.path.isfile(os.path.join(d, "camera.mp4"))):
            out.append(name)
    return out


def pick_run(root):
    runs = list_runs(root)
    if not runs:
        raise SystemExit("no runs with events.raw + camera.mp4 in %s" % root)
    print("runs in %s:" % root)
    for k, name in enumerate(runs):
        n = len(cr.load_pairs(os.path.join(root, name)))
        done = os.path.isdir(os.path.join(root, "dataset", name))
        print("  %2d  %-26s %3d pair%s%s" % (k + 1, name, n,
                                            " " if n == 1 else "s",
                                            "   [already exported]" if done else ""))
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
# Geometry and naming
# ---------------------------------------------------------------------------
def build_map(M, ev_cal, vis_cal, ev_w, ev_h, vid_w, vid_h):
    """remap() tables: for each raw event pixel, the raw webcam pixel to take."""
    gx, gy = np.meshgrid(np.arange(ev_w, dtype=float),
                         np.arange(ev_h, dtype=float))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if ev_cal is not None:
        pts = ev_cal.undistort(pts)
    qx, qy = cr.apply_xform(M, pts[:, 0], pts[:, 1])
    # Well outside the webcam frame the distortion polynomial folds back on
    # itself and would pull in pixels from the far side; leave those black.
    mx, my = 0.25 * vid_w, 0.25 * vid_h
    far = ~(np.isfinite(qx) & np.isfinite(qy)
            & (qx > -mx) & (qx < vid_w - 1 + mx)
            & (qy > -my) & (qy < vid_h - 1 + my))
    q = np.column_stack([np.where(far, 0.0, qx), np.where(far, 0.0, qy)])
    if vis_cal is not None:
        q = vis_cal.distort(q)
    q[far] = -1000.0                      # off the image: BORDER_CONSTANT black
    return (q[:, 0].reshape(ev_h, ev_w).astype(np.float32),
            q[:, 1].reshape(ev_h, ev_w).astype(np.float32))


def rgb_name(t_us):
    """Event time as the HH_MM_SS.ffffff that utils.rgb_frame_seconds parses."""
    us = int(round(t_us))
    s, frac = divmod(us, 1_000_000)
    return "%s_%02d_%02d_%02d.%06d.jpg" % (SEQ, s // 3600, s // 60 % 60,
                                           s % 60, frac)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?",
                   help="run folder name under --runs-dir, or a path; "
                        "omitted: choose from a list")
    p.add_argument("--runs-dir", default=RUNS,
                   help="where the runs live; the dataset goes in "
                        "<runs-dir>/dataset (default lab/runs)")
    p.add_argument("--calib-dir", default=None,
                   help="folder holding vis.calib and event.calib (default: "
                        "the run's parent, as in compare_replay.py)")
    p.add_argument("--raw-vis", action="store_true",
                   help="do not undistort the webcam even if vis.calib is filled in")
    p.add_argument("--raw-events", action="store_true",
                   help="do not undistort the events even if event.calib is filled in")
    p.add_argument("--no-event-frames", action="store_true",
                   help="skip Event/Frames (debug_viewer's view 5 then shows black)")
    p.add_argument("--warp", action="store_true",
                   help="write PADDED_RGB warped into event pixels (the old "
                        "output) instead of the full, undistorted webcam frame")
    p.add_argument("--jpeg-quality", type=int, default=95)
    args = p.parse_args(argv)

    if args.run is None:
        src = pick_run(args.runs_dir)
    elif os.path.isdir(args.run):
        src = args.run
    else:
        src = os.path.join(args.runs_dir, args.run)
    src = os.path.abspath(src)
    name = os.path.basename(src)
    raw = os.path.join(src, "events.raw")
    vid = os.path.join(src, "camera.mp4")
    for path in (raw, vid):
        if not os.path.exists(path):
            raise SystemExit("missing %s" % path)
    out = os.path.join(args.runs_dir, "dataset", name)

    pairs = cr.load_pairs(src)
    if not pairs:
        raise SystemExit("no %s - pick pairs in compare_replay.py and save "
                         "them first" % os.path.join(src, cr.PAIRS_FILE))

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % vid)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("loading %s ..." % raw)
    evs, ts, ev_w, ev_h = cr.load_events(raw)

    # --- registration -----------------------------------------------------
    calib_dir = args.calib_dir or os.path.dirname(src)
    if args.raw_vis:
        vis_cal, vis_note = None, "off (--raw-vis)"
    else:
        vis_cal, vis_note = cr.load_calib(os.path.join(calib_dir, "vis.calib"),
                                          vid_w, vid_h)
    if args.raw_events:
        ev_cal, ev_note = None, "off (--raw-events)"
    else:
        ev_cal, ev_note = cr.load_calib(os.path.join(calib_dir, "event.calib"),
                                        ev_w, ev_h)
    print("vis calib   " + vis_note)
    print("event calib " + ev_note)

    ev_pts = np.array([q[0] for q in pairs], float)
    vid_pts = np.array([q[1] for q in pairs], float)
    if ev_cal is not None:
        ev_pts = ev_cal.undistort(ev_pts)
    if vis_cal is not None:
        vid_pts = vis_cal.undistort(vid_pts)
    view_pairs = [(tuple(a), tuple(b)) for a, b in zip(ev_pts, vid_pts)]
    M, model = cr.fit_transform(view_pairs, vid_w / float(ev_w),
                                vid_h / float(ev_h))
    rms = cr.rms_residual(M, view_pairs)
    print("%d pairs -> %s, rms %.2f px" % (len(pairs), model, rms))
    if len(pairs) < 4:
        print("warning: under 4 pairs, so this is not a homography")
    # Per raw event pixel, where it lands in a PADDED_RGB frame - saved below
    # as registration.npz for lab_viewer_pooled.py. With --warp that is the
    # raw webcam pixel the remap samples. Without it the frames are written
    # undistorted, so the chain stops before vis.calib's distort step.
    if args.warp:
        mapx, mapy = build_map(M, ev_cal, vis_cal, ev_w, ev_h, vid_w, vid_h)
        und_maps = None
    else:
        mapx, mapy = build_map(M, ev_cal, None, ev_w, ev_h, vid_w, vid_h)
        # Same K in and out (Calib.maps), so the frame keeps its size and
        # centre and only the barrel bending comes out.
        und_maps = vis_cal.maps(vid_w, vid_h) if vis_cal is not None else None

    # --- time -------------------------------------------------------------
    stamps = cr.load_stamps(src)
    t_events = cr.load_meta(src).get("events started")
    if stamps is not None and t_events is not None:
        n_frames = min(n_frames, len(stamps))
        frame_us = (stamps[:n_frames] - t_events) * 1e6 + float(ts[0])
        timing = "CSV-synced"
    else:
        frame_us = float(ts[0]) + np.arange(n_frames) / vid_fps * 1e6
        timing = "NO SYNC DATA - frame/fps, approximate"
        print("warning: " + timing)

    # --- output folder ----------------------------------------------------
    ev_dir = os.path.join(out, "Event")
    frames_dir = os.path.join(ev_dir, "Frames")
    rgb_dir = os.path.join(out, "PADDED_RGB")
    # Anything left from an earlier export would interleave with the new
    # frames, and the viewer's caches were computed from the old data.
    for d in (frames_dir, rgb_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for stale in (glob.glob(os.path.join(out, ".usable_px_*.npz"))
                  + glob.glob(os.path.join(out, ".iou_*.npz"))
                  + glob.glob(os.path.join(ev_dir, "*.tmp_index"))):
        os.remove(stale)
    os.makedirs(rgb_dir)
    os.makedirs(frames_dir)

    shutil.copyfile(raw, os.path.join(ev_dir, "events.raw"))
    print("copied events.raw")

    # interpolated_coordinates.txt too: without
    # it case_statistics._is_case_dir rejects the folder, so frame_features.py
    # (the cache the viewer's --wp replay reads) will not build it.
    for gt in (os.path.join(out, "coordinates.txt"),
               os.path.join(out, "interpolated_coordinates.txt")):
        if os.path.exists(gt):
            print("kept existing %s" % gt)
        else:
            open(gt, "w").close()
            print("created empty %s" % gt)

    # --- webcam frames ----------------------------------------------------
    jpg = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    n_rgb = 0
    padded_zero = False
    for i in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        t = frame_us[i]
        if t < 0:                         # webcam ran before the events did
            continue
        if args.warp:
            img = cv2.remap(fr, mapx, mapy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img = fr if und_maps is None else cv2.remap(
                fr, und_maps[0], und_maps[1], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if n_rgb == 0 and int(round(t)) > 0:
            cv2.imwrite(os.path.join(rgb_dir, rgb_name(0)), img, jpg)
            padded_zero = True
        cv2.imwrite(os.path.join(rgb_dir, rgb_name(t)), img, jpg)
        n_rgb += 1
        if n_rgb % 500 == 0:
            print("  rgb %d/%d" % (n_rgb, n_frames))
    cap.release()
    print("wrote %d PADDED_RGB frames%s" % (n_rgb, " (+ a copy of the first "
                                            "at t=0)" if padded_zero else ""))

    # --- event frames -----------------------------------------------------
    n_ev = 0
    if not args.no_event_frames:
        on_c = np.array(FRAME_ON, np.uint8)
        off_c = np.array(FRAME_OFF, np.uint8)
        for k in range(1, int(ts[-1] // WINDOW_US) + 2):
            t = k * WINDOW_US
            lo, hi = np.searchsorted(ts, (t - WINDOW_US, t))
            e = evs[lo:hi]
            img = np.full((ev_h, ev_w, 3), FRAME_BG, np.uint8)
            img[e["y"], e["x"]] = np.where((e["p"] == 1)[:, None], on_c, off_c)
            cv2.imwrite(os.path.join(frames_dir, "%s_frame_%d.png" % (SEQ, t)),
                        img)
            n_ev += 1
            if n_ev % 500 == 0:
                print("  event frame %d" % n_ev)
        print("wrote %d event frames" % n_ev)

    # --- record of what was done --------------------------------------------
    # Machine-readable twin of registration.txt. ev_to_rgb_x/y[y, x] is where
    # raw event pixel (x, y) lands in a PADDED_RGB frame; with --warp the
    # frames already ARE event pixels, so the viewer is told to use identity.
    far = mapx < -500                     # build_map's off-image marker
    np.savez(os.path.join(out, REGISTRATION_FILE),
             ev_to_rgb_x=np.where(far, np.nan, mapx).astype(np.float32),
             ev_to_rgb_y=np.where(far, np.nan, mapy).astype(np.float32),
             M=M, model=np.array(model), rms=np.array(rms),
             rgb_warped=np.array(bool(args.warp)),
             ev_size=np.array([ev_w, ev_h]), rgb_size=np.array([vid_w, vid_h]))
    with open(os.path.join(out, "registration.txt"), "w") as fh:
        fh.write("source        %s\n" % src)
        fh.write("exported      %s\n" % datetime.datetime.now().isoformat())
        fh.write("event sensor  %dx%d   webcam %dx%d\n" % (ev_w, ev_h, vid_w, vid_h))
        fh.write("vis calib     %s\n" % vis_note)
        fh.write("event calib   %s\n" % ev_note)
        fh.write("pairs         %d (pairs.csv, raw px)\n" % len(pairs))
        fh.write("model         %s   rms %.3f px\n" % (model, rms))
        fh.write("M (event view -> webcam view, each undistorted if its calib"
                 " is on):\n")
        for row in M:
            fh.write("  % .9g % .9g % .9g\n" % tuple(row))
        fh.write("timing        %s; first event t = %d us\n" % (timing, ts[0]))
        fh.write("PADDED_RGB    %d frames%s, %s\n"
                 % (n_rgb, " + copy of first at t=0" if padded_zero else "",
                    "warped into event pixels" if args.warp else
                    "full %dx%d webcam frame, %s"
                    % (vid_w, vid_h, "undistorted (vis.calib)"
                       if und_maps is not None else "raw")))
        fh.write("event frames  %d on a %d us grid\n" % (n_ev, WINDOW_US))
    print("\ndone: %s" % out)
    print('view it with:  python debug_viewer.py --seq "%s"' % os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
