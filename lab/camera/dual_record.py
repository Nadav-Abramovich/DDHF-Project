#!/usr/bin/env python
"""
silky_bias_studio with a second, ordinary camera previewed and recorded beside it.

Same viewer, same bias panel, same keys. A third window shows the webcam live
from startup, and `r` now starts BOTH streams, writing them into one folder
instead of a loose .raw:

    rec_20260916_141233/
        events.raw          the Metavision RAW, exactly as the studio writes it
        events.bias         the biases in force, as a sidecar
        camera.mp4          the webcam, nominally 30 fps
        camera_frames.csv   wall-clock timestamp of every webcam frame
        meta.txt            start times, measured fps, camera index, biases
        vis.calib           copies of the lens calibrations, when present
        event.calib

Why the webcam grabs continuously rather than only while recording
    The preview has to be live before you press `r` - that is how you aim and
    frame the shot - and a camera that is already open and warm adds no
    start-up latency when recording begins. So one thread grabs forever and
    recording is just a flag that also hands each frame to a VideoWriter.

Why the CSV exists
    The .mp4 container carries ONE nominal frame rate, but a webcam delivers
    whatever it feels like - 30 fps is a request, not a promise, and USB
    bandwidth or auto-exposure make it sag. The CSV records when each frame
    actually arrived, so if you need to line a webcam frame up against an event
    timestamp you use the CSV, not the frame number. meta.txt reports the
    measured average so you can tell at a glance whether the container rate is
    a lie.

Why the two streams are not sample-synchronous
    They start as close together as software allows - the webcam is already
    open and primed, so `r` only has to open a writer - and both wall-clock
    instants go in meta.txt. There is still a few-millisecond unknown between
    them, and the two clocks drift independently afterwards. For anything
    needing better, the Gen3.1 exposes I_TriggerIn / I_TriggerOut and the right
    answer is a hardware trigger, not tighter code.

Picking the camera
    --camera-index chooses it. With no flag the HIGHEST working index is used,
    because a laptop's built-in camera is almost always index 0 and a camera
    plugged in later enumerates after it. That is a heuristic, not a rule:
    run --list-cameras once and pass the index explicitly if it guesses wrong.

Lens calibration
    vis.calib and event.calib (in --calib-dir, default the record dir - same
    files compare_replay.py reads) undistort the PREVIEWS only: v toggles the
    webcam, u the events (e is already the studio's rotor readout), and
    --undistort starts with both on. What gets recorded is always raw, since
    undistortion can be applied later but never taken back out. Each rec_
    folder gets a copy of both files, so a recording keeps the calibration it
    was taken with even after the originals are redone.

Usage
    python dual_record.py --list-cameras
    python dual_record.py                    # freqcam_quadrotor.bias -> runs/
    python dual_record.py -b wp.bias --camera-index 1 --record-dir other
    python dual_record.py --camera-size 1280x720 --camera-preview-width 800

Every silky_bias_studio flag still works; they are passed straight through.
"""
import os
import sys
import csv
import time
import argparse
import datetime
import shutil
import threading

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB  = os.path.dirname(_HERE)                       # lab/ (data/, runs/, compare_replay)
for _p in (_HERE, os.path.join(_LAB, "code"), os.path.join(_LAB, "code", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import silky_bias_studio as studio
from compare_replay import load_calib

CAM_WIN = "Camera (second sensor)"
CALIB_FILES = ("vis.calib", "event.calib")
# "fast (quad rotor)" biases from Pfrommer, Frequency Cam (arXiv:2211.00198),
# Table 1 - published for this same camera (SilkyEvCam EvC3A), not tuned here.
DEFAULT_BIAS = os.path.join(_LAB, "data", "camera_biases",
                            "freqcam_quadrotor.bias")
DEFAULT_RECORD_DIR = os.path.join(_LAB, "runs")

# ---------------------------------------------------------------------------
# Webcam discovery
# ---------------------------------------------------------------------------
_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def probe_cameras(max_index=8):
    """Indices that open AND return a frame, with their native size and fps.

    Opening alone is not proof: on Windows a ghost index will open happily and
    then never deliver, so each candidate has to be read from before it counts.
    """
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, _BACKEND)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            found.append((idx, w, h, float(cap.get(cv2.CAP_PROP_FPS) or 0.0)))
        finally:
            cap.release()
    return found


def pick_camera(max_index=8):
    """Highest working index - see the module docstring for why."""
    found = probe_cameras(max_index)
    return found[-1][0] if found else None


# ---------------------------------------------------------------------------
# Webcam feed: always previewing, optionally recording
# ---------------------------------------------------------------------------
class WebcamFeed:
    """Grabs on its own thread so the event display never waits on USB.

    cap.read() blocks until the camera has a frame, which paces the loop at the
    camera's own rate for free. While recording, every frame is written and
    timestamped - nothing is dropped or duplicated to hit a target rate,
    because a video whose frames are real is more useful for alignment than one
    whose duration is tidy.
    """

    def __init__(self, index, fps=30.0, size=None):
        self.index = index
        self.fps = fps
        self.size = size
        self.cap = None
        self.width = self.height = 0
        self.error = None

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # most recent frame, for the preview

        self.writer = None           # non-None only while recording
        self.folder = None
        self.frames = 0
        self.t_open = None
        self.t_stop = None
        self._stamps = []

    # -- lifecycle ----------------------------------------------------------
    def open(self):
        """Open and prime the camera. Raises if it cannot deliver a frame."""
        self.cap = cv2.VideoCapture(self.index, _BACKEND)
        if not self.cap.isOpened():
            raise OSError("camera index %d will not open" % self.index)
        # MJPG before the size: raw YUY2 at 1080p30 does not fit through USB 2,
        # so without it the driver drops to a small mode or ~5 fps.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if self.size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = self.cap.read()          # first read is often slow
        if not ok or frame is None:
            self.cap.release()
            raise OSError("camera index %d opened but returned no frame" % self.index)
        self.height, self.width = frame.shape[:2]
        with self._lock:
            self._latest = frame
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.width, self.height

    def _loop(self):
        """Grab continuously; write on a wall-clock grid.

        The writes are paced rather than one-per-grab because cap.read() does
        NOT reliably block until a new frame exists - with DirectShow it often
        returns the buffered frame immediately, so an unpaced loop free-runs at
        CPU speed and writes the same image many times. Measured on this rig:
        142 fps written into a container labelled 30, i.e. a video that plays
        4.7x too slow and is 4.7x too large.

        Pacing advances on a fixed grid (next + period) rather than from `now`,
        so jitter does not accumulate and the long-run rate stays exactly fps.
        The max() stops a stall from being repaid as a burst of catch-up
        writes, and the quarter-period tolerance stops a camera running at
        almost exactly fps from having every other frame skipped.
        """
        period = 1.0 / max(self.fps, 1.0)
        tolerance = period * 0.25
        next_write = None
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    continue
                now = time.time()
                with self._lock:
                    self._latest = frame          # preview always gets the newest
                    writer = self.writer
                    if writer is None:
                        next_write = None
                        continue
                    if next_write is None:
                        next_write = now
                    if now < next_write - tolerance:
                        continue                  # too soon: this one is a dup
                    writer.write(frame)
                    self._stamps.append(now)
                    self.frames += 1
                    next_write = max(now, next_write + period)
        except Exception as exc:                 # never take the viewer down
            self.error = str(exc)

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.cap is not None:
            self.cap.release()

    # -- recording ----------------------------------------------------------
    @property
    def recording(self):
        return self.writer is not None

    def start_recording(self, folder):
        writer = cv2.VideoWriter(os.path.join(folder, "camera.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise OSError("cannot open a VideoWriter in %s" % folder)
        with self._lock:
            self.folder = folder
            self.frames = 0
            self._stamps = []
            self.t_open = time.time()
            self.writer = writer          # grab thread starts writing from here

    def stop_recording(self):
        """Stop, flush, write the timestamp CSV. Returns (frames, measured_fps)."""
        with self._lock:
            writer, self.writer = self.writer, None
            stamps = list(self._stamps)
            folder = self.folder
        if writer is not None:
            writer.release()
        self.t_stop = time.time()
        if folder:
            with open(os.path.join(folder, "camera_frames.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "wall_clock_s", "since_start_s"])
                for i, t in enumerate(stamps):
                    w.writerow([i, "%.6f" % t, "%.6f" % (t - self.t_open)])
        if len(stamps) > 1:
            span = max(1e-6, stamps[-1] - stamps[0])
            measured = (len(stamps) - 1) / span
        else:
            measured = 0.0
        return len(stamps), measured


# ---------------------------------------------------------------------------
# Studio with a second camera bolted on
# ---------------------------------------------------------------------------
class UndistortingAccumulator(studio.EventAccumulator):
    """The studio's accumulator, with its rendered frame optionally remapped."""

    def __init__(self, width, height, accumulation_us):
        super().__init__(width, height, accumulation_us)
        self.maps = None             # remap tables, or None for raw

    def render(self, palette):
        frame = super().render(palette)
        if self.maps is None:
            return frame
        # Nearest, not linear: events are single pixels, and blending them
        # into the background would dim and smear them.
        return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=palette[1])


class DualStudio(studio.Studio):
    """Only the preview and the recording start/stop differ; rest is inherited."""

    def __init__(self, args):
        super().__init__(args)
        self.webcam = None
        self.rec_folder = None
        self.t_events_started = None
        self.cam_index = args.camera_index

        self.calib_dir = args.calib_dir or self.record_dir
        self.ev_cal, note = load_calib(os.path.join(self.calib_dir, "event.calib"),
                                       self.width, self.height)
        print("[calib] event  " + note)
        self.accumulator = UndistortingAccumulator(self.width, self.height,
                                                   self.accumulation_us)
        if args.undistort and self.ev_cal is not None:
            self.accumulator.maps = self.ev_cal.maps(self.width, self.height)
        self.vis_path = os.path.join(self.calib_dir, "vis.calib")
        self.vis_cal = None          # loaded once the webcam size is known
        self.und_vis = False
        self._vis_maps = {}          # preview (w, h) -> remap tables

        if self.offline:
            return
        if self.cam_index is None:
            self.cam_index = pick_camera(args.camera_probe_max)
            if self.cam_index is None:
                print("[cam2] no webcam found - events only")
                return
            print("[cam2] auto-selected camera index %d "
                  "(use --camera-index to override)" % self.cam_index)
        feed = WebcamFeed(self.cam_index, args.camera_fps, args.camera_size)
        try:
            w, h = feed.open()
        except OSError as exc:
            print("[cam2] %s - events only" % exc)
            return
        self.webcam = feed
        print("[cam2] live at %dx%d from index %d" % (w, h, self.cam_index))
        self.vis_cal, note = load_calib(self.vis_path, w, h)
        print("[calib] webcam " + note)
        self.und_vis = args.undistort and self.vis_cal is not None
        if args.camera_preview_width > 0:
            cv2.namedWindow(CAM_WIN, cv2.WINDOW_AUTOSIZE)

    # -- preview ------------------------------------------------------------
    def _draw(self, last_ts):
        """Inherit the whole event view, then paint the webcam beside it.

        Done here rather than in the grab thread because OpenCV's HighGUI is
        only safe from the thread that owns the windows, which is this one.
        """
        super()._draw(last_ts)
        if self.webcam is None or self.args.camera_preview_width <= 0:
            return
        frame = self.webcam.latest()
        if frame is None:
            return
        width = self.args.camera_preview_width
        if frame.shape[1] != width:
            scale = width / float(frame.shape[1])
            frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))))
        else:
            frame = frame.copy()
        if self.und_vis:
            # Undistort after the resize: the same result at preview scale,
            # without remapping a full 1080p frame every draw.
            size = (frame.shape[1], frame.shape[0])
            if size not in self._vis_maps:
                cal, _ = load_calib(self.vis_path, size[0], size[1])
                self._vis_maps[size] = cal.maps(size[0], size[1])
            mx, my = self._vis_maps[size]
            frame = cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
            cv2.putText(frame, "undistorted", (frame.shape[1] - 120, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        if self.webcam.recording:
            cv2.circle(frame, (18, 20), 7, (0, 0, 255), -1)
            cv2.putText(frame, "REC %d" % self.webcam.frames, (32, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(CAM_WIN, frame)

    def _handle_keys(self, key):
        code = key & 0xFF if key != -1 else -1
        if code == ord("v"):
            if self.vis_cal is None:
                self.notify("no usable vis.calib in %s" % self.calib_dir)
            else:
                self.und_vis = not self.und_vis
                self.notify("webcam preview %s"
                            % ("undistorted" if self.und_vis else "raw"))
        elif code == ord("u"):
            if self.ev_cal is None:
                self.notify("no usable event.calib in %s" % self.calib_dir)
            elif self.accumulator.maps is None:
                self.accumulator.maps = self.ev_cal.maps(self.width, self.height)
                self.notify("event preview undistorted")
            else:
                self.accumulator.maps = None
                self.notify("event preview raw")
        else:
            super()._handle_keys(key)

    # -- recording ----------------------------------------------------------
    def toggle_recording(self):
        stream = self.device.get_i_events_stream()
        if stream is None:
            self.notify("no events stream to record")
            return

        if self.recording_path:
            stream.stop_log_raw_data()
            summary = os.path.basename(self.rec_folder or self.recording_path)
            if self.webcam is not None and self.webcam.recording:
                frames, measured = self.webcam.stop_recording()
                self._write_meta(frames, measured)
                summary += "  (%d webcam frames, %.1f fps measured)" % (
                    frames, measured)
                if self.webcam.error:
                    self.notify("webcam error: %s" % self.webcam.error, 8.0)
            self.notify("recording stopped: %s" % summary, 6.0)
            self.recording_path = None
            self.rec_folder = None
            return

        if self.offline:
            self.notify("cannot record while replaying a file")
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.record_dir, "rec_%s" % stamp)
        os.makedirs(folder, exist_ok=True)

        if self.webcam is not None:
            try:
                self.webcam.start_recording(folder)
            except OSError as exc:
                self.notify("webcam: %s - events only" % exc, 8.0)

        raw_path = os.path.join(folder, "events.raw")
        self.t_events_started = time.time()
        stream.log_raw_data(raw_path)
        self.recording_path = raw_path
        self.rec_folder = folder
        studio.write_bias_file(os.path.join(folder, "events.bias"),
                               self.biases.values)
        for name in CALIB_FILES:
            src = os.path.join(self.calib_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(folder, name))
        self.notify("recording to %s" % os.path.basename(folder), 5.0)

    def _write_meta(self, frames, measured):
        cam = self.webcam
        with open(os.path.join(self.rec_folder, "meta.txt"), "w") as fh:
            fh.write("recorded          %s\n" % datetime.datetime.now().isoformat())
            fh.write("camera index      %s\n" % self.cam_index)
            fh.write("camera size       %dx%d\n" % (cam.width, cam.height))
            fh.write("camera frames     %d\n" % frames)
            fh.write("container fps     %.3f   (nominal - what the .mp4 claims)\n"
                     % cam.fps)
            fh.write("measured fps      %.3f   (from camera_frames.csv - the truth)\n"
                     % measured)
            fh.write("webcam started    %.6f\n" % (cam.t_open or 0.0))
            if self.t_events_started is not None and cam.t_open:
                fh.write("events started    %.6f\n" % self.t_events_started)
                fh.write("start offset      %+.4f s  (events minus webcam)\n"
                         % (self.t_events_started - cam.t_open))
            if cam.t_stop and cam.t_open:
                fh.write("duration          %.3f s\n" % (cam.t_stop - cam.t_open))
            fh.write("\nbiases in force:\n")
            for name in studio.BIAS_NAMES:
                fh.write("  %-16s %d\n" % (name, self.biases.values.get(name, -1)))

    def shutdown(self):
        if self.webcam is not None:
            try:
                if self.webcam.recording:
                    frames, measured = self.webcam.stop_recording()
                    if self.rec_folder:
                        self._write_meta(frames, measured)
                    print("[cam2] stopped: %d frames, %.1f fps" % (frames, measured))
                self.webcam.close()
            except Exception as exc:
                print("[cam2] error on shutdown: %s" % exc)
            self.webcam = None
            try:
                cv2.destroyWindow(CAM_WIN)
            except cv2.error:
                pass
        super().shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_EXTRA_HELP = """
dual_record.py adds these to silky_bias_studio's flags:

  --list-cameras            probe indices 0-N, print what works, and exit
  --camera-index N          which webcam to use (default: highest working
                            index, since the built-in is usually 0)
  --camera-fps F            requested webcam rate, and the .mp4's nominal rate
                            (default 30). The camera may ignore it; the
                            measured rate goes in meta.txt.
  --camera-size WxH         requested capture size (default 1920x1080, the
                            size vis.calib was done at)
  --camera-preview-width N  width of the live preview window, 0 to hide it
                            (default 640)
  --camera-probe-max N      how many indices to try (default 8)
  -b / --record-dir         defaults changed here: freqcam_quadrotor.bias and
                            runs/, both next to this script
  --calib-dir DIR           folder holding vis.calib and event.calib
                            (default: the record dir)
  --undistort               start with both previews undistorted (each only
                            if its .calib is filled in). Keys: v webcam,
                            u events. Recordings are always raw.
"""


def _size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("expected WxH, e.g. 1280x720")


def parse_args(argv=None):
    """Our flags first, then hand everything else to the studio unchanged."""
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--list-cameras", action="store_true")
    extra.add_argument("--camera-index", type=int, default=None)
    extra.add_argument("--camera-fps", type=float, default=30.0)
    extra.add_argument("--camera-size", type=_size, default=(1920, 1080))
    extra.add_argument("--camera-preview-width", type=int, default=640)
    extra.add_argument("--camera-probe-max", type=int, default=8)
    extra.add_argument("--calib-dir", default=None)
    extra.add_argument("--undistort", action="store_true")
    mine, rest = extra.parse_known_args(argv)
    if argv is None:
        argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(_EXTRA_HELP)            # studio's parser prints its own and exits
    args = studio.parse_args(rest)
    # Our defaults, not the studio's: both anchored to this script's folder so
    # they hold whatever directory it is launched from.
    if not args.bias:
        args.bias = DEFAULT_BIAS
    if not args.record_dir:
        args.record_dir = DEFAULT_RECORD_DIR
    for key, value in vars(mine).items():
        setattr(args, key, value)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_cameras:
        found = probe_cameras(args.camera_probe_max)
        if not found:
            print("no working cameras found")
            return 1
        print("working camera indices:")
        for idx, w, h, fps in found:
            print("  %d : %dx%d  %.0f fps reported%s"
                  % (idx, w, h, fps, "   <- would be auto-selected"
                     if idx == found[-1][0] else ""))
        return 0
    try:
        app = DualStudio(args)
    except (OSError, FileNotFoundError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
