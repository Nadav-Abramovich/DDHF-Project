import argparse
import multiprocessing as mp
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from metavision_sdk_stream import (
    Camera,
    CameraStreamSlicer,
    HDF5EventFileWriter,
    RAWEvt2EventFileWriter,
    SliceCondition,
)
from undistort_padded import undistort_session

SLICE_US = 20_000        # 20 ms event slices -> 50 writes/second
DEFAULT_BASE = Path(__file__).resolve().parent.parent   # lab root
SESSION_PREFIX = "lab"   # sessions are lab1, lab2, ... so they never collide
                         # with the numeric sequence folders of the dataset

# The external RYS HFR camera is DirectShow index 1; MSMF opens it but cannot
# pull frames. Index 0 is the laptop's built-in ASUS webcam.
RGB_INDEX = 1
RGB_BACKEND = cv2.CAP_DSHOW


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record event + RGB data into a dataset-compatible session folder.")
    parser.add_argument("-d", "--duration", type=float, default=5.0,
                        help="recording duration in seconds (default: 5)")
    parser.add_argument("-b", "--base", type=Path, default=DEFAULT_BASE,
                        help=f"dataset root to create the session in (default: {DEFAULT_BASE})")
    parser.add_argument("-s", "--session", default=None,
                        help="session folder name (default: next free lab<N>)")
    parser.add_argument("-f", "--format", choices=("raw", "hdf5"), default="raw",
                        help="raw is EVT2, readable by metavision_core RawReader (default)")
    parser.add_argument("--rgb-index", type=int, default=RGB_INDEX,
                        help=f"OpenCV/DirectShow camera index (default: {RGB_INDEX} = RYS HFR)")
    parser.add_argument("--rgb-size", default="1280x720",
                        help="capture size; camera supports 640x360, 1280x720, 1920x1080")
    parser.add_argument("--rgb-fps", type=float, default=30.0,
                        help="frames saved per second (default: 30, matching the 33.3 ms window)")
    parser.add_argument("--no-rgb", action="store_true",
                        help="record events only, skip the RGB camera")
    parser.add_argument("--no-undistort", action="store_true",
                        help="keep the raw lens distortion instead of correcting it afterwards")
    return parser.parse_args()


def next_session_dir(base: Path) -> Path:
    """First unused base/lab<N>. Numeric dataset sequence folders are ignored,
    so this never proposes a name the dataset already uses."""
    used = set()
    if base.is_dir():
        for entry in base.iterdir():
            m = re.fullmatch(rf"{SESSION_PREFIX}(\d+)", entry.name)
            if m:
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return base / f"{SESSION_PREFIX}{n}"


def pad_to(frame, width, height):
    """Letterbox `frame` into width x height: scale to fit, centre, pad black.
    Puts RGB pixels on the same coordinate scale as the event sensor."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    return cv2.copyMakeBorder(resized, top, height - new_h - top,
                              left, width - new_w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def rgb_process(index, size, save_fps, out_dir, session, pad_w, pad_h,
                ready, go, stop, counter, actual_wh):
    """Capture the RGB camera in a separate PROCESS.

    A thread is not enough: iterating the Metavision slicer holds the GIL for
    much of each slice, which starved a capture thread down to ~10 fps. In its
    own process the capture keeps the camera's full rate. This also keeps every
    DirectShow call on one thread, which DirectShow's apartment model requires.

    Sets `ready` once the camera is warm, waits for `go` so the first saved frame
    lines up with event t0, and runs until `stop`.
    """
    cap = cv2.VideoCapture(index, RGB_BACKEND)
    if not cap.isOpened():
        ready.set()
        return
    # This camera only offers MJPEG; without the fourcc the driver falls back to
    # its slowest mode. Properties must be set before the first read.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    try:
        for _ in range(10):  # let auto-exposure settle before recording starts
            if not cap.read()[0]:
                ready.set()
                return
        actual_wh[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_wh[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir.mkdir(parents=True, exist_ok=True)
        ready.set()

        while not go.is_set() and not stop.is_set():
            cap.read()  # keep the pipeline fresh while events spin up

        # Pace with read(), never a grab() spin: on this camera's DSHOW backend
        # grab() returns the SAME buffered sample immediately without waiting
        # for a new frame, so a grab()-spin + occasional retrieve() saves one
        # frozen frame over and over (that bug shipped a session of 83
        # byte-identical JPEGs). read() decodes every delivered frame (~8 ms) -
        # affordable in this dedicated process - and each save is the newest
        # live frame. The camera free-runs above save_fps (~98 fps at 720p),
        # so the clock gate simply keeps one frame per save interval.
        min_interval = 1.0 / save_fps
        next_save = time.perf_counter()
        last_sig = None
        dups = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.perf_counter()
            if now < next_save:
                continue
            next_save = now + min_interval
            # Tripwire: consecutive byte-identical frames mean the stream has
            # gone stale again (sensor noise makes real frames always differ).
            sig = frame[::64, ::64].tobytes()
            if sig == last_sig:
                dups += 1
            last_sig = sig
            ts = datetime.now()
            name = f"Video_{session}_{ts:%H_%M_%S}.{ts.microsecond:06d}.jpg"
            cv2.imwrite(str(out_dir / name), pad_to(frame, pad_w, pad_h))
            counter.value += 1
        if dups:
            print(f"[rgb] WARNING: {dups}/{counter.value} frames were identical to "
                  f"their predecessor - the capture stream was stale, re-record")
    finally:
        cap.release()


def main():
    args = parse_args()
    try:
        rgb_size = tuple(int(v) for v in args.rgb_size.lower().split("x"))
        if len(rgb_size) != 2:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--rgb-size must look like 1280x720, got {args.rgb_size!r}")

    session_dir = args.base / args.session if args.session else next_session_dir(args.base)
    if session_dir.exists():
        raise SystemExit(f"refusing to overwrite existing session: {session_dir}")
    event_dir = session_dir / "Event"
    event_dir.mkdir(parents=True)
    duration_us = int(args.duration * 1_000_000)

    ready, go, stop = (mp.Event() for _ in range(3))
    counter = mp.Value("i", 0)
    actual_wh = mp.Array("i", [0, 0])

    camera = Camera.from_first_available()
    slicer = CameraStreamSlicer(camera.move(), SliceCondition.make_n_us(SLICE_US))
    width = slicer.camera().width()
    height = slicer.camera().height()

    proc = None
    if not args.no_rgb:
        # Created after the event camera opens: frames are padded to its geometry.
        proc = mp.Process(
            target=rgb_process,
            args=(args.rgb_index, rgb_size, args.rgb_fps, session_dir / "PADDED_RGB",
                  session_dir.name, width, height, ready, go, stop, counter, actual_wh),
            daemon=True)
        proc.start()
        ready.wait(timeout=20)
        if actual_wh[0] == 0:
            print(f"WARNING: RGB camera index {args.rgb_index} unavailable - recording events only")
            stop.set()
            proc.join(timeout=5)
            proc = None
        else:
            print(f"rgb {actual_wh[0]}x{actual_wh[1]} -> padded to {width}x{height} "
                  f"@ {args.rgb_fps:g} fps")

    output = event_dir / ("events.hdf5" if args.format == "hdf5" else "events.raw")
    metadata = {"geometry": f"{width}x{height}"}
    if args.format == "hdf5":
        writer = HDF5EventFileWriter(output, metadata)
    else:
        writer = RAWEvt2EventFileWriter(width, height, output, False, metadata)

    print(f"session {session_dir.name}: event {width}x{height}, recording {args.duration}s")

    total = 0
    reordered = 0
    dropped = 0
    last_written = None
    first_t = None
    last_t = 0
    try:
        for slice_ in slicer:
            # Release RGB on the first slice so frame 0 lines up with event t0 -
            # code/utils.py measures RGB time relative to the first frame.
            if first_t is None:
                first_t = slice_.t
                go.set()

            if slice_.n_events:
                events = slice_.events
                # This camera emits a few out-of-order timestamps per second,
                # both inside a slice and across slice boundaries. Both writers
                # require a monotonic stream: HDF5 raises "events are older than
                # last added buffer", and the EVT2 encoder silently overruns its
                # buffer and kills the process with 0xC0000409. So sort each
                # slice, then drop anything still older than the last event
                # written. The drops are a handful out of ~50k events/s.
                times = events["t"]
                if times.size > 1 and np.any(np.diff(times) < 0):
                    events = events[np.argsort(times, kind="stable")]
                    reordered += 1
                if last_written is not None:
                    keep = events["t"] >= last_written
                    if not keep.all():
                        dropped += int((~keep).sum())
                        events = events[keep]
                if events.size:
                    writer.add_cd_events(events)
                    last_written = int(events["t"][-1])
                    total += events.size

            last_t = slice_.t
            if last_t - first_t >= duration_us:
                break
    except KeyboardInterrupt:
        print("\ninterrupted - closing files")
    finally:
        stop.set()
        if proc is not None:
            proc.join(timeout=10)
        writer.close()

    elapsed = (last_t - first_t) / 1_000_000 if first_t is not None else 0.0
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"{total:,} events over {elapsed:.3f}s -> Event/{output.name} ({size_mb:.1f} MB)")
    if reordered or dropped:
        print(f"  ({reordered} slices resorted, {dropped} out-of-order events dropped)")
    if proc is not None:
        rate = counter.value / elapsed if elapsed else 0.0
        print(f"{counter.value} RGB frames -> PADDED_RGB/ ({rate:.1f} fps)")

    # Correct lens distortion once recording is done rather than per frame, so
    # nothing competes with the cameras while they are streaming. Never let this
    # lose a recording: the frames are already safely on disk by now.
    if proc is not None and not args.no_undistort and counter.value:
        try:
            undistort_session(session_dir, capture_size=rgb_size)
        except Exception as e:
            print(f"WARNING: undistortion failed ({type(e).__name__}: {e})")
            print("         frames are intact in PADDED_RGB; "
                  "rerun with undistort_padded.py")

    print(f"session folder: {session_dir}")


if __name__ == "__main__":
    main()
