"""Undistort a session's PADDED_RGB frames in place.

The originals are moved to PADDED_RGB_DISTORTED and PADDED_RGB is rewritten with
the undistorted images, keeping every filename identical so the timestamps the
pipeline parses are preserved.

main.py calls undistort_session() automatically once a recording finishes; the
CLI here is for sessions recorded before that, or for re-running with different
coefficients.

K is calibration-resolution specific, so it is rescaled through the two steps
main.py applied: calibration size -> capture size -> letterboxed into the event
geometry. Undistortion runs on the content region only; the black bars are
re-added afterwards so they stay straight instead of being warped.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# MATLAB cameraParams for the RYS HFR camera (Intrinsics.K, 1-based pixels).
K_CALIB = np.array([
    [8.627759194466139e+02, 0.0,                   1.022968154310892e+03],
    [0.0,                   8.687430567201790e+02, 4.777874586799277e+02],
    [0.0,                   0.0,                   1.0],
], dtype=np.float64)
RADIAL = (-0.2434, 0.0466)      # Intrinsics.RadialDistortion  [k1 k2]
TANGENTIAL = (0.0, 0.0)         # Intrinsics.TangentialDistortion [p1 p2]
CALIB_SIZE = (1920, 1080)       # Intrinsics.ImageSize, as (width, height)
CAPTURE_SIZE = (1280, 720)      # main.py --rgb-size default

DISTORTED_DIRNAME = "PADDED_RGB_DISTORTED"


def wh(text):
    w, h = (int(v) for v in str(text).lower().split("x"))
    return w, h


def letterbox_geometry(src_w, src_h, dst_w, dst_h):
    """Reproduce main.py:pad_to - uniform scale, centred, black bars."""
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    return scale, new_w, new_h, (dst_w - new_w) // 2, (dst_h - new_h) // 2


def content_intrinsics(pad_w, pad_h, calib_size, capture_size):
    """K and the letterbox geometry for the padded frames' content region."""
    calib_w, calib_h = calib_size
    cap_w, cap_h = capture_size
    if abs(calib_w / calib_h - cap_w / cap_h) > 0.01:
        raise ValueError(
            f"aspect mismatch: calibration {calib_w}x{calib_h} vs capture {cap_w}x{cap_h}. "
            "A non-uniform scale between them invalidates the radial model - "
            "recalibrate at the capture resolution instead.")

    s_cap = cap_w / calib_w                                    # calibration -> capture
    geom = letterbox_geometry(cap_w, cap_h, pad_w, pad_h)      # capture -> content region
    s = s_cap * geom[0]

    K = K_CALIB.copy()
    K[0, 2] -= 1.0      # MATLAB pixel centres are 1-based; OpenCV is 0-based
    K[1, 2] -= 1.0
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    return K, geom, s_cap


def undistort_session(session, radial=RADIAL, tangential=TANGENTIAL,
                      calib_size=CALIB_SIZE, capture_size=CAPTURE_SIZE,
                      dry_run=False, verbose=True):
    """Undistort session/PADDED_RGB in place. Returns the number of frames written.

    Returns 0 (and does nothing) if there is no PADDED_RGB to work on.
    Raises FileExistsError if the session has already been undistorted.
    """
    session = Path(session)
    rgb_dir = session / "PADDED_RGB"
    backup = session / DISTORTED_DIRNAME
    if not rgb_dir.is_dir():
        return 0
    if backup.exists():
        raise FileExistsError(f"{backup} already exists - refusing to overwrite it")

    files = sorted(rgb_dir.glob("*.jpg"))
    if not files:
        return 0

    k1, k2 = radial[0], radial[1]
    k3 = radial[2] if len(radial) > 2 else 0.0
    dist = np.array([k1, k2, tangential[0], tangential[1], k3], dtype=np.float64)

    probe = cv2.imread(str(files[0]))
    pad_h, pad_w = probe.shape[:2]
    K, (s_pad, content_w, content_h, off_x, off_y), s_cap = content_intrinsics(
        pad_w, pad_h, calib_size, capture_size)

    if verbose:
        print(f"undistort: {len(files)} frames at {pad_w}x{pad_h}, "
              f"content {content_w}x{content_h} at ({off_x},{off_y})")
        print(f"  K scaled by {s_cap:.4f} x {s_pad:.4f} -> fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
              f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")
        print(f"  D = {np.array2string(dist, precision=6)}")
    if dry_run:
        if verbose:
            print("  --dry-run: nothing written")
        return 0

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (content_w, content_h), 0)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (content_w, content_h), cv2.CV_16SC2)

    rgb_dir.rename(backup)
    rgb_dir.mkdir()
    for src in sorted(backup.glob("*.jpg")):
        img = cv2.imread(str(src))
        content = img[off_y:off_y + content_h, off_x:off_x + content_w]
        fixed = cv2.remap(content, map1, map2, cv2.INTER_LINEAR)
        out = np.zeros_like(img)
        out[off_y:off_y + content_h, off_x:off_x + content_w] = fixed
        cv2.imwrite(str(rgb_dir / src.name), out)

    if verbose:
        print(f"  {len(files)} frames undistorted, originals -> {DISTORTED_DIRNAME}/")
    return len(files)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, help="session folder containing PADDED_RGB")
    p.add_argument("--radial", default=",".join(str(v) for v in RADIAL),
                   help=f"Intrinsics.RadialDistortion (default {RADIAL})")
    p.add_argument("--tangential", default=",".join(str(v) for v in TANGENTIAL),
                   help=f"Intrinsics.TangentialDistortion (default {TANGENTIAL})")
    p.add_argument("--calib-size", default="x".join(str(v) for v in CALIB_SIZE),
                   help=f"resolution K was calibrated at (default {CALIB_SIZE[0]}x{CALIB_SIZE[1]})")
    p.add_argument("--capture-size", default="x".join(str(v) for v in CAPTURE_SIZE),
                   help=f"resolution main.py captured at (default {CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]})")
    p.add_argument("--dry-run", action="store_true",
                   help="report the derived matrices and exit without writing")
    return p.parse_args()


def main():
    args = parse_args()
    radial = tuple(float(v) for v in args.radial.replace(" ", "").split(","))
    tangential = tuple(float(v) for v in args.tangential.replace(" ", "").split(","))
    if len(radial) not in (2, 3) or len(tangential) != 2:
        raise SystemExit("--radial takes 2 or 3 values, --tangential takes 2")

    try:
        n = undistort_session(args.session, radial, tangential,
                              wh(args.calib_size), wh(args.capture_size),
                              dry_run=args.dry_run)
    except (ValueError, FileExistsError) as e:
        raise SystemExit(str(e))
    if n == 0 and not args.dry_run:
        raise SystemExit(f"no PADDED_RGB frames found in {args.session}")


if __name__ == "__main__":
    main()
