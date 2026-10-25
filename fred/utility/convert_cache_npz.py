"""
convert_cache_npz.py — binary (.npz) mirrors of the frame_data CSV caches.

Loading frames_<seq>_<mode>.csv.gz costs ~13 s for a dense case (gzip + text
parsing); the same columns as a compressed .npz load in ~1.5 s at roughly half
the file size. This script converts every cached case of the given mode(s) and
VERIFIES each conversion by reloading and comparing arrays. frame_explorer's
load_D prefers the .npz when it exists and silently falls back to the CSV, so
conversion is safe to run at any time and the CSVs remain the source of truth
(pass --delete-csv only if you need the disk back).

Usage:
    python convert_cache_npz.py --mode pixelize
    python convert_cache_npz.py --mode baseline,pixelize --workers 4
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_DT = {"frame": np.int64, "x": np.int32, "y": np.int32, "n_events": np.int32,
       "blob_size": np.int32, "sf": np.float32, "snr": np.float32}
COLS = list(_DT)


def convert_one(job):
    csv_path, npz_path = job
    import pandas as pd
    t0 = time.perf_counter()
    try:
        try:
            df = pd.read_csv(csv_path, dtype=_DT)
        except pd.errors.EmptyDataError:
            df = None
        arrs = ({c: df[c].to_numpy() for c in COLS} if df is not None and not df.empty
                else {c: np.array([], _DT[c]) for c in COLS})
        tmp = npz_path + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        z = np.load(tmp)                             # verify before replacing
        for c in COLS:
            if not np.array_equal(z[c], arrs[c]):
                raise ValueError(f"verify failed on column {c}")
        z.close()
        os.replace(tmp, npz_path)
        return (os.path.basename(csv_path), arrs["frame"].size,
                round(time.perf_counter() - t0, 1), None)
    except Exception as exc:
        return (os.path.basename(csv_path), 0, 0.0, repr(exc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="baseline",
                    help="Comma-separated modes to convert (default baseline).")
    ap.add_argument("--data-dir", default=os.path.join(_THIS_DIR, "frame_data"))
    ap.add_argument("--workers", type=int, default=2,
                    help="Conversion processes (default 2 — gentle while "
                         "searches are running).")
    ap.add_argument("--force", action="store_true",
                    help="Reconvert even when the .npz already exists.")
    ap.add_argument("--delete-csv", action="store_true",
                    help="Delete each CSV after a VERIFIED conversion.")
    args = ap.parse_args()

    jobs = []
    for mode in [m.strip() for m in args.mode.split(",") if m.strip()]:
        import glob
        for csv_path in sorted(glob.glob(os.path.join(
                args.data_dir, f"frames_*_{mode}.csv.gz"))):
            npz_path = csv_path[:-len(".csv.gz")] + ".npz"
            if args.force or not os.path.exists(npz_path):
                jobs.append((csv_path, npz_path))
    print(f"{len(jobs)} case(s) to convert")
    if not jobs:
        return

    n_err = 0
    with mp.Pool(max(1, args.workers)) as pool:
        for i, (name, rows, dt, err) in enumerate(
                pool.imap_unordered(convert_one, jobs), 1):
            if err:
                n_err += 1
                print(f"[{i}/{len(jobs)}] {name}: FAILED {err}")
            else:
                print(f"[{i}/{len(jobs)}] {name}: {rows} rows ({dt}s)")
    if args.delete_csv and not n_err:
        for csv_path, npz_path in jobs:
            if os.path.exists(npz_path):
                os.remove(csv_path)
        print("CSVs deleted.")
    print(f"done, {n_err} failure(s)")


if __name__ == "__main__":
    main()
