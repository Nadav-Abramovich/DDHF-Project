"""
Build a per-sequence "which drone(s)" map for the FRED dataset WITHOUT
downloading the multi-GB zips.

HuggingFace serves the dataset's LFS zips over HTTP with range-request support,
so remotezip can read just each archive's central directory + the tiny
coordinates.txt out of it. coordinates.txt lines are:

    time: x1, y1, x2, y2, id, class

We collect the distinct `class` labels (the drone models) per sequence and
write them to fred/data/fred_drone_map.csv. Multiple drones in one sequence -> multiple
classes, separated by '|'.

Usage:
    pip install remotezip requests
    python fred_drone_map.py
"""
import csv
import os
import sys
import requests
from remotezip import RemoteZip

REPO = "GabrieleMagrini/FRED"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                   "fred_drone_map.csv")


def list_zips(split):
    """Return sorted sequence zip names under a split dir, e.g. ('0.zip', ...)."""
    r = requests.get(f"{API}/{split}?recursive=false", timeout=60)
    r.raise_for_status()
    names = [e["path"].split("/")[-1] for e in r.json()
             if e["type"] == "file" and e["path"].lower().endswith(".zip")]
    return sorted(names, key=lambda n: int(n.split(".")[0]))


def classes_in(url):
    """Distinct drone class labels in the zip's coordinates.txt."""
    with RemoteZip(url) as z:
        cand = [n for n in z.namelist() if n.lower().endswith("coordinates.txt")]
        if not cand:
            return []
        seen = {}                       # preserve first-seen order
        for line in z.read(cand[0]).decode("utf-8", "replace").splitlines():
            if ":" not in line:
                continue
            parts = line.split(":", 1)[1].split(",")
            if len(parts) >= 6:
                cls = parts[5].strip()
                if cls:
                    seen[cls] = None
        return list(seen)


def main():
    rows = []
    for split in ("train", "test"):
        try:
            zips = list_zips(split)
        except Exception as e:
            print(f"!! could not list {split}: {e}")
            continue
        print(f"{split}: {len(zips)} sequences")
        for name in zips:
            seq = name.split(".")[0]
            url = f"{BASE}/{split}/{name}"
            try:
                cls = classes_in(url)
                label = "|".join(cls) if cls else "(none)"
            except Exception as e:
                label = f"ERROR: {type(e).__name__}: {e}"
            print(f"  {split}/{seq:>4}: {label}")
            rows.append((split, int(seq), label))
            # Write incrementally so partial progress survives interruption.
            with open(OUT, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["split", "sequence", "drone_class(es)"])
                w.writerows(sorted(rows, key=lambda r: (r[0], r[1])))
    print(f"\nWrote {OUT} ({len(rows)} sequences).")


if __name__ == "__main__":
    main()
