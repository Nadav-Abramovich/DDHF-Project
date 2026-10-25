"""FRED case data: on-demand download from HuggingFace + case-folder layout.

Dataset: https://huggingface.co/datasets/GabrieleMagrini/FRED
One zip per case under train/ or test/; each expands to a top-level <seq>/ with

    <seq>/Event/events.hdf5      event stream (HDF5 - NOT the legacy .raw)
    <seq>/Event/Frames/*.png     rendered event frames  (frame_<us>.png)
    <seq>/RGB/*.jpg              RGB frames, ALREADY padded to the event geometry
    <seq>/coordinates.txt        ground truth
    <seq>/coordinates_rgb.txt

Older locally-prepared cases use Event/events.raw and PADDED_RGB/ instead. The
helpers below accept either layout, so both kinds of case folder load unchanged:

    event_file(case_dir)  ->  events.raw if present, else events.hdf5
    rgb_dir(case_dir)     ->  PADDED_RGB/ if present, else RGB/

Lab copy: kept so the copied modules import unchanged; the lab tools never
download (they always pass --no-download).
"""
import io
import os
import json
import zipfile
import urllib.request

REPO      = "GabrieleMagrini/FRED"
_API      = "https://huggingface.co/api/datasets/%s/tree/main/" % REPO
_RESOLVE  = "https://huggingface.co/datasets/%s/resolve/main/" % REPO
_SPLITS   = ("train", "test")
_TIMEOUT  = 60

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# lab copy: the lab never downloads FRED cases (its tools pass --no-download);
# these defaults only keep the module inside lab/
_DEFAULT_HF_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "data")
DEFAULT_CASES_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "runs", "dataset")


# ── case-folder layout (HF or legacy) ────────────────────────────────────────────

def event_file(case_dir):
    """The case's event stream: the legacy Event/events.raw when it exists,
    otherwise the HF Event/events.hdf5. Raises when neither is there."""
    ev = os.path.join(case_dir, "Event")
    for name in ("events.raw", "events.hdf5"):
        p = os.path.join(ev, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "no events.raw or events.hdf5 under %s" % ev)


def rgb_dir(case_dir):
    """The case's RGB frame folder: legacy PADDED_RGB/ when it exists, otherwise
    the HF RGB/ (which is already padded to the event geometry)."""
    for name in ("PADDED_RGB", "RGB"):
        d = os.path.join(case_dir, name)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        "no PADDED_RGB/ or RGB/ under %s" % case_dir)


def is_hdf5(path):
    return str(path).lower().endswith((".hdf5", ".h5"))


# ── HuggingFace index ────────────────────────────────────────────────────────────

def _index_path(data_dir=None):
    return os.path.join(data_dir or _DEFAULT_HF_DIR, "hf_index.json")


def case_index(data_dir=None, refresh=False):
    """{seq: split} for every case in the dataset, cached in data/hf_index.json
    so the listing is fetched once. Falls back to an empty map when offline."""
    p = _index_path(data_dir)
    if not refresh and os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    idx = {}
    for split in _SPLITS:
        try:
            with urllib.request.urlopen(_API + split, timeout=_TIMEOUT) as r:
                for e in json.load(r):
                    name = os.path.basename(e.get("path", ""))
                    if name.endswith(".zip"):
                        idx[name[:-4]] = split
        except Exception as exc:                     # offline / API change
            print("[hf] could not list %s/: %s" % (split, exc))
    if idx:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(idx, fh, indent=1, sort_keys=True)
        except Exception:
            pass
    return idx


def _zip_url(seq, data_dir=None):
    """The download URL for a case, resolving its split (train/ or test/)."""
    seq = str(seq)
    split = case_index(data_dir).get(seq)
    if split:
        return _RESOLVE + "%s/%s.zip" % (split, seq)
    for split in _SPLITS:                            # index unavailable: probe
        url = _RESOLVE + "%s/%s.zip" % (split, seq)
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=_TIMEOUT).close()
            return url
        except Exception:
            continue
    return None


# ── download + extract ───────────────────────────────────────────────────────────

def _safe_extract(zf, dest, progress=None):
    """Extract rejecting absolute paths and .. traversal (zip-slip).
    progress(done_bytes, total_bytes) is called after each member."""
    dest_abs = os.path.abspath(dest)
    members = zf.infolist()
    for m in members:
        target = os.path.abspath(os.path.join(dest, m.filename))
        if not (target + os.sep).startswith(dest_abs + os.sep):
            raise ValueError("refusing unsafe zip member: %s" % m.filename)
    total = sum(m.file_size for m in members) or 1
    done = 0
    for m in members:
        zf.extract(m, dest)
        done += m.file_size
        if progress:
            progress(done, total)


def download_case(seq, cases_dir, keep_zip=False, data_dir=None, progress=None):
    """Download <split>/<seq>.zip and extract it into cases_dir.

    The archive already contains a top-level <seq>/ folder, so the result is
    cases_dir/<seq>/. Returns that path, or None when the case is unavailable.

    progress(phase, done, total), when given, is called as the work advances:
    phase "download" (bytes; total 0 when the server sends no length) and then
    "extract" (uncompressed bytes). It runs on the calling thread."""
    def _report(phase, done, total):
        if progress:
            try:
                progress(phase, done, total)
            except Exception:
                pass

    seq = str(seq)
    _report("lookup", 0, 0)
    url = _zip_url(seq, data_dir)
    if url is None:
        print("[hf] case %s is not in the FRED dataset (train/ or test/)" % seq)
        return None

    os.makedirs(cases_dir, exist_ok=True)
    zip_path = os.path.join(cases_dir, seq + ".zip")
    part     = zip_path + ".part"
    print("[hf] downloading case %s from %s" % (seq, url))
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done  = 0
            step  = max(total // 20, 1 << 22) if total else (1 << 24)
            nxt   = step
            with open(part, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    _report("download", done, total)
                    if done >= nxt:
                        nxt += step
                        if total:
                            print("[hf]   %5.1f%%  (%.0f/%.0f MB)"
                                  % (100.0 * done / total, done / 1e6, total / 1e6))
                        else:
                            print("[hf]   %.0f MB" % (done / 1e6))
        os.replace(part, zip_path)
    except Exception as exc:
        print("[hf] download failed: %s" % exc)
        for p in (part, zip_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return None

    print("[hf] extracting -> %s" % cases_dir)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, cases_dir,
                          progress=lambda d, t: _report("extract", d, t))
    except Exception as exc:
        print("[hf] extract failed: %s" % exc)
        return None
    finally:
        if not keep_zip:
            try:
                os.remove(zip_path)
            except OSError:
                pass

    out = os.path.join(cases_dir, seq)
    if not os.path.isdir(out):                       # unexpected archive shape
        print("[hf] extracted, but %s/ is not there - check the archive" % seq)
        return None
    print("[hf] case %s ready: %s" % (seq, out))
    return out


def ensure_case(seq, cases_dir, download=True, data_dir=None, progress=None):
    """cases_dir/<seq> if already present, else download it. None on failure."""
    out = os.path.join(cases_dir, str(seq))
    if os.path.isdir(out):
        return out
    if not download:
        return None
    return download_case(seq, cases_dir, data_dir=data_dir, progress=progress)


if __name__ == "__main__":                           # tiny CLI: fetch some cases
    import argparse
    ap = argparse.ArgumentParser(description="Download FRED cases from HuggingFace.")
    ap.add_argument("seq", nargs="+", help="case ids, e.g. 0 3 68")
    ap.add_argument("--cases-dir", default=DEFAULT_CASES_DIR)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--refresh-index", action="store_true")
    a = ap.parse_args()
    if a.refresh_index:
        case_index(refresh=True)
    for s in a.seq:
        download_case(s, a.cases_dir, keep_zip=a.keep_zip)
