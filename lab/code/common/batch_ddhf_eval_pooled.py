"""
batch_ddhf_eval_pooled.py — logging and cleanup helpers.

This used to be the batch evaluator for ddhf_eval_pooled.py (and later held its
Google-Drive download helpers). Both are gone; only these two helpers are still
used, by frame_features.py and case_statistics.py:

    _log          print + append a timestamped line to a run log
    _safe_remove  delete a file/tree with retries (Windows file locks)
"""

import os
import time
import shutil
import datetime


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(log_path, msg):
    """Print and append a timestamped line to the run log (best-effort)."""
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _safe_remove(path, retries=5, delay=1.0):
    """Delete a file or directory tree, never raising. Retries a few times because
    on Windows a just-released handle (e.g. metavision's event-file lock) can
    linger briefly after the child process exits. Returns True if gone."""
    for attempt in range(retries):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)        # not ignore_errors: we want to know
            elif os.path.exists(path):
                os.remove(path)
            if not os.path.exists(path):
                return True
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  WARNING: could not delete {path} after {retries} tries: {exc}",
                      flush=True)
                return False
            time.sleep(delay)
    return not os.path.exists(path)
