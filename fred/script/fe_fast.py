"""
fe_fast.py — numba-JIT hot kernels for frame_explorer's per-frame loops.

Profiling a tracked evaluation showed ~50% of compute in two overhead-bound
calls made ~3,400x per case (once per frame): scipy.ndimage.label on a small
cropped grid, and _merge_rects_fast's sparse-graph connected components. Both
are reimplemented here as compiled union-find over the raw pixel/box lists,
with IDENTICAL semantics:

  * cc_label_px  == ndimage.label(grid, 3x3)[sy,sx]-1 : 8-connected components
    of the pass pixels, numbered in raster-scan order of first occurrence
    (ndimage's numbering), returned per input pixel, 0-based.
  * merge_rects  == _merge_rects_fast without the scipy csgraph detour: boxes
    whose centroid distance < alpha x max(w,h) of either box merge
    transitively; each group becomes its envelope, groups ordered by first box.

frame_explorer imports this opportunistically (set FE_FAST=0 to disable) and
falls back to the scipy path when numba is unavailable. Compiled kernels are
disk-cached (cache=True), so worker processes pay the compile once per machine.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:                         # path compression
        parent[i], i = root, parent[i]
    return root


@njit(cache=True)
def _union(parent, a, b):
    ra = _find(parent, a); rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


@njit(cache=True)
def cc_label_px(sx, sy):
    """8-connected component id (0-based, raster-scan numbering) per pixel."""
    n = sx.size
    x0 = sx.min(); y0 = sy.min()
    w = int(sx.max() - x0 + 1); h = int(sy.max() - y0 + 1)
    grid = np.full((h, w), -1, np.int64)
    for i in range(n):
        grid[sy[i] - y0, sx[i] - x0] = i
    parent = np.arange(n)
    for yy in range(h):
        for xx in range(w):
            i = grid[yy, xx]
            if i < 0:
                continue
            if xx > 0 and grid[yy, xx - 1] >= 0:
                _union(parent, grid[yy, xx - 1], i)
            if yy > 0:
                if grid[yy - 1, xx] >= 0:
                    _union(parent, grid[yy - 1, xx], i)
                if xx > 0 and grid[yy - 1, xx - 1] >= 0:
                    _union(parent, grid[yy - 1, xx - 1], i)
                if xx < w - 1 and grid[yy - 1, xx + 1] >= 0:
                    _union(parent, grid[yy - 1, xx + 1], i)
    lab = np.full(n, -1, np.int64)
    root_lab = np.full(n, -1, np.int64)
    nxt = 0
    for yy in range(h):                              # ndimage numbering order
        for xx in range(w):
            i = grid[yy, xx]
            if i < 0:
                continue
            r = _find(parent, i)
            if root_lab[r] < 0:
                root_lab[r] = nxt
                nxt += 1
            lab[i] = root_lab[r]
    return lab


@njit(cache=True)
def _merge_rects_nb(x1, y1, x2, y2, alpha):
    k = x1.size
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    s = np.empty(k, np.float64)
    for i in range(k):
        wi = x2[i] - x1[i] + 1.0; hi = y2[i] - y1[i] + 1.0
        s[i] = wi if wi > hi else hi
    parent = np.arange(k)
    for i in range(k):
        for j in range(i + 1, k):
            m = s[i] if s[i] > s[j] else s[j]
            thr = alpha * m
            dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
            if dx * dx + dy * dy < thr * thr:
                _union(parent, i, j)
    lab = np.full(k, -1, np.int64)
    root_lab = np.full(k, -1, np.int64)
    nxt = 0
    for i in range(k):                               # groups by first box
        r = _find(parent, i)
        if root_lab[r] < 0:
            root_lab[r] = nxt
            nxt += 1
        lab[i] = root_lab[r]
    out = np.empty((nxt, 4), np.int64)
    for g in range(nxt):
        out[g, 0] = out[g, 1] = 2 ** 60
        out[g, 2] = out[g, 3] = -2 ** 60
    for i in range(k):
        g = lab[i]
        if x1[i] < out[g, 0]: out[g, 0] = int(x1[i])
        if y1[i] < out[g, 1]: out[g, 1] = int(y1[i])
        if x2[i] > out[g, 2]: out[g, 2] = int(x2[i])
        if y2[i] > out[g, 3]: out[g, 3] = int(y2[i])
    return out


def merge_rects(x1, y1, x2, y2, alpha):
    """List of merged (x1,y1,x2,y2) tuples — same contract as
    _merge_rects_fast's core (its n==1 / n>2000 shortcuts stay in the caller)."""
    out = _merge_rects_nb(x1.astype(np.float64), y1.astype(np.float64),
                          x2.astype(np.float64), y2.astype(np.float64),
                          float(alpha))
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in out]
