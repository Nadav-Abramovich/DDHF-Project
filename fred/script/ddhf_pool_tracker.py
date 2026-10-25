"""
ddhf_pool_tracker.py -- single-drone closed-loop tracker for the PER-PIXEL (pool)
DDHF detector. Shared by the tracking optimizer and debug_viewer_pooled --track.

Unlike the components tracker (tracking_eval), the location prior here relaxes the
PER-PIXEL gate spatially: each candidate pixel's thresholds are interpolated
strict<->floor by g(d)=exp(-d^2/2 sigma^2), where d is the pixel's distance to the
tracker's last box centre (g=0 / strict when untracked). The relaxed green mask is
then connected-component-boxed; the emitted box is the strongest when acquiring,
or the nearest to the last centre when following. The track drops after
`lose_after` consecutive empty windows. GT is never used here.
"""
import numpy as np
from scipy.ndimage import label as _label, find_objects as _find_objects, binary_dilation as _bdil
from utils import BoundingBox, _merge_boxes

_S3 = np.ones((3, 3), int)
_GRID_CAP = 250_000        # label() never runs on a grid bigger than this
_GREEN_CAP = 30_000        # above this many green px, skip CC (degenerate, too-loose gate)


def boxes_from_green(xs, ys, mrp, mba, alpha, flag_n=1):
    """Connected-component boxes from green pixel coords (matches _extract_boxes:
    min_rotor_pixels, min_box_area, alpha-merge). With flag_n>1 each detected pixel
    flags its flag_n x flag_n neighbourhood (binary dilation) before labelling, so
    sparse detections grow/merge into boxes."""
    if xs.size == 0:
        return []
    pad = max(0, int(flag_n) // 2)
    xmin = int(xs.min()); ymin = int(ys.min())
    x0 = xmin - pad; y0 = ymin - pad
    gw = int(xs.max()) - xmin + 1 + 2 * pad; gh = int(ys.max()) - ymin + 1 + 2 * pad
    if gw * gh > _GRID_CAP:                                   # degenerate: one big box
        b = BoundingBox(x0, y0, int(xs.max()) + pad, int(ys.max()) + pad)
        return [b] if (b.area >= mba and xs.size >= mrp) else []
    grid = np.zeros((gh, gw), dtype=bool); grid[ys - ymin + pad, xs - xmin + pad] = True
    if flag_n > 1:
        grid = _bdil(grid, structure=np.ones((int(flag_n), int(flag_n)), bool))
    lbl, nc = _label(grid, structure=_S3)
    if nc == 0:
        return []
    comp_px = np.bincount(lbl.ravel(), minlength=nc + 1)
    boxes = []
    for ci, sl in enumerate(_find_objects(lbl), start=1):
        if sl is None or comp_px[ci] < mrp:
            continue
        y1 = sl[0].start + y0; y2 = sl[0].stop - 1 + y0
        x1 = sl[1].start + x0; x2 = sl[1].stop - 1 + x0
        b = BoundingBox(int(x1), int(y1), int(x2), int(y2))
        if b.area >= mba:
            boxes.append(b)
    return _merge_boxes(boxes, alpha)


class PoolTracker:
    """strict/floor are dicts with keys min_events, tau_sf, tau_comb."""
    def __init__(self, strict, floor, sigma, lose_after, mrp, mba, alpha, flag_n=1):
        self.s = strict; self.f = floor; self.sigma = float(sigma); self.lose_after = int(lose_after)
        self.mrp = mrp; self.mba = mba; self.alpha = alpha; self.flag_n = int(flag_n)
        self.last = None; self.misses = 0

    def _miss(self):
        self.misses += 1
        if self.last is not None and self.misses > self.lose_after:
            self.last = None
        return None, []

    def step(self, xs, ys, gc, sf, snr):
        """One window of candidate pixels -> (chosen BoundingBox or None, all boxes)."""
        xs = np.asarray(xs); ys = np.asarray(ys)
        if xs.size == 0:
            return self._miss()
        if self.last is None:
            g = np.zeros(xs.size)
        else:
            d2 = (xs - self.last[0]) ** 2 + (ys - self.last[1]) ** 2
            g = np.exp(-d2 / (2.0 * self.sigma ** 2))
        me   = self.s["min_events"] - (self.s["min_events"] - self.f["min_events"]) * g
        tsf  = self.s["tau_sf"]     + (self.f["tau_sf"]     - self.s["tau_sf"])     * g
        tcomb = self.s["tau_comb"]  - (self.s["tau_comb"]   - self.f["tau_comb"])   * g
        green = (gc >= me) & (sf <= tsf) & (snr >= tcomb)
        gx = xs[green]; gy = ys[green]
        if gx.size > _GREEN_CAP:                              # too-loose gate -> one big box (cheap)
            b = BoundingBox(int(gx.min()), int(gy.min()), int(gx.max()), int(gy.max()))
            boxes = [b] if b.area >= self.mba else []
        else:
            boxes = boxes_from_green(gx, gy, self.mrp, self.mba, self.alpha, self.flag_n)
        if not boxes:
            return self._miss()
        cents = [((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0) for b in boxes]
        if self.last is None:
            i = int(np.argmax([b.area for b in boxes]))                 # acquire: largest
        else:
            i = int(np.argmin([(c[0] - self.last[0]) ** 2 + (c[1] - self.last[1]) ** 2
                               for c in cents]))                        # follow: nearest
        self.last = cents[i]; self.misses = 0
        return boxes[i], boxes
