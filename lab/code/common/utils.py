"""
DDHF: Drone Detection via Harmonic Fingerprinting
Based on: https://arxiv.org/html/2603.08386v1
This implementation is a direct translation of the paper's algorithm, optimised for clarity and correctness rather than raw speed. The core spectral computations
are vectorised over the pixel axis to efficiently process the whole window at once, but the overall structure closely follows the original description.


"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# ── Dataset helpers c─────────────────────────────────────────────────────────────

# Parse the wall-clock timestamp embedded in an RGB frame filename, e.g.
# "Video_0_16_03_03.363444.jpg" -> 16:03:03.363444 -> seconds-of-day (float).
_RGB_TS_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.jpg$", re.IGNORECASE)


def rgb_frame_seconds(filename):
    """Seconds-of-day for an RGB frame, parsed from its filename timestamp."""
    m = _RGB_TS_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse timestamp from RGB filename: {filename}")
    hh, mm, ss, frac = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + float("0." + frac)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DDHFConfig:
    """Hyperparameters — defaults are the Bayesian-optimised values from the paper."""
    window_ms:           float = 33.3    # time window duration [ms]
    step_ms:             float = 33.3    # sliding step [ms]; equal to window_ms → non-overlapping
    max_freq_hz:         float = 6300.0  # spectrum spans 0..max_freq_hz; bin COUNT is derived
                                         #   from this + window (see frequency_bins). Must be
                                         #   >= num_harmonics * (highest rotor_bands_hz fundamental)
                                         #   so the whole comb stays in range. 6300 ≈ 4·1500 Hz +
                                         #   margin, covering every drone in rotor_bands_hz. (The
                                         #   paper's 1920 Hz / K=64 only reaches ~450 Hz combs.)
    tau_sf:              float = 0.89    # spectral flatness upper threshold (discard above)
    tau_comb:            float = 1.5     # harmonic comb SNR lower threshold (accept above)
    # Blade-pass fundamental bands [Hz] for the drones in the dataset. The comb
    # search only tests fundamentals ω whose frequency ω·(1000/window_ms) lands
    # in one of these bands, instead of sweeping every candidate. Empty () falls
    # back to the original "sweep all fundamentals" behaviour.
    rotor_bands_hz: Tuple[Tuple[float, float], ...] = (
        (280.0,  340.0),    # DJI Mini 2 (~310–340) & DJI Mini 3 (~280–340)
        (450.0,  700.0),    # DJI Tello EDU
        (800.0, 1500.0),    # DarwinFPV CineApe20 (~800–1200) & BetaFPV Air75 (~900–1500)
    )
    num_harmonics:       int   = 4       # M — harmonics evaluated per comb candidate
    min_harmonics:       int   = 4       # h_min — minimum harmonics that must be valid
    min_present_harmonics: int = 0       # if >0: a comb ω is only accepted when at least this many
                                         #   of its harmonics are individually PRESENT (peak >=
                                         #   harmonic_present_factor × noise-median). Rejects
                                         #   "one bump + noise windows" false combs. 0 = disabled.
    harmonic_present_factor: float = 2.0 # per-harmonic "present" threshold, in × the noise median
    noise_band_harmonics: float = 0.0    # if >0: estimate the comb noise floor LOCALLY, over bins
                                         #   only up to this multiple of the top harmonic (C·Mω),
                                         #   instead of the whole [0,K) spectrum. Removes the comb
                                         #   SNR's dependence on the arbitrary max_freq_hz/K, and
                                         #   gives a floor representative near a low-freq comb.
                                         #   0 = paper behaviour (full complement set).
    harmonic_half_width: int   = 3       # Δk — frequency half-bandwidth per harmonic [bins]
    min_events:          int   = 6       # n_min — minimum events per pixel per window
    min_own_events:      int   = 0       # n_own — extra gate (POOLING only): the centre pixel's
                                         #   OWN events must be >= this, on top of its 3x3 pool
                                         #   meeting min_events. 0 = disabled. No effect under
                                         #   pixelize/baseline, where own == the group count.
    min_box_area:        int   = 147     # A_min — minimum bounding-box area [px²]
    min_rotor_pixels:    int   = 0       # minimum rotor pixels per connected component
                                         #   (0 = disabled). A denser drone/noise separator than
                                         #   min_box_area: a real rotor fills its box, whereas a
                                         #   sparse noise cluster can have a wide box but few px.
                                         #   Applied by the 3x3-pooled _extract_boxes only.
    pixel_block:         int   = 3       # pixelization block size B [px]: bin the sensor into
                                         #   non-overlapping BxB super-pixels and run ONE NDFT per
                                         #   block. Used only by the pixelize detector (one calc
                                         #   per block, vs pooling's per-pixel overlapping calcs).
    alpha:               float = 2   # α — box-merge join-radius factor (Eq. 12):
                                         #   merge boxes when dist(c1,c2) <
                                         #   α·max(w1,h1,w2,h2). Paper Table 2 value.

    @property
    def frequency_bins(self) -> int:
        """NDFT bin count K. Derived so the spectrum always reaches max_freq_hz
        regardless of window_ms (each bin = 1000/window_ms Hz). This is what
        prevents a longer window from silently dropping the rotor harmonics.
        Default (6300 Hz @ 33.3 ms) gives K≈210, enough for a 4-harmonic comb on
        the highest band fundamental (~1500 Hz). The paper used 1920 Hz / K=64."""
        k = int(round(self.max_freq_hz * self.window_ms / 1000.0))
        return max(self.num_harmonics + 1, k)


_DEFAULT = DDHFConfig()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0

    @property
    def w(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def h(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


# ── Core spectral functions (vectorised over the pixel axis) ───────────────────
#
# Every function in this section takes a *batch* of pixels at once: the last
# array axis (G) indexes the candidate pixels, so one NumPy call processes the
# whole window. The maths for a single pixel is written out in each docstring.

def _ndft_power_batch(
    power_buf: None,
    pol_sorted: np.ndarray,
    tn_sorted: np.ndarray,
    group_starts: np.ndarray,
    K: int,
    max_chunk_elems: int = 4_000_000,
) -> np.ndarray:
    """
    Non-Uniform DFT power spectrum for every pixel group, computed at once.

    For a single pixel the paper defines
        F_k = Σ_j p_j · exp(i·k·t_j),   k = 0 … K-1
        P_k = |F_k|²
    where the sum runs over that pixel's events (timestamps already normalised
    to [-π, π) — see ``_normalise_timestamps``).

    Vectorised form
    ---------------
    All events of the window are concatenated, sorted by pixel id, so each
    pixel occupies a contiguous slice ``[group_starts[g] : group_starts[g+1]]``.
    We build the per-event phasors  E[k, j] = p_j · exp(i·k·t_j)  and sum the
    columns belonging to each pixel with an ``np.add.reduceat`` segment reduction.

    The phasor matrix E is dense (K, N) and is the program's peak allocation;
    since K·N grows like window_ms², a long window can exhaust RAM. To bound it
    we process the pixel groups in chunks of at most ``max_chunk_elems`` matrix
    entries, never materialising more than one chunk at a time. Each pixel group
    stays wholly inside a chunk, so the reduceat sums — and the result — are
    bit-identical to computing the whole window at once.

    Returns
    -------
    power : (K, G) float64 — power spectrum per pixel group.
    """
    G = group_starts.size
    N = tn_sorted.size
    k = np.arange(K, dtype=np.float64)                       # (K,)
    power = np.empty((K, G), dtype=np.float64)               # (K, G)

    # Max events per chunk so that K·n_events ≤ max_chunk_elems (≥1 group always).
    target_events = max(1, max_chunk_elems // K)
    g0 = 0
    while g0 < G:
        e0 = int(group_starts[g0])
        g0_end = int(group_starts[g0 + 1]) if g0 + 1 < G else N

        # A SINGLE group larger than the budget (e.g. a dense pooled component with
        # millions of events) cannot be packed with reduceat without materialising
        # the full (K, n_events) phasor matrix — which is what OOMs. Since F_k is a
        # plain sum over the group's events, accumulate it over event sub-batches so
        # the peak allocation stays ≈ (K, target_events). Result is identical.
        if (g0_end - e0) > target_events:
            Fk = np.zeros(K, dtype=np.complex128)
            c = e0
            while c < g0_end:
                c1 = min(g0_end, c + target_events)
                phase = np.outer(k, tn_sorted[c:c1])        # (K, ≤target_events)
                Fk += (pol_sorted[c:c1][None, :] * np.exp(1j * phase)).sum(axis=1)
                c = c1
            power[:, g0] = np.abs(Fk) ** 2
            g0 += 1
            continue

        # Grow the chunk one group at a time until the next group would push the
        # event count over the budget; always take at least the first group.
        g1 = g0 + 1
        while g1 < G and (int(group_starts[g1]) - e0) <= target_events:
            g1 += 1
        e1 = int(group_starts[g1]) if g1 < G else N

        phase = np.outer(k, tn_sorted[e0:e1])               # (K, n_e)
        E = pol_sorted[e0:e1][None, :] * np.exp(1j * phase)  # (K, n_e)
        local_starts = group_starts[g0:g1] - e0             # group starts within chunk
        F = np.add.reduceat(E, local_starts, axis=1)        # (K, g1-g0)
        power[:, g0:g1] = np.abs(F) ** 2
        g0 = g1

    return power


def _spectral_flatness_batch(power_no_dc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    SF = geometric_mean(P) / arithmetic_mean(P)  ∈ (0, 1],  per pixel.

    Values near 1 indicate white noise; values near 0 indicate concentrated
    peaks. As in the paper (Eq. 7, sum over k = 1 … K-1) the DC bin is excluded,
    so ``power_no_dc`` must be ``power[1:]`` with shape (K-1, G).
    """
    p = power_no_dc + eps
    geo = np.exp(np.mean(np.log(p), axis=0))                 # (G,)
    arith = np.mean(p, axis=0)                               # (G,)
    return geo / arith                                      # (G,)


def _comb_snr_batch(power: np.ndarray, cfg: DDHFConfig,
                    return_omega: bool = False):
    """
    Best median-normalised harmonic-comb SNR over all candidate fundamentals ω,
    evaluated for every pixel group at once.

    For each ω in 1 … K // M (paper Eq. 8–11):
      N(ω)      = mean over harmonics m·ω of the peak power inside a ±Δk window
      noise(ω)  = median power of the bins *outside* every harmonic window
      SNR_M(ω)  = N(ω) / noise(ω)
    The score returned per pixel is  max_ω SNR_M(ω).

    When ``cfg.rotor_bands_hz`` is non-empty the sweep is restricted to the
    fundamentals of the dataset's drones: a candidate ω is only tested if its
    frequency ω·bin_hz (bin_hz = 1000/window_ms) lands inside one of the bands.

    ``power`` has shape (K, G); the result has shape (G,). With
    ``return_omega=True`` also returns the winning fundamental bin per pixel
    (G,) int64, 0 where no candidate was evaluated.
    """
    K, G = power.shape
    best = np.zeros(G, dtype=np.float64)
    best_om = np.zeros(G, dtype=np.int64) if return_omega else None

    bands  = cfg.rotor_bands_hz
    bin_hz = 1000.0 / cfg.window_ms

    for omega in range(1, K // cfg.num_harmonics + 1):
        # Only look for the combs of the dataset's drones: skip any fundamental
        # whose frequency falls outside every blade-pass band.
        if bands and not any(lo <= omega * bin_hz <= hi for lo, hi in bands):
            continue

        harmonics = [m * omega for m in range(1, cfg.num_harmonics + 1) if m * omega < K]
        if len(harmonics) < cfg.min_harmonics:
            continue

        harmonic_mask = np.zeros(K, dtype=bool)
        peak_powers: List[np.ndarray] = []                  # each (G,)

        for h in harmonics:
            lo = max(0, h - cfg.harmonic_half_width)
            hi = min(K - 1, h + cfg.harmonic_half_width)
            peak_powers.append(power[lo : hi + 1, :].max(axis=0))
            harmonic_mask[lo : hi + 1] = True

        # Noise floor = bins outside every harmonic window. Exclude the DC bin
        # (k=0): it carries the squared net polarity, which is large and not
        # part of the rotor comb, so leaving it in would inflate the floor.
        noise_mask = ~harmonic_mask
        noise_mask[0] = False
        # Optional LOCAL noise floor: restrict the complement to bins up to
        # noise_band_harmonics × the top harmonic, so the floor is measured near
        # the comb and doesn't depend on the arbitrary spectrum length K.
        nbh = getattr(cfg, "noise_band_harmonics", 0.0)
        if nbh > 0:
            band_hi = min(K, int(nbh * harmonics[-1]) + cfg.harmonic_half_width + 1)
            noise_mask[band_hi:] = False
        noise = power[noise_mask, :]                         # (n_noise, G)
        if noise.shape[0] == 0:
            continue

        median_noise = np.median(noise, axis=0)             # (G,)
        # Optional median floor (off by default): a near-empty spectrum has a
        # ~0 median, which makes SNR explode for any scrap of low-frequency power
        # (the 30 Hz collapse). Floor the median at a fraction of the per-pixel
        # mean power so SNR stays bounded. Enable via cfg.snr_median_floor_frac>0.
        mff = getattr(cfg, "snr_median_floor_frac", 0.0)
        if mff > 0:
            median_noise = np.maximum(median_noise, mff * power[1:, :].mean(axis=0))
        mean_peaks = np.mean(np.stack(peak_powers, axis=0), axis=0)  # (G,)

        # Per-pixel guard: where the noise floor underflows, the paper skips ω.
        snr = np.zeros(G, dtype=np.float64)
        ok = median_noise >= 1e-10
        snr[ok] = mean_peaks[ok] / median_noise[ok]

        # Optional "harmonics present" gate: a comb is only valid if enough of its
        # harmonics individually rise above the noise floor — rejects a single
        # strong bump whose other harmonic windows are just noise.
        mph = getattr(cfg, "min_present_harmonics", 0)
        if mph > 0:
            factor = getattr(cfg, "harmonic_present_factor", 2.0)
            present = np.zeros(G, dtype=np.int32)
            for pk in peak_powers:
                present += (pk >= factor * median_noise)
            snr = np.where(present >= mph, snr, 0.0)

        better = snr > best
        best[better] = snr[better]
        if return_omega:
            best_om[better] = omega

    if return_omega:
        return best, best_om
    return best


# ── Per-window rotor detection ────────────────────────────────────────────────

def _detect_rotor_pixels(
    x: np.ndarray, y: np.ndarray,
    p: np.ndarray, t_norm: np.ndarray,
    height: int, width: int, cfg: DDHFConfig,
    return_freq: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    For every pixel with enough events: NDFT → spectral-flatness filter →
    harmonic-comb SNR.  Returns (rotor_mask [H,W bool], score_map [H,W float]).

    With ``return_freq=True`` a third map is returned: the blade-pass
    fundamental in Hz of the comb that won each accepted pixel, i.e.
    ω·(1000/window_ms) for the ω that maximised the comb SNR, and 0 where no
    rotor was accepted. It is quantised to the bin width 1000/window_ms Hz
    (30 Hz at the default 33.3 ms window) - the same number the SNR was scored
    on, not a refined estimate.

    Pixels are grouped by a sorted pixel id so that all groups can be processed
    with the vectorised spectral helpers above — the per-pixel Python loop of
    the original version is gone, but the decision for each pixel is identical.
    Each window is analysed on its own (paper §3.5), with no cross-window
    averaging.
    """
    rotor  = np.zeros((height, width), dtype=bool)
    scores = np.zeros((height, width), dtype=np.float32)
    freqs  = np.zeros((height, width), dtype=np.float32)
    if x.size == 0:
        return (rotor, scores, freqs) if return_freq else (rotor, scores)

    # Group events by pixel via a sorted pixel-id (contiguous slice per pixel).
    pixel_id   = y * width + x
    order      = np.argsort(pixel_id, kind="stable")
    pid_sorted = pixel_id[order]
    pol_sorted = p[order]
    tn_sorted  = t_norm[order]

    group_starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_sorted)) + 1))
    group_sizes  = np.diff(np.append(group_starts, pid_sorted.size))

    # Discard pixels with fewer than n_min events *before* the NDFT (paper §3.5).
    # Background/noise pixels (1–2 events each) are the vast majority, so skipping
    # their transforms is the dominant speedup — and the result is identical,
    # since those pixels fail the n_min test anyway.
    big = group_sizes >= cfg.min_events                                     # n_min
    if not big.any():
        return (rotor, scores, freqs) if return_freq else (rotor, scores)
    keep = np.repeat(big, group_sizes)            # per-event mask (sorted order)
    pid_k = pid_sorted[keep]
    pol_k = pol_sorted[keep]
    tn_k  = tn_sorted[keep]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_k)) + 1))

    # Vectorised NDFT power spectra for the surviving pixel groups: (K, G)
    power = _ndft_power_batch(None, pol_k, tn_k, starts, cfg.frequency_bins)

    # Pixel coordinates of each surviving group.
    grp_pid = pid_k[starts]
    py = grp_pid // width
    px = grp_pid % width

    # Apply the remaining acceptance criteria pixel-wise (all evaluated as arrays).
    valid  = _spectral_flatness_batch(power[1:]) <= cfg.tau_sf              # skip DC bin
    if return_freq:
        snr, omega = _comb_snr_batch(power, cfg, return_omega=True)
    else:
        snr = _comb_snr_batch(power, cfg)
    valid &= snr >= cfg.tau_comb

    sel = np.flatnonzero(valid)
    rotor[py[sel], px[sel]]  = True
    scores[py[sel], px[sel]] = snr[sel].astype(np.float32)
    if not return_freq:
        return rotor, scores
    # bin → Hz: the NDFT is anchored to the fixed window duration (see
    # _normalise_timestamps), so bin k is k·(1000/window_ms) Hz in every window.
    freqs[py[sel], px[sel]] = (omega[sel] * (1000.0 / cfg.window_ms)).astype(np.float32)
    return rotor, scores, freqs


def _extract_boxes(
    rotor: np.ndarray, scores: np.ndarray, cfg: DDHFConfig
) -> List[BoundingBox]:
    """8-connected component labeling → area filter → box merging."""
    if not rotor.any():
        return []

    labeled, n = ndimage.label(rotor, structure=np.ones((3, 3)))
    boxes: List[BoundingBox] = []

    for lid in range(1, n + 1):
        mask = labeled == lid
        ys, xs = np.where(mask)
        b = BoundingBox(
            x1=int(xs.min()), y1=int(ys.min()),
            x2=int(xs.max()), y2=int(ys.max()),
            score=float(scores[mask].mean()),
        )
        if b.area >= cfg.min_box_area:
            boxes.append(b)

    return _merge_boxes(boxes, cfg.alpha)


def _merge_boxes(boxes: List[BoundingBox], alpha: float) -> List[BoundingBox]:
    """Merge boxes per the paper (Eq. 12): two boxes with sizes (w_i, h_i) and
    centroids c_i are merged when

        dist(c1, c2) < alpha * max(w1, h1, w2, h2),

    with alpha the validation-tuned join-radius factor. The rule is applied to
    the ORIGINAL boxes and grouped transitively via connected components, so each
    component becomes one detection box. (Using the original box sizes -- not a
    growing merged box -- keeps the join radius bounded; a runaway scale would
    otherwise chain distant boxes across the frame.)"""
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cents = [b.centroid for b in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(cents[i][0] - cents[j][0], cents[i][1] - cents[j][1])
            scale = alpha * max(boxes[i].w, boxes[i].h, boxes[j].w, boxes[j].h)
            if dist < scale:
                parent[find(j)] = find(i)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: List[BoundingBox] = []
    for idxs in groups.values():
        merged.append(BoundingBox(
            min(boxes[i].x1 for i in idxs), min(boxes[i].y1 for i in idxs),
            max(boxes[i].x2 for i in idxs), max(boxes[i].y2 for i in idxs),
            max(boxes[i].score for i in idxs),
        ))
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_polarity(p: np.ndarray) -> np.ndarray:
    """Map {0, 1} or {-1, +1} encodings to float64 {-1.0, +1.0}."""
    p = np.asarray(p)
    if p.min() >= 0:
        return np.where(p > 0, 1.0, -1.0).astype(np.float64)
    return p.astype(np.float64)


def _normalise_timestamps(t: np.ndarray, window_span: Optional[float] = None) -> np.ndarray:
    """
    Normalise timestamps to [-π, π) as specified in the paper.

    Normalisation is *global per window*: every pixel shares the same time
    reference, so frequency bin k means the same physical frequency for all
    pixels (k cycles across the window).

    The frequency axis is anchored to the *fixed window duration* rather than to
    the span of whichever events happen to be present: pass ``window_span`` (the
    window length, in the same units as ``t`` — microseconds) so that bin k maps
    to a constant physical frequency k / window_span across every window. The
    absolute anchor (t.min) only sets a global phase, which |F_k|² discards, so
    it does not affect the power spectrum. If ``window_span`` is omitted we fall
    back to the event span (legacy behaviour).
    """
    t = np.asarray(t, dtype=np.float64)
    t_min = t.min()
    span = float(window_span) if window_span else (t.max() - t_min)
    if span <= 0.0:
        return np.zeros_like(t)
    return (t - t_min) / span * 2.0 * np.pi - np.pi


# ── DDHF detector ─────────────────────────────────────────────────────────────

class DDHF:
    """Drone Detection via Harmonic Fingerprinting."""

    def __init__(self, height: int, width: int, cfg: Optional[DDHFConfig] = None) -> None:
        self.height = height
        self.width  = width
        self.cfg    = cfg or DDHFConfig()

    # ------------------------------------------------------------------
    def detect_window(
        self,
        x: np.ndarray, y: np.ndarray,
        p: np.ndarray, t: np.ndarray,
    ) -> List[BoundingBox]:
        """
        Detect drones in a single pre-sliced event window.

        Parameters
        ----------
        x, y : pixel coordinates (integer arrays)
        p    : polarity — {0,1} or {-1,+1}, normalised internally
        t    : timestamps in microseconds

        Returns
        -------
        List of BoundingBox detections for this window.
        """
        if len(t) < self.cfg.min_events:
            return []

        pol    = _normalise_polarity(p)
        t_norm = _normalise_timestamps(t, window_span=self.cfg.window_ms * 1000.0)
        rotor, scores = _detect_rotor_pixels(
            x.astype(np.int64), y.astype(np.int64),
            pol, t_norm,
            self.height, self.width, self.cfg,
        )
        return _extract_boxes(rotor, scores, self.cfg)

    # ------------------------------------------------------------------
    def detect_stream(
        self, events: np.ndarray
    ) -> Iterator[Tuple[int, int, List[BoundingBox]]]:
        """
        Slide a time window over a full event stream.

        Uses binary search (searchsorted) for O(log N) window extraction,
        avoiding the O(N) boolean mask used in naive implementations.

        Parameters
        ----------
        events : structured numpy array with fields x, y, p, t

        Yields
        ------
        (window_start_us, window_end_us, boxes)
        """
        if len(events) == 0:
            return

        # sort once by timestamp
        order  = np.argsort(events["t"], kind="stable")
        events = events[order]
        t_all  = events["t"].astype(np.int64)

        window_us = int(self.cfg.window_ms * 1_000)
        step_us   = int(self.cfg.step_ms   * 1_000)
        t0        = int(t_all[0])
        t_end     = int(t_all[-1])

        for t_win in range(t0, t_end, step_us):
            lo = int(np.searchsorted(t_all, t_win,            side="left"))
            hi = int(np.searchsorted(t_all, t_win + window_us, side="left"))

            sl    = events[lo:hi]
            boxes = self.detect_window(
                sl["x"].astype(np.int64),
                sl["y"].astype(np.int64),
                sl["p"],
                sl["t"].astype(np.float64),
            )
            yield t_win, t_win + window_us, boxes


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_events(path: Path) -> np.ndarray:
    """Load a raw event stream via the Metavision SDK."""
    import os

    # Make the Prophesee/Metavision native libraries discoverable before import.
    plugin_folder = r"C:\Program Files\Prophesee\lib\metavision\hal\plugins"
    dll_directories = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_deps", "openeb_bin"),
        r"C:\Program Files\Prophesee\bin",
        plugin_folder,
        r"C:\Program Files\Prophesee\lib\hdf5\plugin"
    ]

    for directory in dll_directories:
        if os.path.exists(directory):
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(directory)
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")

    os.environ["MV_HAL_PLUGIN_PATH"] = plugin_folder
    os.environ["HDF5_PLUGIN_PATH"] = r"C:\Program Files\Prophesee\lib\hdf5\plugin"

    from metavision_core.event_io import EventsIterator

    batches = [b for b in EventsIterator(str(path)) if len(b) > 0]
    if not batches:
        raise RuntimeError(f"No events loaded from {path}")
    return np.concatenate(batches)


def accumulate_events(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Build a 2-D event-count image for visualisation."""
    img = np.zeros((height, width), dtype=np.int32)
    x   = events["x"].astype(np.int64)
    y   = events["y"].astype(np.int64)
    ok  = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.add.at(img, (y[ok], x[ok]), 1)
    return img


def draw_detections(
    event_image: np.ndarray,
    boxes: List[BoundingBox],
    title: str = "DDHF Detections",
    output_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(np.log1p(event_image), cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    for i, b in enumerate(boxes, 1):
        ax.add_patch(plt.Rectangle(
            (b.x1, b.y1), b.w, b.h,
            edgecolor="lime", facecolor="none", linewidth=2,
        ))
        ax.text(b.x1 + 2, b.y2 - 4, f"#{i} snr={b.score:.1f}",
                color="yellow", fontsize=9, weight="bold")
    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _find_raw_file() -> Path:
    """Auto-detect the first .raw event file in the project directory tree."""
    matches = sorted(Path(__file__).parent.rglob("*.raw"))
    if not matches:
        raise FileNotFoundError("No .raw event file found under the project directory.")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDHF: Drone Detection via Harmonic Fingerprinting")
    p.add_argument("--input",               type=Path,  default=None)  # None -> _find_raw_file()
    p.add_argument("--width",               type=int,   default=1280)
    p.add_argument("--height",              type=int,   default=720)
    p.add_argument("--window-ms",           type=float, default=_DEFAULT.window_ms)
    p.add_argument("--step-ms",             type=float, default=_DEFAULT.step_ms)
    p.add_argument("--max-freq-hz",         type=float, default=_DEFAULT.max_freq_hz)
    p.add_argument("--tau-sf",              type=float, default=_DEFAULT.tau_sf)
    p.add_argument("--tau-comb",            type=float, default=_DEFAULT.tau_comb)
    p.add_argument("--num-harmonics",       type=int,   default=_DEFAULT.num_harmonics)
    p.add_argument("--min-harmonics",       type=int,   default=_DEFAULT.min_harmonics)
    p.add_argument("--harmonic-half-width", type=int,   default=_DEFAULT.harmonic_half_width)
    p.add_argument("--min-events",          type=int,   default=_DEFAULT.min_events)
    p.add_argument("--min-box-area",        type=int,   default=_DEFAULT.min_box_area)
    p.add_argument("--alpha",               type=float, default=_DEFAULT.alpha)
    p.add_argument("--output",              type=Path,  default=Path("detections.png"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = DDHFConfig(
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        max_freq_hz=args.max_freq_hz,
        tau_sf=args.tau_sf,
        tau_comb=args.tau_comb,
        num_harmonics=args.num_harmonics,
        min_harmonics=args.min_harmonics,
        harmonic_half_width=args.harmonic_half_width,
        min_events=args.min_events,
        min_box_area=args.min_box_area,
        alpha=args.alpha,
    )

    if args.input is None:
        args.input = _find_raw_file()
    events   = load_events(args.input)
    detector = DDHF(height=args.height, width=args.width, cfg=cfg)

    print(f"Loaded {len(events):,} events  "
          f"[{events['t'].min()/1e6:.3f}s – {events['t'].max()/1e6:.3f}s]")

    all_boxes: List[BoundingBox] = []
    for t_start, t_end, boxes in detector.detect_stream(events):
        if boxes:
            t_s = t_start / 1e6
            desc = ", ".join(
                f"({b.x1},{b.y1})→({b.x2},{b.y2}) snr={b.score:.2f}" for b in boxes
            )
            print(f"  t={t_s:.3f}s  {len(boxes)} detection(s): {desc}")
            all_boxes.extend(boxes)

    print(f"\nTotal detections across all windows: {len(all_boxes)}")
    event_image = accumulate_events(events, args.height, args.width)
    draw_detections(event_image, all_boxes, output_path=args.output)


if __name__ == "__main__":
    main()
