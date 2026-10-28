"""
bayes_working_point.py — Bayesian optimisation of ONE global working point
(detector + tracking thresholds) for high precision across ALL cached cases.

A Gaussian-process surrogate (Matern 5/2, numpy/scipy only — no skopt/optuna)
models score(params) from every evaluation done so far; each iteration picks
the next point by Expected Improvement, evaluates it over the selected cases
with frame_explorer's pipeline (same _compute_job path as the old find_working_point's
eval stage), and appends the result to a JSONL journal.

Objective (maximised):
    precision_<agg>  -  penalty * max(0, min_gtf - gtf_hit_rate)
                     [- penalty * max(0, fp_per_frame - max_fppf) if set]
where <agg> is 'mean' (mean of per-case precisions, default), 'min' (worst
case — the strictest reading of "high precision across all cases") or
'pooled' (sum TP / sum TP+FP). The recall floor (--min-gtf) stops the
optimiser from collapsing to a single confident detection.

RESUMABLE: every evaluation is journalled to bayes_wp_<tag>.jsonl the moment
it finishes. Rerunning the same command reloads the journal, skips everything
already evaluated and continues the search from there — Ctrl-C at any time
loses at most the evaluation in flight. Scores are recomputed from the stored
per-case counts on load, so --agg / --min-gtf / --penalty / --max-fppf may be
changed between runs without invalidating old evaluations. Changing the case
set, mode, tracking or matching DOES invalidate them (the journal is
fingerprinted and the run refuses to mix; use --reset or another --state).

Usage:
    python lab_bayes_working_point.py --mode pixelize --track     # all lab runs
    python lab_bayes_working_point.py --mode pixelize --track --workers 8
    python lab_bayes_working_point.py --mode pixelize --track --agg min
    python lab_bayes_working_point.py --mode pixelize --track --iterations 120  # resume
The winner is written as wps/working_point_bayes_<tag>.json (auto_tune
best-point format) plus the stage-4 per-case CSV
wps/working_point_eval_bayes_<tag>.csv, so
    python show_lab_case_graphs.py --eval-csv working_point_eval_bayes_pixelize_track.csv
plots its per-case scores.

lab copy of utility/bayes_working_point.py - same search, objective,
journal and workers - with these lab rules:
  * cases: only the runs under lab/runs/dataset that have a frame_data
    cache for --mode (--seqs / --first-n / --stride pick among them; no
    --split, lab runs have none).
  * working points: journal, winner, per-case CSV and seed JSONs all live in
    data/working_points/; --state / --out / --seed-json are file names there.
  * no --skip-start: a lab run has no pre-GT lead-in, so every frame counts.
  * tau_comb spans --tau-comb-range (default 3-130, log scale), and the comb
    bands are searched too (--drones, default all,tello), and so is
    min_present_harmonics (--present-harmonics, default 0,2,3,4). Both change
    the stored sf / snr, so each combination reads its own feature table:
    --data-dir for its own settings, lab_viewer_pooled's variant folders for
    the others, built on first use. The winner records both, which the
    viewer's --wp replay follows.
"""
import os
import sys
import json
import time
import math
import argparse
import multiprocessing as mp

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular
from scipy.special import ndtr

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # lab/code
_THIS_DIR = os.path.join(_CODE_DIR, "common")           # lab/code/common: frame_explorer, auto_tune
_LAB_DIR  = os.path.dirname(_CODE_DIR)                    # lab/
WPS_DIR   = os.path.join(_LAB_DIR, "data", "working_points")       # everything this copy reads / writes
DATASETS  = os.path.join(_LAB_DIR, "runs", "dataset")   # the only cases it uses
for _p in (_THIS_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import frame_explorer as FE
from auto_tune import DEFAULT_START

PERSIST = [(1, 1), (2, 2), (2, 3), (3, 4), (3, 5)]


# ── search space ─────────────────────────────────────────────────────────────
# Bounds cover auto_tune's LADDERS; ints round after decoding, heavy-tailed
# axes search in log space, persist is an ordinal index into PERSIST.

def build_space(track, tau_comb_range=(3.0, 130.0), drones=("all",),
                present=(0,), factors=(2.0,), snr_avg_max=10.0):
    """tau_comb spans tau_comb_range, on a log scale once it is wide (hi > 4 lo)
    so 3-10 gets as much of the search as 30-130. With more than one drone /
    min_present_harmonics value those become axes too - each combination
    reads its own feature table (spectral_data_dirs)."""
    lo, hi = tau_comb_range
    dims = [
        dict(name="min_events",       kind="int",   lo=2,   hi=16),
        dict(name="tau_sf",           kind="float", lo=0.5, hi=0.95),
        dict(name="tau_comb",         kind="float", lo=lo,  hi=hi,
             log=hi > 4.0 * lo),
        dict(name="min_rotor_pixels", kind="int",   lo=1,   hi=60,   log=True),
        dict(name="min_box_area",     kind="int",   lo=1,   hi=300,  log=True),
        dict(name="min_fill_ratio",   kind="float", lo=0.0, hi=0.5),
        dict(name="min_snr_sum",      kind="float", lo=0.0, hi=400.0, log1p=True),
        dict(name="min_snr_avg",      kind="float", lo=0.0, hi=snr_avg_max),
        dict(name="alpha",            kind="float", lo=1.0, hi=4.0),
        dict(name="persist",          kind="cat",   choices=PERSIST),
    ]
    if track:
        dims += [
            dict(name="track_coast",   kind="int",   lo=0,   hi=8),
            dict(name="track_confirm", kind="int",   lo=1,   hi=3),
            dict(name="track_acquire", kind="float", lo=1.0, hi=3.0),
            dict(name="track_predict", kind="bool"),
        ]
    if len(drones) > 1:
        dims.append(dict(name="drone", kind="choice", choices=list(drones),
                         default=drones[0]))
    if len(present) > 1:
        dims.append(dict(name="min_present_harmonics", kind="choice",
                         choices=list(present), default=present[0]))
    if len(factors) > 1 and any(m > 0 for m in present):
        dims.append(dict(name="harmonic_present_factor", kind="choice",
                         choices=list(factors), default=factors[0]))
    return dims


def _dim_encode(d, val):
    """Parameter value -> [0, 1]."""
    if d["kind"] == "choice":                        # named options (drone)
        ch = d["choices"]
        i = ch.index(val) if val in ch else ch.index(d["default"])
        return (i + 0.5) / len(ch)
    if d["kind"] == "cat":
        try:
            i = d["choices"].index(tuple(val))
        except ValueError:
            i = int(np.argmin([abs(c[0] - val[0]) + abs(c[1] - val[1])
                               for c in d["choices"]]))
        return (i + 0.5) / len(d["choices"])
    if d["kind"] == "bool":
        return 1.0 if val else 0.0
    lo, hi = d["lo"], d["hi"]
    v = min(max(float(val), lo), hi)
    if d.get("log"):
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    if d.get("log1p"):
        return math.log1p(v) / math.log1p(hi)
    return (v - lo) / (hi - lo)


def _dim_decode(d, u):
    """[0, 1] -> parameter value."""
    u = min(max(float(u), 0.0), 1.0)
    if d["kind"] in ("cat", "choice"):
        return d["choices"][min(len(d["choices"]) - 1, int(u * len(d["choices"])))]
    if d["kind"] == "bool":
        return u >= 0.5
    lo, hi = d["lo"], d["hi"]
    if d.get("log"):
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    elif d.get("log1p"):
        v = math.expm1(u * math.log1p(hi))
    else:
        v = lo + u * (hi - lo)
    return int(round(v)) if d["kind"] == "int" else round(v, 4)


def encode(space, params):
    params = {**DEFAULT_START, **params}             # seed JSONs may lack axes
    vec = []
    for d in space:
        if d["name"] == "persist":
            vec.append(_dim_encode(d, (params["persist_m"], params["persist_k"])))
        elif d["kind"] == "choice":                  # older points lack it: they
            vec.append(_dim_encode(d, params.get(d["name"], d["default"])))  # ran on the default
        else:
            vec.append(_dim_encode(d, params[d["name"]]))
    return np.array(vec)


def decode(space, u):
    """Unit vector -> full params dict (frame_explorer keys)."""
    p = dict(DEFAULT_START)                          # track_* defaults when untracked
    for d, ui in zip(space, u):
        v = _dim_decode(d, ui)
        if d["name"] == "persist":
            p["persist_m"], p["persist_k"] = v
        else:
            p[d["name"]] = v
        if d["name"] == "harmonic_present_factor":
            factor_dim = d
    # With the presence check off the factor does nothing: pin it, so those
    # points share one dedupe key (and one feature table) instead of five.
    if "harmonic_present_factor" in p and int(p.get("min_present_harmonics", 0)) == 0:
        p["harmonic_present_factor"] = factor_dim["default"]
    return p


def params_key(space, params):
    """Canonical dedupe key: only the searched axes, canonically rounded."""
    return json.dumps(encode_decode_canon(space, params), sort_keys=True)


def encode_decode_canon(space, params):
    return decode(space, encode(space, params))


# ── GP surrogate + Expected Improvement ──────────────────────────────────────

def _matern52(dist, ls):
    a = math.sqrt(5.0) * dist / ls
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def gp_fit(X, y):
    """Fit y ~ GP(0, matern52) on standardised targets; pick (lengthscale,
    noise) from a small grid by log marginal likelihood."""
    n = len(y)
    mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
    ys = (y - mu) / sd
    dist = cdist(X, X)
    best = None
    for ls in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5):
        K0 = _matern52(dist, ls)
        for noise in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2):
            try:
                L = cholesky(K0 + noise * np.eye(n), lower=True)
            except np.linalg.LinAlgError:
                continue
            a = solve_triangular(L.T, solve_triangular(L, ys, lower=True))
            lml = (-0.5 * float(ys @ a) - float(np.log(np.diag(L)).sum())
                   - 0.5 * n * math.log(2 * math.pi))
            if best is None or lml > best[0]:
                best = (lml, ls, noise, L, a)
    _, ls, noise, L, a = best
    return dict(X=X, L=L, a=a, ls=ls, noise=noise, mu=mu, sd=sd)


def gp_predict(m, Xq):
    Kq = _matern52(cdist(Xq, m["X"]), m["ls"])
    mean = Kq @ m["a"]
    v = solve_triangular(m["L"], Kq.T, lower=True)
    var = np.clip(1.0 + m["noise"] - np.sum(v * v, axis=0), 1e-12, None)
    return mean * m["sd"] + m["mu"], np.sqrt(var) * m["sd"]


def expected_improvement(mean, sd, best, xi):
    z = (mean - best - xi) / sd
    return (mean - best - xi) * ndtr(z) + sd * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def propose(space, X, y, seen, rng, xi, n_cand=4096):
    """Argmax-EI over random candidates + jitters around the incumbents,
    skipping already-evaluated points."""
    d = len(space)
    cand = rng.random((n_cand, d))
    top = X[np.argsort(y)[-5:]]
    jit = np.clip(np.repeat(top, 256, axis=0)
                  + rng.normal(0, 0.07, (len(top) * 256, d)), 0.0, 1.0)
    cand = np.vstack([cand, jit])
    model = gp_fit(X, y)
    mean, sd = gp_predict(model, cand)
    ei = expected_improvement(mean, sd, float(np.max(y)), xi)
    for i in np.argsort(ei)[::-1]:
        p = decode(space, cand[i])
        if params_key(space, p) not in seen:
            return p
    return decode(space, rng.random(d))              # everything seen: explore


# ── objective ────────────────────────────────────────────────────────────────

def _f1(p, r):
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def metrics_from_rows(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    gtf = sum(r["gt_frames"] for r in rows)
    hit = sum(r["gt_frames_hit"] for r in rows)
    fpf = sum(r.get("fp_frames", 0) for r in rows)
    frames = sum(r["frames"] for r in rows)
    precs = [r["tp"] / max(1, r["tp"] + r["fp"]) for r in rows] or [0.0]
    f1s = [_f1(r["tp"] / max(1, r["tp"] + r["fp"]),
               r["tp"] / max(1, r["tp"] + r["fn"])) for r in rows] or [0.0]
    p_box = tp / max(1, tp + fp); r_box = tp / max(1, tp + fn)
    p_fr = hit / max(1, hit + fpf); r_fr = hit / max(1, gtf)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=round(p_box, 4),
                precision_mean=round(float(np.mean(precs)), 4),
                precision_min=round(float(np.min(precs)), 4),
                recall=round(r_box, 4),
                f1=round(_f1(p_box, r_box), 4),          # pooled box-level F1
                f1_mean=round(float(np.mean(f1s)), 4),   # mean per-case box F1
                f1_min=round(float(np.min(f1s)), 4),
                f1_frames=round(_f1(p_fr, r_fr), 4),     # frame-level F1
                precision_frames=round(p_fr, 4),
                gtf_hit_rate=round(r_fr, 4),
                fp_per_frame=round(fp / max(1, frames), 4))


def score_of(m, args):
    """The maximised quantity: the chosen quality measure, softly penalised for
    violating the recall floor / FP budget. F1 objectives balance precision and
    recall on their own, so they are normally run with --min-recall 0.

    Everything is BOX-level by default (every GT box matched is a TP, every
    stray box an FP) — including the recall floor. --recall-level frames uses
    the GT-frame hit rate instead, which gives full credit for finding one of
    two drones in a frame; the two differ only on the 25 multi-drone cases."""
    obj = getattr(args, "objective", "precision")
    if obj == "precision":
        base = {"mean": m["precision_mean"], "min": m["precision_min"],
                "pooled": m["precision"]}[args.agg]
    elif obj == "f1":                                # box-level F1
        base = {"mean": m["f1_mean"], "min": m["f1_min"],
                "pooled": m["f1"]}[args.agg]
    elif obj == "f1_frames":                         # frame-level F1 (pooled)
        base = m["f1_frames"]
    else:
        raise ValueError(f"unknown objective {obj!r}")
    rec = (m["gtf_hit_rate"] if getattr(args, "recall_level", "box") == "frames"
           else m["recall"])
    s = base - args.penalty * max(0.0, args.min_gtf - rec)
    if args.max_fppf is not None:
        s -= args.penalty * max(0.0, m["fp_per_frame"] - args.max_fppf)
    return round(s, 6)


def _case_row(seq, mode, data_dir, pp, match, thr):
    """Per-case counts; with skip_start the score uses only frames from the
    first GT sighting on, but the lead-in-INCLUSIVE counts are stored too
    (full_*) so the graphs can toggle the metric without re-evaluating."""
    pp = dict(pp)
    skip = bool(pp.pop("skip_start", False))         # keep dets unfiltered here
    D = FE.load_D(seq, mode, data_dir)
    _res, dets, _caches, _comps = FE._compute_job(D, mode, pp)
    f_lo = FE.first_gt_frame(D) if skip else None
    c = FE.count_matches(D, dets, match, thr, f_lo=f_lo)
    row = dict(seq=seq, frames=c["frames_counted"],
               gt_frames=c["gt_frames"], gt_frames_hit=c["gt_frames_hit"],
               tp=c["tp"], fp=c["fp"], fn=c["fn"],
               tn_frames=c["tn_frames"], fp_frames=c["fp_frames"])
    cf = FE.count_matches(D, dets, match, thr) if skip else c
    row.update(full_frames=cf["frames_counted"], full_fp=cf["fp"],
               full_fp_frames=cf["fp_frames"], full_tn_frames=cf["tn_frames"])
    return row


def _pool_init():
    """Worker processes ignore Ctrl-C; the parent handles it and terminates."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _pool_case(job):
    """One case in a worker process; the D cache is cleared after each case.
    MemoryError reports as 'mem' so the scheduler can retry the case solo."""
    seq, mode, data_dir, pp, match, thr = job
    try:
        row = _case_row(seq, mode, data_dir, pp, match, thr)
        return ("ok", row)
    except MemoryError as exc:
        return ("mem", f"case {seq}: {exc!r}")
    except Exception as exc:
        return ("err", f"case {seq}: {exc!r}")
    finally:
        FE._D_CACHE.clear()
        FE._CC_CACHE.update(key=None, stage0=None, cg=None)
        import gc
        gc.collect()


# Peak working set of one case is ~10x its frames_<seq>_<mode>.csv.gz size
# (measured: pixel arrays + derived columns + sort/track temporaries).
_RAM_PER_GZ_BYTE = 10.0


def _case_cost(seq, mode, data_dir):
    """Estimated peak RAM (bytes) of evaluating one case."""
    try:
        return _RAM_PER_GZ_BYTE * os.path.getsize(
            os.path.join(data_dir, f"frames_{seq}_{mode}.csv.gz"))
    except OSError:
        return 0.0


def _total_ram_bytes():
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _MSX(ctypes.Structure):
            _fields_ = ([("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD)]
                        + [(n, ctypes.c_uint64) for n in
                           ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                            "ullAvailPageFile", "ullTotalVirtual",
                            "ullAvailVirtual", "ullAvailExtendedVirtual")])
        m = _MSX()
        m.dwLength = ctypes.sizeof(_MSX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullTotalPhys)
    except Exception:
        pass
    return 16 << 30


def _evaluate_parallel(pool, nworkers, budget, seqs, args, pp, thr):
    """Memory-aware fan-out: cases are admitted to the pool largest-first only
    while their summed RAM estimates fit `budget`, so several small cases run
    in parallel but the giant ones effectively serialise instead of exhausting
    commit memory. A case that still dies of MemoryError is retried once solo
    (nothing else in flight); only a second failure fails the evaluation."""
    cost = {s: min(_case_cost(s, args.mode, args.data_dir), budget) for s in seqs}
    pending = sorted(seqs, key=lambda s: -cost[s])
    inflight, used = {}, 0.0
    rows, retried = {}, set()
    while pending or inflight:
        for s in list(pending):                      # admit what fits
            if len(inflight) >= nworkers:
                break
            if cost[s] <= budget - used or not inflight:
                pending.remove(s)
                used += cost[s]
                inflight[s] = pool.apply_async(
                    _pool_case,
                    ((s, args.mode, args.data_dir, pp, args.match, thr),))
        reaped = False
        for s, ar in list(inflight.items()):         # reap what finished
            if not ar.ready():
                continue
            reaped = True
            del inflight[s]
            used -= cost[s]
            status, payload = ar.get()
            if status == "ok":
                rows[payload["seq"]] = payload
            elif status == "mem" and s not in retried:
                retried.add(s)
                cost[s] = budget                     # forces a solo slot
                pending.append(s)                    # after the queue drains
                print(f"    case {s} hit MemoryError — will retry it solo")
            else:
                raise RuntimeError(payload + (" (even solo)" if s in retried
                                              else ""))
            print(f"    {len(rows)}/{len(seqs)} cases", end="\r", flush=True)
        if not reaped:
            time.sleep(0.1)
    print(" " * 24, end="\r")
    return [rows[s] for s in seqs]


def table_key(drone, mph, factor):
    """The feature table a (drone, min_present_harmonics, harmonic_present
    _factor) combination reads. With the presence check off (mph 0) the
    factor changes nothing, so it drops out of the key."""
    mph = int(mph)
    return (drone, mph, round(float(factor), 6) if mph > 0 else None)


def spectral_data_dirs(args, seqs):
    """{table_key: frame_data folder} for every combination of --drones x
    --present-harmonics x --present-factors (args.factors; one factor, the
    base one, when absent).

    All three change the stored sf / snr, so each combination needs its own
    feature table. --data-dir is the base: its spectral settings (from its
    variant.json, else the preset: drone 'all', min_present_harmonics 0) are
    kept and only those differ. The other tables are lab_viewer_pooled's
    variants (utility/frame_data_variants/<hash>/ - the very folders the
    viewer's --wp replay reads), built here for every case that lacks one."""
    from ddhf_config import build_cfg
    base = dict(drone="all")
    vj = os.path.join(args.data_dir, "variant.json")
    if os.path.exists(vj):
        with open(vj, encoding="utf-8") as fh:
            base = json.load(fh)
    spectral = ("window_ms", "pixel_block", "num_harmonics", "min_harmonics",
                "harmonic_half_width", "max_freq_hz", "noise_band_harmonics",
                "min_present_harmonics", "harmonic_present_factor")
    ov = {k: base[k] for k in spectral if k in base}
    if "window_ms" in ov:
        ov["step_ms"] = ov["window_ms"]
    base_cfg = build_cfg(**ov)
    base_mph = int(base_cfg.min_present_harmonics)
    base_fac = float(base_cfg.harmonic_present_factor)
    if not getattr(args, "factors", None):
        args.factors = [base_fac]
    # what --data-dir holds: a journalled point with no drone / present values
    # recorded (made before those were axes) ran on exactly this table
    args.base_key = (base["drone"], base_mph, base_fac)
    dirs = {table_key(*args.base_key): args.data_dir}
    todo = []
    for d in args.drones:
        for m in args.present:
            for f in (args.factors if m > 0 else [args.factors[0]]):
                k = table_key(d, m, f)
                if k not in dirs and k not in [table_key(*t) for t in todo]:
                    todo.append((d, m, f))
    if not todo:
        return dirs
    import lab_viewer_pooled as LV                   # lazy: main process only
    print(f"{len(todo)} feature table(s) besides --data-dir; any a run lacks "
          "is built now (the full DDHF precompute - minutes per run, once)")
    for drone, mph, fac in todo:
        cfg = build_cfg(**dict(ov, min_present_harmonics=mph,
                               harmonic_present_factor=fac))
        for seq in seqs:                             # builds what is missing
            d, _note = LV._wp_feature_dir(os.path.join(DATASETS, seq), args.mode,
                                          cfg, drone, {}, False)
        dirs[table_key(drone, mph, fac)] = d
        print(f"drone {drone}, min_present_harmonics {mph}"
              + (f", factor {fac:g}" if mph > 0 else "") + f": {d}")
    return dirs


def evaluate_point(params, seqs, args, thr, keep_cache, pool_info=None):
    """Per-case counts at `params` over `seqs` — in-process one case at a time,
    or fanned out over the worker pool (cases are independent). The drone and
    min_present_harmonics pick which feature table the cases are read from."""
    pp = dict(params)
    key = table_key(pp.pop("drone", args.drones[0]),
                    pp.pop("min_present_harmonics", args.present[0]),
                    pp.pop("harmonic_present_factor", args.factors[0]))
    args = argparse.Namespace(**vars(args))
    args.data_dir = args.spectral_dirs[key]
    pp["tracking"] = args.track; pp["track_sigma"] = args.track_sigma
    pp["match"] = args.match; pp["skip_start"] = args.skip_start
    if pool_info is not None:
        pool, nworkers, budget = pool_info
        return _evaluate_parallel(pool, nworkers, budget, seqs, args, pp, thr)
    rows = []
    for seq in seqs:
        rows.append(_case_row(seq, args.mode, args.data_dir, pp, args.match, thr))
        if not keep_cache:
            FE._D_CACHE.clear()
        FE._CC_CACHE.update(key=None, stage0=None, cg=None)
    return rows


# ── journal (the resume mechanism) ───────────────────────────────────────────

def fingerprint(args, seqs, thr):
    return (f"mode={args.mode}|track={args.track}:{args.track_sigma:g}"
            f"|match={args.match}:{thr:g}"
            + ("|skip_start" if args.skip_start else "")
            + f"|seqs={','.join(seqs)}")


def load_journal(path, fp):
    """Return the evaluation records of a compatible journal ([] if absent)."""
    if not os.path.exists(path):
        return []
    records, meta = [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "meta":
                meta = rec
            elif rec.get("kind") == "eval":
                records.append(rec)
    if meta is None or meta.get("fingerprint") != fp:
        if not records:                              # meta only — nothing to lose
            os.remove(path)
            print(f"discarded {path} (different setup, no evaluations in it)")
            return []
        raise SystemExit(
            f"{path} holds {len(records)} evaluation(s) for a different setup:\n"
            f"  journal: {meta.get('fingerprint') if meta else '<none>'}\n"
            f"  current: {fp}\n"
            "Its scores are not comparable — pass --reset to discard it or "
            "--state <other file> to keep both.")
    return records


def append_journal(path, rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ── seed points ──────────────────────────────────────────────────────────────

def seed_points(args, tag):
    """auto_tune's default start + any previously tuned points on disk."""
    seeds = [("default", dict(DEFAULT_START))]
    if args.seed_json and args.seed_json.lower() == "none":
        return seeds
    if args.seed_json:
        paths = [os.path.join(WPS_DIR, os.path.basename(s.strip()))
                 for s in args.seed_json.split(",") if s.strip()]
        for p in paths:
            if not os.path.exists(p):
                print(f"  seed {os.path.basename(p)} not in {WPS_DIR} — skipped")
    else:
        paths = [os.path.join(WPS_DIR, f"{stem}_{tag}.json")
                 for stem in ("tuned_wp", "best_point", "working_point",
                              "working_point_bayes")]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    params = json.load(fh).get("params")
                if params:
                    seeds.append((os.path.basename(p), params))
            except Exception as exc:
                print(f"  seed {p} unreadable ({exc}) — skipped")
    return seeds


def write_best(out, params, m, rows, score, seqs, args, thr, n_evals):
    """Winner as working_point_bayes_<tag>.json + the stage-4-format per-case
    CSV working_point_eval_bayes_<tag>.csv next to it, so show_case_graphs.bat
    --eval-csv <csv> plots it (and picks the params for the title from the
    companion JSON)."""
    import csv
    if not args.track:                               # drop the inert track axes
        params = {k: v for k, v in params.items() if not k.startswith("track_")}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(params=params, metrics=m, score=score,
                       objective=dict(name=args.objective, agg=args.agg,
                                      min_recall=args.min_gtf,
                                      recall_level=args.recall_level,
                                      penalty=args.penalty, max_fppf=args.max_fppf),
                       cases=seqs, mode=args.mode, match=args.match,
                       match_thr=thr, track=args.track,
                       track_sigma=args.track_sigma,
                       skip_start=args.skip_start, n_evals=n_evals),
                  fh, indent=2)
    eval_csv = _eval_csv_of(out)
    split_map = _load_split_map()
    cols = ["seq", "split", "frames", "gt_frames", "gt_frames_hit", "tp", "fp",
            "fn", "tn_frames", "fp_frames", "precision", "recall",
            "gtf_hit_rate", "fp_per_frame",
            "full_frames", "full_fp", "full_fp_frames", "full_tn_frames"]
    with open(eval_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["seq"], split_map.get(r["seq"], ""), r["frames"],
                        r["gt_frames"], r["gt_frames_hit"], r["tp"], r["fp"],
                        r["fn"], r.get("tn_frames", 0), r.get("fp_frames", 0),
                        round(r["tp"] / max(1, r["tp"] + r["fp"]), 4),
                        round(r["tp"] / max(1, r["tp"] + r["fn"]), 4),
                        round(r["gt_frames_hit"] / max(1, r["gt_frames"]), 4),
                        round(r["fp"] / max(1, r["frames"]), 4),
                        r.get("full_frames", r["frames"]),
                        r.get("full_fp", r["fp"]),
                        r.get("full_fp_frames", r.get("fp_frames", 0)),
                        r.get("full_tn_frames", r.get("tn_frames", 0))])
    return eval_csv


def _eval_csv_of(out_json):
    """working_point_bayes_x.json -> working_point_eval_bayes_x.csv (the naming
    show_case_graphs.py inverts to locate the companion JSON)."""
    base = os.path.basename(out_json)
    if base.startswith("working_point_"):
        base = "working_point_eval_" + base[len("working_point_"):]
    else:
        base = "working_point_eval_" + base
    return os.path.join(os.path.dirname(out_json) or ".",
                        os.path.splitext(base)[0] + ".csv")


# ── main ─────────────────────────────────────────────────────────────────────

def _load_split_map():
    import csv
    path = os.path.join(_LAB_DIR, "data", "drone_map", "lab_drone_map.csv")
    mp = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = (row.get("sequence") or "").strip()
                if seq:
                    mp[seq] = (row.get("split") or "").strip()
    return mp


def lab_cases(mode, data_dir):
    """The runs under lab/runs/dataset that frame_data has cached for
    `mode` - the only cases this copy ever evaluates."""
    if not os.path.isdir(DATASETS):
        return []
    out = []
    for name in sorted(os.listdir(DATASETS)):
        if not os.path.isfile(os.path.join(DATASETS, name, "Event", "events.raw")):
            continue
        base = os.path.join(data_dir, f"frames_{name}_{mode}")
        if os.path.exists(base + ".npz") or os.path.exists(base + ".csv.gz"):
            out.append(name)
    return out


def resolve_cases(args):
    lab = lab_cases(args.mode, args.data_dir)
    print(f"cached lab runs ({len(lab)}): {', '.join(lab) or 'none'}")
    if args.seqs:
        want = [s.strip() for s in args.seqs.split(",") if s.strip()]
        bad = [s for s in want if s not in lab]
        if bad:
            print(f"--seqs: not a cached lab run, ignored: {', '.join(bad)}")
        seqs = [s for s in want if s in lab]
    else:
        seqs = lab
    if args.first_n:
        seqs = seqs[:args.first_n]
    if args.stride > 1:
        seqs = seqs[::args.stride]
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("baseline", "pool", "pixelize", "components"),
                    default="baseline")
    ap.add_argument("--data-dir", default=os.path.join(_LAB_DIR, "frame_data"))
    ap.add_argument("--match", choices=FE.MATCH_MODES, default="center")
    ap.add_argument("--match-thr", type=float, default=None)
    ap.add_argument("--track", action=argparse.BooleanOptionalAction, default=True,
                    help="Tracking mode (DEFAULT ON — --no-track for the plain "
                         "detector).")
    ap.add_argument("--track-sigma", type=float, default=20.0)
    # which cases
    ap.add_argument("--seqs", default=None,
                    help="Comma-separated lab runs (default: every cached lab run).")
    ap.add_argument("--first-n", type=int, default=None,
                    help="Use only the first N cases of the list.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Evaluate every Nth case only — a cheap proxy of the "
                         "full objective for faster iterations (default 1 = "
                         "every case). NOTE: changes the objective, so it gets "
                         "its own journal fingerprint.")
    # objective
    ap.add_argument("--objective", choices=("precision", "f1", "f1_frames"),
                    default="precision",
                    help="What to maximise: 'precision' (default, needs a "
                         "--min-gtf floor to stay honest), 'f1' = box-level F1, "
                         "'f1_frames' = frame-level F1. F1 balances precision "
                         "and recall itself — run it with --min-gtf 0.")
    ap.add_argument("--agg", choices=("mean", "min", "pooled"), default="mean",
                    help="How the per-case quality aggregates: mean (default), "
                         "min = worst case, pooled = one ratio over summed "
                         "counts. Applies to precision and f1 objectives.")
    ap.add_argument("--min-recall", "--min-gtf", dest="min_gtf", type=float,
                    default=0.5,
                    help="Recall floor: penalise recall below this (default "
                         "0.5) so 'detect almost nothing' can't win. Applies to "
                         "whichever --recall-level is selected.")
    ap.add_argument("--recall-level", choices=("box", "frames"), default="box",
                    help="Which recall the floor constrains: 'box' (default) = "
                         "TP/(TP+FN) over GT boxes, so BOTH drones must be "
                         "found in the multi-drone cases; 'frames' = GT-frame "
                         "hit rate (any detection in the frame suffices).")
    ap.add_argument("--penalty", type=float, default=3.0,
                    help="Score penalty per unit of constraint violation "
                         "(default 3).")
    ap.add_argument("--tau-comb-range", default="3,130",
                    help="LO,HI of the tau_comb axis (default 3,130; searched on "
                         "a log scale when HI > 4 LO).")
    ap.add_argument("--snr-avg-max", type=float, default=10.0,
                    help="Upper end of the min_snr_avg axis (default 10; searched "
                         "0..this).")
    ap.add_argument("--drones", default="tello",
                    help="Comb bands to try, comma-separated (default 'tello'; "
                         "'tello,all' searches both). Each extra drone "
                         "needs its own feature table per run, built on first "
                         "use (the full DDHF precompute - minutes per run).")
    ap.add_argument("--present-harmonics", default="0,2,3,4",
                    help="min_present_harmonics values to try, comma-separated "
                         "(default 0,2,3,4; 0 = off; one value fixes it). Like "
                         "--drones, each value is its own feature table per "
                         "run, for every drone.")
    ap.add_argument("--present-factors", default=None,
                    help="harmonic_present_factor values to try with "
                         "min_present_harmonics > 0, comma-separated (e.g. "
                         "5,10,20,30,50; default: --data-dir's own, the "
                         "preset 2). Again one feature table per value per "
                         "run; points with the check off share one table.")
    ap.add_argument("--max-fppf", type=float, default=None,
                    help="Optional FP/frame budget (soft, same penalty).")
    # search
    ap.add_argument("--iterations", type=int, default=60,
                    help="TOTAL evaluations including journalled ones — rerunning "
                         "with a higher value continues the search (default 60).")
    ap.add_argument("--init-random", type=int, default=8,
                    help="Random probes after the seeds before the GP takes "
                         "over (default 8).")
    ap.add_argument("--xi", type=float, default=0.01,
                    help="EI exploration margin (default 0.01).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-json", default=None,
                    help="Comma-separated JSON names in data/working_points (auto_tune "
                         "format) to seed with; default: auto-discover tuned/"
                         "best/working points for this tag there; 'none' "
                         "disables.")
    # bookkeeping
    ap.add_argument("--state", default=None,
                    help="Journal file name in data/working_points (default "
                         "bayes_wp_<tag>.jsonl). Delete it or pass --reset to "
                         "start over.")
    ap.add_argument("--reset", action="store_true",
                    help="Discard an existing journal and start fresh.")
    ap.add_argument("--out", default=None,
                    help="Best-point JSON name in data/working_points (default "
                         "working_point_bayes_<tag>.json; the per-case CSV for "
                         "show_lab_case_graphs.py is written next to it).")
    ap.add_argument("--cache-cases", type=int, default=6,
                    help="Keep loaded cases in memory when there are at most "
                         "this many (default 6; big speedup on small subsets; "
                         "only applies with --workers 1).")
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 2) - 1)),
                    help="Worker processes evaluating cases in parallel "
                         "(default: min(8, cores-1)). Concurrency is also "
                         "capped by --mem-budget-gb, so big cases serialise "
                         "instead of exhausting RAM. 1 = in-process.")
    ap.add_argument("--mem-budget-gb", type=float, default=None,
                    help="Cap on the summed RAM estimates of cases in flight "
                         "(default: 45%% of physical RAM). Lower it if other "
                         "programs need the machine; raising it beyond RAM "
                         "risks MemoryErrors again.")
    args = ap.parse_args()
    args.skip_start = False          # lab runs: every frame counts (no option)
    args.split = "all"

    tag = f"{args.mode}{'_track' if args.track else ''}"
    # case-subset flags get their own default files, so e.g. a --stride 4
    # search and the full run keep separate journals / winners side by side
    sub = ((f"_n{args.first_n}" if args.first_n else "")
           + (f"_s{args.stride}" if args.stride > 1 else ""))
    os.makedirs(WPS_DIR, exist_ok=True)
    state = os.path.join(WPS_DIR, os.path.basename(args.state) if args.state
                         else f"bayes_wp_{tag}{sub}.jsonl")
    out = os.path.join(WPS_DIR, os.path.basename(args.out) if args.out
                       else f"working_point_bayes_{tag}{sub}.json")
    thr = args.match_thr if args.match_thr is not None else FE.MATCH_DEFAULT_THR[args.match]

    seqs = resolve_cases(args)
    if not seqs:
        raise SystemExit(f"no lab runs under {DATASETS} have a frame_data cache "
                         f"for [{args.mode}] in {args.data_dir} - build them with "
                         "cache_frames.bat")
    fp = fingerprint(args, seqs, thr)
    try:
        lo, hi = (float(v) for v in args.tau_comb_range.split(","))
    except ValueError:
        raise SystemExit("--tau-comb-range wants LO,HI, e.g. 3,130")
    if not 0 < lo < hi:
        raise SystemExit("--tau-comb-range: need 0 < LO < HI")
    args.drones = [d.strip() for d in args.drones.split(",") if d.strip()]
    if not args.drones:
        raise SystemExit("--drones: give at least one")
    try:
        args.present = [int(v) for v in args.present_harmonics.split(",")
                        if v.strip()]
    except ValueError:
        raise SystemExit("--present-harmonics wants integers, e.g. 0,2,3,4")
    if not args.present or any(not 0 <= m <= 4 for m in args.present):
        raise SystemExit("--present-harmonics: values 0-4 (4 harmonics are tested)")
    args.factors = None
    if args.present_factors:
        try:
            args.factors = [float(v) for v in args.present_factors.split(",")
                            if v.strip()]
        except ValueError:
            raise SystemExit("--present-factors wants numbers, e.g. 5,10,20")
        if not args.factors or any(f <= 0 for f in args.factors):
            raise SystemExit("--present-factors: positive values")
    args.spectral_dirs = spectral_data_dirs(args, seqs)   # fills args.factors
    if args.snr_avg_max <= 0:
        raise SystemExit("--snr-avg-max: need > 0")
    space = build_space(args.track, (lo, hi), args.drones, args.present,
                        args.factors, args.snr_avg_max)
    print(f"tau_comb: {lo:g}-{hi:g}{' (log scale)' if hi > 4 * lo else ''}   "
          f"drones: {', '.join(args.drones)}   "
          f"min_present_harmonics: {', '.join(map(str, args.present))}   "
          f"present factor: {', '.join(f'{f:g}' for f in args.factors)}")

    if args.reset and os.path.exists(state):
        os.remove(state)
        print(f"discarded {state}")
    records = load_journal(state, fp)
    if not os.path.exists(state):
        append_journal(state, dict(kind="meta", fingerprint=fp, tag=tag,
                                   created=time.strftime("%Y-%m-%d %H:%M:%S")))

    workers = max(1, args.workers)
    keep_cache = workers == 1 and len(seqs) <= args.cache_cases
    if keep_cache:
        FE._D_CACHE_MAX = len(seqs) + 1
    pool = pool_info = None
    if workers > 1 and len(seqs) > 1:
        n = min(workers, len(seqs))
        budget = ((args.mem_budget_gb * 2 ** 30) if args.mem_budget_gb
                  else 0.45 * _total_ram_bytes())
        # worker processes are recycled every few cases: numpy frees its
        # buffers, but heap fragmentation otherwise grows over a long run
        pool = mp.Pool(n, initializer=_pool_init, maxtasksperchild=6)
        pool_info = (pool, n, budget)
        print(f"workers: {n} processes, mem budget {budget / 2**30:.1f} GB "
              f"(cases admitted only while their RAM estimates fit)")

    # Rebuild the observation set; scores use the CURRENT objective flags.
    seen, X_list, y_list, evald = set(), [], [], []
    n_other = 0
    for rec in records:
        if rec.get("error"):
            continue          # failures (e.g. MemoryError) may pass on retry —
        # Points from before drone / min_present_harmonics were recorded ran
        # on --data-dir's table: fill that in, and leave out any point whose
        # combination is outside this run's --drones / --present-harmonics -
        # its score belongs to a different feature table.
        p = rec["params"]
        p.setdefault("drone", args.base_key[0])
        p.setdefault("min_present_harmonics", args.base_key[1])
        p.setdefault("harmonic_present_factor", args.base_key[2])
        mph = int(p["min_present_harmonics"])
        if (p["drone"] not in args.drones or mph not in args.present
                or (mph > 0 and not any(abs(float(p["harmonic_present_factor"])
                                            - f) < 1e-6 for f in args.factors))):
            n_other += 1
            continue
        key = params_key(space, rec["params"])       # leave them proposable
        seen.add(key)
        m = metrics_from_rows(rec["rows"])
        s = score_of(m, args)
        X_list.append(encode(space, rec["params"]))
        y_list.append(s)
        evald.append((s, rec["params"], m, rec["rows"]))
    if n_other:
        print(f"journal: {n_other} evaluation(s) on another drone / "
              "min_present_harmonics left out (not in --drones / "
              "--present-harmonics)")
    print("tracking: " + (f"ON (sigma={args.track_sigma:g}; searching "
                          "track_coast/confirm/acquire/predict too — pass "
                          "--no-track for the plain detector)"
                          if args.track else "off"))
    print(f"cases ({len(seqs)}): {', '.join(seqs)}")
    base_name = ("f1_frames" if args.objective == "f1_frames"
                 else f"{args.objective}_{args.agg}")
    rec_name = "gtf_hit_rate" if args.recall_level == "frames" else "recall_boxes"
    print(f"objective: {base_name}"
          + (f" - {args.penalty:g}*max(0, {args.min_gtf:g} - {rec_name})"
             if args.min_gtf > 0 else "")
          + (f" - {args.penalty:g}*max(0, fppf - {args.max_fppf:g})"
             if args.max_fppf is not None else ""))
    print(f"journal: {state}  ({len(records)} evaluation(s) loaded)")

    pending_seeds = [(name, p) for name, p in seed_points(args, tag)
                     if params_key(space, p) not in seen]
    n_warm = len(seen) + len(pending_seeds) + args.init_random
    best = max(evald, key=lambda t: t[0]) if evald else None
    if best:
        print(f"best so far: score={best[0]:.4f}  f1={best[2]['f1']:.3f}  "
              f"prec_mean={best[2]['precision_mean']:.3f}  "
              f"gtf={best[2]['gtf_hit_rate']:.3f}")

    t_run = time.perf_counter()
    try:
        while len(seen) < args.iterations:
            i = len(seen) + 1
            rng = np.random.default_rng(args.seed + 7919 * len(seen))
            if pending_seeds:
                src, params = pending_seeds.pop(0)
                params = encode_decode_canon(space, params)
            elif len(seen) < n_warm or len(y_list) < max(4, len(space) // 2) \
                    or len(set(y_list)) < 2:
                src, params = "random", decode(space, rng.random(len(space)))
            else:
                src = "gp-ei"
                params = propose(space, np.array(X_list), np.array(y_list),
                                 seen, rng, args.xi)
            key = params_key(space, params)
            if key in seen:
                continue
            # the drone and min_present_harmonics are part of the working
            # point even when not searched: the viewer's --wp replay
            # reads them to pick the feature table
            params.setdefault("drone", args.drones[0])
            params.setdefault("min_present_harmonics", args.present[0])
            params.setdefault("harmonic_present_factor", args.factors[0])

            shown = {k: v for k, v in sorted(params.items())
                     if args.track or not k.startswith("track_")}
            print(f"[{i}/{args.iterations}] ({src}) {shown}")
            t0 = time.perf_counter()
            rec = dict(kind="eval", i=i, src=src, params=params, error=None)
            try:
                rows = evaluate_point(params, seqs, args, thr, keep_cache,
                                      pool_info)
                rec["rows"] = rows
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                rec["error"] = repr(exc)
                print(f"    EVAL FAILED ({exc!r}) — point recorded and avoided")
            rec["elapsed"] = round(time.perf_counter() - t0, 1)
            append_journal(state, rec)
            seen.add(key)
            if rec["error"]:
                continue

            m = metrics_from_rows(rows)
            s = score_of(m, args)
            X_list.append(encode(space, params)); y_list.append(s)
            marker = ""
            if best is None or s > best[0]:
                best = (s, params, m, rows)
                write_best(out, params, m, rows, s, seqs, args, thr, len(seen))
                marker = "   ** new best **"
            print(f"    score={s:.4f}  F1(box/frame)={m['f1']:.3f}/"
                  f"{m['f1_frames']:.3f}  prec_mean={m['precision_mean']:.3f}  "
                  f"recall(box/frame)={m['recall']:.3f}/{m['gtf_hit_rate']:.3f}  "
                  f"fppf={m['fp_per_frame']:.3f}  ({rec['elapsed']:.0f}s)" + marker)
    except KeyboardInterrupt:
        print("\ninterrupted — journal is saved; rerun the same command to "
              "resume where it stopped.")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if best is None:
        raise SystemExit("no successful evaluations yet.")
    s, params, m, rows = best
    eval_csv = write_best(out, params, m, rows, s, seqs, args, thr, len(seen))
    dt = (time.perf_counter() - t_run) / 60.0
    print(f"\nBEST after {len(seen)} evaluation(s) ({dt:.1f} min this run)")
    print(f"  score={s:.4f}  F1 boxes={m['f1']:.3f} (mean {m['f1_mean']:.3f}) "
          f"frames={m['f1_frames']:.3f}")
    print(f"  precision mean={m['precision_mean']:.3f} "
          f"min={m['precision_min']:.3f} pooled={m['precision']:.3f}")
    print(f"  gtf_hit_rate={m['gtf_hit_rate']:.3f}  fp/frame={m['fp_per_frame']:.3f}"
          f"  recall={m['recall']:.3f}   TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print("  params:", {k: params[k] for k in sorted(params)
                        if args.track or not k.startswith("track_")})
    print(f"  saved -> {out}")
    print(f"  per-case  -> {eval_csv}")
    print(f"  graphs:     python show_lab_case_graphs.py --eval-csv "
          f"{os.path.basename(eval_csv)}")


if __name__ == "__main__":
    main()
