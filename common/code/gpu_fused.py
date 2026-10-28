"""Fused GPU spectral stage: NUFT power -> spectral flatness -> harmonic-comb SNR
(mean or exact median noise floor) -> optional subharmonic rejection, in ONE CUDA
kernel with one thread block per pixel group. The group's power spectrum lives in
shared memory only; just (sf, snr, omega) per group go back to the host.

Same definitions as utils._ndft_power_batch / _spectral_flatness_batch /
_comb_snr_batch / _subharmonic_reject (float32 spectrum, float64 reductions), so
results match the CPU path to ~1e-5 relative. Opt-in: debug_viewer_pooled uses it
when DDHF_GPU_FUSED=1 (frame_features.py --gpu-fused); default runs are unchanged.

Median noise floor without re-sorting per omega: each group's K bins are sorted
once (bitonic, shared memory); the j-th smallest NOISE bin of an omega is then the
smallest sorted position p with  (p + 1) - #{excluded bins of rank <= p} = j + 1,
found by binary search over p in [j, j + n_excluded]. Ties do not matter: the
order statistic is a value, not a bin."""
import numpy as np

from utils import _os, _bin_hz

try:
    import cupy as _cp
except Exception:                                    # cupy missing: callers fall back
    _cp = None

_TPB = 256          # threads per block
_MAXK = 1024        # max spectrum length (shared-memory arrays)
_MAXE = 160         # max excluded bins per comb (main comb or subharmonic comb)

_SRC = r'''
#define TPB 256
#define MAXK 1024
#define MAXE 160
#define KPT (MAXK / TPB)

__device__ float ostat(int j, const short* rE, int nE, const float* sv, int K) {
    // value of the j-th smallest (0-based) bin NOT in the excluded set
    int lo = j, hi = min(K - 1, j + nE);
    while (lo < hi) {
        int p = (lo + hi) >> 1, cnt = 0;
        for (int e = 0; e < nE; ++e) cnt += (rE[e] <= p);
        if (p + 1 - cnt >= j + 1) hi = p; else lo = p + 1;
    }
    return sv[lo];
}

__device__ float wmax(const float* P, int lo, int hi) {
    float m = P[lo];
    for (int b = lo + 1; b <= hi; ++b) m = fmaxf(m, P[b]);
    return m;
}

__device__ float comb_snr_at(const float* P, const float* sv, const short* rk, int K,
                             double f0, int M, int hmin, int hw) {
    // utils._comb_snr_at: centres rint(m*f0), windows [max(1,c-hw), min(K-1,c+hw)],
    // DC + windows excluded, median of the rest
    short rE[MAXE]; int nE = 0, nc = 0, last = 0;
    rE[nE++] = rk[0];                                     // DC
    double psum = 0.0;
    for (int m = 1; m <= M; ++m) {
        int c = (int)rint(m * f0);
        if (c < 1 || c >= K) continue;
        ++nc;
        int lo = max(1, c - hw), hi = min(K - 1, c + hw);
        psum += (double)wmax(P, lo, hi);
        for (int b = max(lo, last + 1); b <= hi && nE < MAXE; ++b) rE[nE++] = rk[b];
        last = max(last, hi);
    }
    if (nc < hmin || nE >= K) return 0.f;
    int nnz = K - nE;
    double med = 0.5 * ((double)ostat((nnz - 1) / 2, rE, nE, sv, K) + (double)ostat(nnz / 2, rE, nE, sv, K));
    if (med < 1e-10) return 0.f;
    return (float)((psum / nc) / med);
}

__device__ float between_peak(const float* P, int K, double s, int k, int M, int dk, double fmin) {
    double sum = 0.0; int n = 0;
    for (int j = 1; j < k * M; ++j) {
        if (j % k == 0 || j * s < fmin) continue;
        int c = (int)rint(j * s);
        if (c + dk >= K) continue;
        sum += (double)wmax(P, max(1, c - dk), c + dk); ++n;
    }
    return n ? (float)(sum / n) : 0.f;
}

extern "C" __global__ void fused(
    const float* __restrict__ tn, const float* __restrict__ pol, const int* __restrict__ gstart,
    int G, int N, int K, int Kp2, float kstep,
    int n_om, const int* __restrict__ om_val, const int* __restrict__ hoff,
    const int* __restrict__ hlo, const int* __restrict__ hhi,
    const int* __restrict__ eoff, const short* __restrict__ ebin, const int* __restrict__ nnz_arr,
    int noise_mean, int need_sort,
    int sub_mode, int M, int hmin, int dk, int hw, double fmin, int kmin,
    double bin_hz, int n_bands, const double* __restrict__ bands, float ratio,
    float* __restrict__ sf_out, float* __restrict__ snr_out, int* __restrict__ om_out)
{
    __shared__ float P[MAXK];
    __shared__ float sv[MAXK];
    __shared__ short si[MAXK];
    __shared__ short rk[MAXK];
    __shared__ float tt[TPB], pp[TPB];
    __shared__ double r1[TPB], r2[TPB], r3[TPB];
    __shared__ float bs[TPB];
    __shared__ int bi[TPB];
    const int g = blockIdx.x, tid = threadIdx.x;
    const int s0 = gstart[g], e0 = (g + 1 < G) ? gstart[g + 1] : N;

    // ---- 1. NUFT power: thread owns bins k = tid + j*TPB ----
    float re[KPT], im[KPT];
    #pragma unroll
    for (int j = 0; j < KPT; ++j) { re[j] = 0.f; im[j] = 0.f; }
    for (int c = s0; c < e0; c += TPB) {
        const int n = min(TPB, e0 - c);
        __syncthreads();
        if (tid < n) { tt[tid] = tn[c + tid]; pp[tid] = pol[c + tid]; }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int k = tid + j * TPB;
            if (k < K) {
                const float kk = k * kstep;
                float ar = re[j], ai = im[j];
                for (int i = 0; i < n; ++i) {
                    float sn, cs; sincosf(kk * tt[i], &sn, &cs);
                    ar += pp[i] * cs; ai += pp[i] * sn;
                }
                re[j] = ar; im[j] = ai;
            }
        }
    }
    double slog = 0.0, sp = 0.0, sall = 0.0;
    #pragma unroll
    for (int j = 0; j < KPT; ++j) {
        const int k = tid + j * TPB;
        if (k < K) {
            const float p = re[j] * re[j] + im[j] * im[j];
            P[k] = p; sall += (double)p;
            if (k >= 1) { const double q = (double)p + 1e-12; slog += log(q); sp += q; }
        }
    }
    r1[tid] = slog; r2[tid] = sp; r3[tid] = sall;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1) {
        if (tid < s) { r1[tid] += r1[tid + s]; r2[tid] += r2[tid + s]; r3[tid] += r3[tid + s]; }
        __syncthreads();
    }
    const double S_all = r3[0];
    if (tid == 0) sf_out[g] = (float)(exp(r1[0] / (K - 1)) / (r2[0] / (K - 1)));

    // ---- 2. sort the group's bins once (bitonic), rank of every bin ----
    if (need_sort) {
        for (int i = tid; i < Kp2; i += TPB) {
            sv[i] = (i < K) ? P[i] : __int_as_float(0x7f800000); si[i] = (short)i;
        }
        __syncthreads();
        for (int size = 2; size <= Kp2; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < Kp2; i += TPB) {
                    const int l = i ^ stride;
                    if (l > i) {
                        const bool asc = (i & size) == 0;
                        const float a = sv[i], b = sv[l];
                        if ((a > b) == asc) {
                            sv[i] = b; sv[l] = a;
                            const short t = si[i]; si[i] = si[l]; si[l] = t;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int p = tid; p < K; p += TPB) rk[si[p]] = (short)p;
        __syncthreads();
    }

    // ---- 3. comb SNR over all candidate fundamentals, best (first max) wins ----
    float best_t = 0.f; int bidx_t = -1;
    for (int o = tid; o < n_om; o += TPB) {
        double psum = 0.0;
        const int h0 = hoff[o], h1 = hoff[o + 1];
        for (int h = h0; h < h1; ++h) psum += (double)wmax(P, hlo[h], hhi[h]);
        const double mpk = psum / (h1 - h0);
        const int ea = eoff[o], eb = eoff[o + 1], nnz = nnz_arr[o];
        double noise;
        if (noise_mean) {
            double ex = 0.0;
            for (int e = ea; e < eb; ++e) ex += (double)P[ebin[e]];
            noise = (S_all - ex) / nnz;
        } else {
            short rE[MAXE]; const int nE = eb - ea;
            for (int e = 0; e < nE; ++e) rE[e] = rk[ebin[ea + e]];
            noise = 0.5 * ((double)ostat((nnz - 1) / 2, rE, nE, sv, K) + (double)ostat(nnz / 2, rE, nE, sv, K));
        }
        const float snr = (noise >= 1e-10) ? (float)(mpk / noise) : 0.f;
        if (snr > best_t) { best_t = snr; bidx_t = o; }
    }
    bs[tid] = best_t; bi[tid] = bidx_t;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = bs[tid], b = bs[tid + s]; const int ia = bi[tid], ib = bi[tid + s];
            if (b > a || (b == a && ib >= 0 && (ia < 0 || ib < ia))) { bs[tid] = b; bi[tid] = ib; }
        }
        __syncthreads();
    }
    float best = bs[0];
    const int om = (best > 0.f && bi[0] >= 0) ? om_val[bi[0]] : 0;
    if (best <= 0.f) best = 0.f;
    __syncthreads();

    // ---- 4. subharmonic rejection of the winning fundamental ----
    if (sub_mode > 0 && om > 0 && best > 0.f) {
        float sub_t = 0.f;
        const int nk = (int)floor(om / fmin) - kmin + 1;  // k with om/k >= fmin
        for (int t = tid; t < max(nk, 0) + 2; t += TPB) {
            const int k = kmin + t;
            if ((double)om / k < fmin) continue;
            const double s = (double)om / k;
            bool inband = false;
            for (int b = 0; b < n_bands; ++b)
                if (bands[2 * b] <= s * bin_hz && s * bin_hz <= bands[2 * b + 1]) inband = true;
            if (inband) continue;
            const float v = (sub_mode == 2) ? between_peak(P, K, s, k, M, dk, fmin)
                                            : comb_snr_at(P, sv, rk, K, s, M, hmin, dk);
            sub_t = fmaxf(sub_t, v);
        }
        bs[tid] = sub_t;
        __syncthreads();
        for (int s = TPB / 2; s > 0; s >>= 1) {
            if (tid < s) bs[tid] = fmaxf(bs[tid], bs[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            float ref;
            if (sub_mode == 2) {
                double rs = 0.0; int nr = 0;
                for (int c = om; c <= M * om; c += om)
                    if (c < K) { rs += (double)wmax(P, max(1, c - hw), min(K - 1, c + hw)); ++nr; }
                ref = nr ? (float)(rs / nr) : 0.f;
            } else {
                ref = comb_snr_at(P, sv, rk, K, (double)om, M, hmin, dk);
            }
            if (bs[0] > ratio * ref) best = 0.f;
        }
    }
    if (tid == 0) { snr_out[g] = best; om_out[g] = om; }
}
'''

_kernel = None
_PLAN = {}


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _cp.RawKernel(_SRC, "fused", options=("--std=c++14",))
    return _kernel


def supported(cfg) -> bool:
    """Options the fused kernel implements; anything else must use the old paths."""
    return (_cp is not None
            and cfg.frequency_bins <= _MAXK
            and not getattr(cfg, "detrend_polarity", False)
            and not (getattr(cfg, "snr_median_floor_frac", 0.0) or 0.0) > 0
            and not (getattr(cfg, "min_present_harmonics", 0) or 0) > 0
            and getattr(cfg, "comb_noise_stat", "median") in ("median", "mean")
            and getattr(cfg, "subharmonic_mode", "comb") in ("comb", "between"))


def _plan(cfg):
    """Per-omega harmonic windows and excluded (non-noise) bins, exactly as
    utils._comb_snr_batch selects them. Cached per configuration."""
    K = cfg.frequency_bins
    key = (K, cfg.num_harmonics, cfg.min_harmonics, cfg.harmonic_half_width, cfg.window_ms,
           cfg.rotor_bands_hz, getattr(cfg, "noise_band_harmonics", 0.0), _os(cfg))
    pl = _PLAN.get(key)
    if pl is not None:
        return pl
    bands, bin_hz = cfg.rotor_bands_hz, _bin_hz(cfg)
    hw = cfg.harmonic_half_width * _os(cfg)
    oms, hoff, hlo, hhi, eoff, ebin, nnz = [], [0], [], [], [0], [], []
    for omega in range(1, K // cfg.num_harmonics + 1):
        if bands and not any(lo <= omega * bin_hz <= hi for lo, hi in bands):
            continue
        harmonics = [m * omega for m in range(1, cfg.num_harmonics + 1) if m * omega < K]
        if len(harmonics) < cfg.min_harmonics:
            continue
        hmask = np.zeros(K, bool)
        for h in harmonics:
            lo, hi = max(0, h - hw), min(K - 1, h + hw)
            hlo.append(lo); hhi.append(hi); hmask[lo:hi + 1] = True
        noise = ~hmask
        noise[0] = False
        nbh = getattr(cfg, "noise_band_harmonics", 0.0)
        if nbh > 0:
            noise[min(K, int(nbh * harmonics[-1]) + hw + 1):] = False
        if not noise.any():
            del hlo[len(hlo) - len(harmonics):], hhi[len(hhi) - len(harmonics):]
            continue
        ex = np.flatnonzero(~noise)
        if ex.size > _MAXE:
            raise NotImplementedError(f"{ex.size} excluded bins > {_MAXE}")
        oms.append(omega); hoff.append(len(hlo))
        ebin.extend(ex.tolist()); eoff.append(len(ebin)); nnz.append(int(noise.sum()))
    i32 = lambda a: _cp.asarray(np.asarray(a, np.int32))
    pl = dict(n_om=len(oms), om=i32(oms), hoff=i32(hoff), hlo=i32(hlo), hhi=i32(hhi),
              eoff=i32(eoff), ebin=_cp.asarray(np.asarray(ebin, np.int16)), nnz=i32(nnz),
              bands=_cp.asarray(np.asarray(bands or [(0.0, -1.0)], np.float64).ravel()),
              n_bands=len(bands or ()))
    _PLAN[key] = pl
    return pl


def fused_stats(pol_k, tn_k, starts, cfg):
    """(sf, snr, omega) per pixel group, numpy float64/float64/int64 — the same
    triple compute_pixel_stats gets from the NUFT -> SF -> comb chain."""
    G = int(starts.size); N = int(tn_k.size); K = int(cfg.frequency_bins)
    pl = _plan(cfg)
    sub = getattr(cfg, "subharmonic_reject", False)
    sub_mode = 0 if not sub else (2 if getattr(cfg, "subharmonic_mode", "comb") == "between" else 1)
    noise_mean = getattr(cfg, "comb_noise_stat", "median") == "mean"
    need_sort = int((not noise_mean) or sub_mode == 1)
    Kp2 = 1 << int(np.ceil(np.log2(max(K, 2))))
    sf = _cp.empty(G, _cp.float32); snr = _cp.empty(G, _cp.float32); om = _cp.empty(G, _cp.int32)
    if G == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    if pl["n_om"] == 0:
        raise NotImplementedError("empty comb plan")
    o = _os(cfg)
    _get_kernel()((G,), (_TPB,), (
        _cp.asarray(tn_k, _cp.float32), _cp.asarray(pol_k, _cp.float32),
        _cp.asarray(starts, _cp.int32), np.int32(G), np.int32(N), np.int32(K), np.int32(Kp2),
        np.float32(1.0 / o),
        np.int32(pl["n_om"]), pl["om"], pl["hoff"], pl["hlo"], pl["hhi"], pl["eoff"], pl["ebin"], pl["nnz"],
        np.int32(noise_mean), np.int32(need_sort),
        np.int32(sub_mode), np.int32(cfg.num_harmonics), np.int32(cfg.min_harmonics),
        np.int32(int(getattr(cfg, "subharmonic_half_width", 1)) * o),
        np.int32(int(cfg.harmonic_half_width) * o),
        np.float64(getattr(cfg, "subharmonic_min_hz", 60.0) / _bin_hz(cfg)),
        np.int32(max(2, int(getattr(cfg, "subharmonic_min_k", 2)))),
        np.float64(_bin_hz(cfg)), np.int32(pl["n_bands"]), pl["bands"],
        np.float32(getattr(cfg, "subharmonic_ratio", 1.0)),
        sf, snr, om))
    return (_cp.asnumpy(sf).astype(np.float64), _cp.asnumpy(snr).astype(np.float64),
            _cp.asnumpy(om).astype(np.int64))
