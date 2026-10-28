"""
ddhf_config.py — Detector hyperparameter preset + CLI plumbing for the pooled
DDHF evaluation.

Kept separate from ddhf_eval_pooled (and free of the Metavision / CuPy imports)
so the batch driver can define its CLI flags, and build/print/forward the config,
WITHOUT importing the eval module (which loads the Prophesee SDK in the parent).

Single source of truth: _PRESET holds the defaults. Every CLI flag defaults to
None and only overrides the field the user actually passed, so changing a value
on the command line never silently drops the rest of the preset.
"""
from utils import DDHFConfig

# The preset = the defaults you ship. Edit values here; CLI flags override.
_PRESET = dict(
    window_ms=33.333, step_ms=33.333,
    tau_sf=0.5, tau_comb=6, min_events=16,
    min_own_events=0,           # extra POOLING gate on the centre pixel's own events (0=off)
    min_box_area=50, alpha=1.5,
    min_rotor_pixels=60,        # works ONLY because the strict gate keeps noise sparse
    num_harmonics=4, min_harmonics=4,
    min_present_harmonics=0,    # >0: require this many harmonics individually above noise (off)
    harmonic_present_factor=2.0,  # per-harmonic "present" threshold (× noise median)
    noise_band_harmonics=0.0,   # >0: local noise floor up to C×(top harmonic) (0 = full band)
    harmonic_half_width=3, max_freq_hz=6300.0,
    pixel_block=3,              # pixelization block size BxB (used by pixelize mode)
    subharmonic_reject=False,   # reject combs that are harmonics of a lower source (off)
    subharmonic_half_width=1, subharmonic_min_hz=60.0, subharmonic_ratio=1.0,
)

# CLI flag stem -> (DDHFConfig field, type). window_ms/step_ms are driven by the
# existing --window-ms flag (see add_cfg_cli_args usage), so they're not here.
_CFG_CLI_FIELDS = [
    ("tau-sf",              "tau_sf",              float),
    ("tau-comb",            "tau_comb",            float),
    ("min-events",          "min_events",          int),
    ("min-own-events",      "min_own_events",      int),
    ("min-box-area",        "min_box_area",        int),
    ("alpha",               "alpha",               float),
    ("min-rotor-pixels",    "min_rotor_pixels",    int),
    ("num-harmonics",       "num_harmonics",       int),
    ("min-harmonics",       "min_harmonics",       int),
    ("min-present-harmonics",   "min_present_harmonics",   int),
    ("harmonic-present-factor", "harmonic_present_factor", float),
    ("noise-band-harmonics",    "noise_band_harmonics",    float),
    ("harmonic-half-width", "harmonic_half_width", int),
    ("max-freq-hz",         "max_freq_hz",         float),
    ("pixel-block",         "pixel_block",         int),
    ("subharmonic-reject",      "subharmonic_reject",      int),   # 1 = on
    ("subharmonic-half-width",  "subharmonic_half_width",  int),
    ("subharmonic-min-hz",      "subharmonic_min_hz",      float),
    ("subharmonic-ratio",       "subharmonic_ratio",       float),
]


def build_cfg(**overrides):
    """Preset hyperparameters with optional per-field overrides. None values are
    ignored, so a CLI flag the user did not pass keeps the preset default."""
    params = dict(_PRESET)
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    return DDHFConfig(**params)


def format_cfg(cfg):
    """Render a DDHFConfig as a copy-pasteable DDHFConfig(...) block. Values are
    read from the live object, so the printed block always matches what ran."""
    return "\n".join([
        "cfg = DDHFConfig(",
        f"    window_ms={cfg.window_ms}, step_ms={cfg.step_ms},",
        f"    tau_sf={cfg.tau_sf}, tau_comb={cfg.tau_comb}, min_events={cfg.min_events},",
        f"    min_own_events={cfg.min_own_events},  # POOLING-only: centre pixel's own events",
        f"    min_box_area={cfg.min_box_area}, alpha={cfg.alpha},",
        f"    min_rotor_pixels={cfg.min_rotor_pixels},  # works ONLY because the strict gate keeps noise sparse",
        f"    num_harmonics={cfg.num_harmonics}, min_harmonics={cfg.min_harmonics},",
        f"    min_present_harmonics={cfg.min_present_harmonics}, "
        f"harmonic_present_factor={cfg.harmonic_present_factor},",
        f"    noise_band_harmonics={cfg.noise_band_harmonics},",
        f"    harmonic_half_width={cfg.harmonic_half_width}, max_freq_hz={cfg.max_freq_hz},",
        f"    pixel_block={cfg.pixel_block},  # pixelization block size BxB (pixelize mode)",
        f"    subharmonic_reject={bool(cfg.subharmonic_reject)}, "
        f"subharmonic_half_width={cfg.subharmonic_half_width}, "
        f"subharmonic_min_hz={cfg.subharmonic_min_hz}, "
        f"subharmonic_ratio={cfg.subharmonic_ratio},",
        f"    rotor_bands_hz={cfg.rotor_bands_hz},",
        f"    # derived: frequency_bins K={cfg.frequency_bins}",
        ")",
    ])


def add_cfg_cli_args(parser):
    """Add --tau-sf / --tau-comb / ... override flags (default None -> preset kept).
    The help text shows each preset value so `--help` documents the defaults."""
    for flag, field, typ in _CFG_CLI_FIELDS:
        parser.add_argument(f"--{flag}", type=typ, default=None,
                            help=f"Override cfg.{field} (preset: {_PRESET[field]}).")


def cfg_overrides_from_args(args):
    """Dict {field: value} for only the cfg flags the user actually passed."""
    out = {}
    for _flag, field, _typ in _CFG_CLI_FIELDS:
        v = getattr(args, field, None)
        if v is not None:
            out[field] = v
    return out


def cfg_args_to_cli(args):
    """Re-emit the passed cfg overrides as ['--flag', 'value', ...] so a child
    process (the per-sequence eval) builds the identical config."""
    out = []
    for flag, field, _typ in _CFG_CLI_FIELDS:
        v = getattr(args, field, None)
        if v is not None:
            out += [f"--{flag}", str(v)]
    return out


# ── Per-drone rotor bands ───────────────────────────────────────────────────────
# Blade-pass fundamental band [Hz] for each drone class, keyed by a lowercase
# substring of the drone_class(es) string in fred_drone_map.csv. Mirrors the band
# layout documented on DDHFConfig.rotor_bands_hz. Used to restrict the comb search
# to ONLY the drone(s) actually present in a sequence (see bands_for_drones).
_DRONE_BAND_HZ = {
    "dji mini 2": (280.0, 340.0),
    "dji mini 3": (280.0, 340.0),
    "dji tello":  (450.0, 700.0),
    "darwinfpv":  (800.0, 1500.0),   # cineape20 / cineape20ger
    "betafpv":    (800.0, 1500.0),   # air75
}


def bands_for_drones(drone_classes):
    """Map a fred_drone_map drone_class(es) string (drones joined by '|') to the
    tuple of rotor_bands_hz covering ONLY those drones. Unknown names are skipped;
    returns () when nothing matched (caller should fall back to all bands)."""
    bands = []
    for seg in str(drone_classes).split("|"):
        s = seg.strip().lower()
        if not s:
            continue
        for key, band in _DRONE_BAND_HZ.items():
            if key in s and band not in bands:
                bands.append(band)
                break
    return tuple(bands)


def rotor_bands_to_cli(bands):
    """Serialise a rotor_bands_hz tuple to a 'lo:hi,lo:hi' CLI string."""
    return ",".join(f"{lo}:{hi}" for lo, hi in bands)


def rotor_bands_from_cli(s):
    """Parse a 'lo:hi,lo:hi' CLI string back into a rotor_bands_hz tuple."""
    bands = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition(":")
        bands.append((float(lo), float(hi)))
    return tuple(bands)


# ── Run description (mode + parameters) for CLI banners and output files ─────────

def mode_label(pooling, pixelize, pixel_block=3):
    """Human label for the detector variant a run used: single-pixel (no spatial
    aggregation), pixelization (BxB blocks), or 3x3 spatial pooling. Shared by the
    eval and both batches so the CLI, the CSV and the xlsx all agree."""
    if not pooling:
        return "single-pixel (no pooling)"
    if pixelize:
        return f"pixelize {pixel_block}x{pixel_block}"
    return "3x3 pooled"


# The detector variant is selected by a single required --mode flag. These are the
# accepted values and their mapping to the internal (pooling, pixelize) booleans.
MODE_CHOICES = ("pixelize", "pool", "single")


def add_mode_cli_arg(parser):
    """Add the required --mode flag selecting the detector variant. Replaces the
    old --no-pooling / --pixelize / --no-pixelize trio so the choice is explicit
    and unambiguous (no more 'which flag wins')."""
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES,
                        help="Detector variant (required): 'pixelize' = non-overlapping BxB "
                             "blocks, one NDFT per block (set --pixel-block); 'pool' = 3x3 "
                             "spatial pooling; 'single' = single-pixel baseline, no spatial "
                             "aggregation.")


def pooling_pixelize_from_mode(mode):
    """Map a --mode value to the (pooling, pixelize) booleans used internally."""
    return {
        "single":   (False, False),
        "pool":     (True,  False),
        "pixelize": (True,  True),
    }[mode]


def run_param_items(args, mode_tag, cfg):
    """Ordered (label, value) pairs fully describing one run for the output files
    and the CLI: the detector MODE, the run knobs from the command line, and every
    DDHFConfig hyperparameter read off the live cfg (so the files record exactly
    what produced the numbers). `args` may be any object with the attributes; any
    that are missing are simply skipped."""
    items = [("mode", mode_tag)]
    for label in ("window_ms", "iou_thr", "match", "center_dist_scale", "center_dist_px",
                  "batch_windows", "target_present_drones", "stats_from_first_gt"):
        if hasattr(args, label):
            items.append((label, getattr(args, label)))
    items += [
        ("tau_sf", cfg.tau_sf),
        ("tau_comb", cfg.tau_comb),
        ("min_events", cfg.min_events),
        ("min_own_events", cfg.min_own_events),
        ("min_box_area", cfg.min_box_area),
        ("alpha", cfg.alpha),
        ("min_rotor_pixels", cfg.min_rotor_pixels),
        ("num_harmonics", cfg.num_harmonics),
        ("min_harmonics", cfg.min_harmonics),
        ("min_present_harmonics", cfg.min_present_harmonics),
        ("harmonic_present_factor", cfg.harmonic_present_factor),
        ("noise_band_harmonics", cfg.noise_band_harmonics),
        ("harmonic_half_width", cfg.harmonic_half_width),
        ("max_freq_hz", cfg.max_freq_hz),
        ("pixel_block", cfg.pixel_block),
        ("subharmonic_reject", bool(cfg.subharmonic_reject)),
        ("subharmonic_half_width", cfg.subharmonic_half_width),
        ("subharmonic_min_hz", cfg.subharmonic_min_hz),
        ("subharmonic_ratio", cfg.subharmonic_ratio),
        ("rotor_bands_hz", rotor_bands_to_cli(cfg.rotor_bands_hz) if cfg.rotor_bands_hz else "(all)"),
        ("frequency_bins_K", cfg.frequency_bins),
    ]
    return items


def format_run_params(items, prefix=""):
    """Render run_param_items as aligned 'label : value' lines (one per param),
    each prefixed with `prefix` (e.g. '# ' for a CSV comment header)."""
    w = max(len(k) for k, _ in items)
    return "\n".join(f"{prefix}{k.ljust(w)} : {v}" for k, v in items)
