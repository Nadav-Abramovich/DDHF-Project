# drone_project_final

Setup and usage: see **`how_to.txt`**. Lab camera work: see **`lab/README.md`**.

| Folder | Contents |
|---|---|
| `fred/` | FRED dataset work: the `.bat` launchers, `utility/` (working-point tools, `fred_drone_map.py`) and `script/` (self-contained copy of the case-graphs tool) |
| `lab/` | Lab camera work, **self-contained** (own copy of the detector in `lab/code/common/`, own DLLs in `lab/_deps/`, own `requirements.txt` and venv) - see `lab/README.md` |
| `common/` | The FRED side's shared modules (`code/`, `utility/`), the OpenEB/libusb DLLs (`_deps/`), the Python venv and `requirements.txt` |

Each project keeps its small data in `<project>/data/<type>/`:

- `fred/data/`: `working_points/`, plus `fred_drone_map.csv`, `case_statistics.csv` and `hf_index.json`
- `lab/data/`: `working_points` (was `lab_tests/wps`), `drone_map/lab_drone_map.csv` (train/test split of the lab runs),
  `case_statistics` (read by `show_lab_case_graphs.py`), optional `rgb_offsets`, `camera_biases`, `hot_pixels`, `bias_search`

The shared modules in `common/` default to FRED data (`fred/script/cases/<seq>`,
`fred/data/...`, `fred/utility/frame_data`). The lab has its own copies of the
modules it needs (`lab/code/common/`) and never uses the top-level `common/`.

## Where large data goes

- FRED cases: downloaded on demand from HuggingFace (`GabrieleMagrini/FRED`) into
  `fred/script/cases/<N>/` (~1 GB each). `cache_frames.bat` deletes the ones it downloads
  once their cache is built; cases opened from the graphs window are kept.
- Frame-feature cache (all modes, all tools): `fred/utility/frame_data/`
- Lab recordings: `lab/runs/` (exported datasets in `lab/runs/dataset/`, with `.calib` files next to the recordings)
- Lab camera sessions: `lab/lab<N>/` (written by `lab/camera/record.bat`)
- Lab feature caches: `lab/frame_data/`, `lab/frame_data_variants/`

## External requirements

- Python 3.12 (64-bit) and a venv made from `common/requirements.txt` (`how_to.txt`, step 2).
- Prophesee Metavision SDK 5.2 in `C:\Program Files\Prophesee` (also provides the
  `metavision_*` Python modules, which is why the venv uses `--system-site-packages`).
- Internet access for the FRED cases (HuggingFace, `GabrieleMagrini/FRED`).
- Lab only: the CenturyArks SilkyEvCam and its driver in `C:\Program Files\CenturyArks`
  (`lab/README.md`).
