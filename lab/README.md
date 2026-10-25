# lab

Lab work with a real event camera: recording sessions, tuning the sensor
biases, exporting recordings as FRED-style datasets, labelling them, and
running the DDHF detector / working-point search on them.

This folder is **self-contained**: it can be copied and used on its own. It
has its own copy of the detector (`code/common/`), its own Prophesee/libusb DLLs
(`_deps/`) and its own Python requirements. Outside it you only need the
software below (Python 3.12, the Metavision SDK, and the camera driver for the
camera scripts).

## Setup (once)

1. Install Python 3.12 (64-bit, python.org; tick "Add python.exe to PATH"),
   then the **Prophesee Metavision SDK 5.2** (installs to
   `C:\Program Files\Prophesee` and puts its `metavision_*` Python modules into
   that Python - install Python first).
2. For the camera scripts: the CenturyArks driver (below).
3. In a Command Prompt opened in this folder:

       py -3.12 -m venv --system-site-packages .venv
       .venv\Scripts\python -m pip install -r requirements.txt

   (`--system-site-packages` lets the venv see the SDK's modules, which are
   not on pip.) Every launcher here uses `.venv` automatically.

## Hardware and drivers

| Needed for | What |
|---|---|
| everything that talks to the event camera | **CenturyArks SilkyEvCam VGA** (EvC3A, Gen3.1 sensor) on USB |
| the camera's driver | **CenturyArks SilkyEvCam driver / HAL plugin**, installed to `C:\Program Files\CenturyArks` (`plugins\` and `bin\`). Without it the camera is not discovered: Metavision only finds the SilkyEvCam through the CenturyArks plugin. Install it from CenturyArks' SilkyEvCam software package. |
| reading events at all (live or recorded) | **Prophesee Metavision SDK 5.2** in `C:\Program Files\Prophesee` (see Setup) |
| `dual_record.py`, `camera/record.bat` | an ordinary USB webcam for the RGB stream, next to the event camera |

The scripts add both driver folders to the DLL / plugin search path themselves
(`silky_bias_studio.py` keeps the CenturyArks plugin on `MV_HAL_PLUGIN_PATH`),
so nothing needs to be set by hand once the driver and SDK are installed. The
OpenEB / libusb DLLs the SDK needs ship in `_deps/`.

## Launchers (double-click, or run from a Command Prompt in this folder)

| Launcher | Does |
|---|---|
| `show_lab_case_graphs.bat` | per-case graphs of the lab working points. Run names are green when the exported run is on disk, red when not; click one (or double-click its column) to open it in the lab viewer or frame_explorer - a missing feature cache is built first, with progress shown in the window |
| `cache_frames.bat [mode]` | build the feature cache (`lab/frame_data/`) for every exported run; asks for the mode when none is given; resumable |
| `bayes_working_point.bat [--mode m ...]` | search a lab working point over the cached runs; asks for the mode (and shows how many runs are cached per mode) when none is given; resumable. The winner lands in `data/working_points/` and shows up in `show_lab_case_graphs.bat` |
| `compare_replay.bat [rec]` | play a `dual_record` recording (RGB / events / overlay) and pick the registration pairs; lists the recordings to choose from when none is given |
| `export_dataset.bat [rec]` | export a recording as a FRED-style run into `runs/dataset/`; lists the recordings when none is given |
| `gt_editor.bat [run]` | draw / edit the ground-truth boxes of an exported run; lists the exported runs when none is given |
| `camera\record.bat`, `camera\view.bat` | record / view a `lab<N>` session (see below) |

The usual chain: record with `camera\dual_record.py` -> `compare_replay.bat`
(pick the registration pairs) -> `export_dataset.bat` -> `gt_editor.bat` (draw the
GT boxes) -> `cache_frames.bat` -> `bayes_working_point.bat` ->
`show_lab_case_graphs.bat`.

## Layout: `camera/` needs the camera, `code/` works on recordings

**`camera/` - camera plugged in:**

| Script | For |
|---|---|
| `silky_bias_studio.py` | live viewer + interactive bias tuner (the others build on it) |
| `dual_record.py` | record events + webcam into `lab/runs/rec_*` (start of the offline chain below) |
| `live_detect_viewer.py` | live view with the drone detector drawn on top |
| `bias_optimize.py`, `event_yield_optimize.py` | automatic bias search (detectability / event yield) |
| `hot_pixels.py` | find + mask hot pixels (`lab/data/hot_pixels/cam.hotpx`) |
| `flicker_check.py`, `roi_spectrum.py` | rotor-vs-lamp check, full peak list of one region |
| `record.bat` (`main.py`, `undistort_padded.py`) | second recorder: FRED-layout sessions in `lab/lab<N>/` |
| `view.bat` | open a `record.bat` session in the debug viewer (no camera needed) |

**`code/` - recordings only.** The lab's own tools are in `code/`; the
modules they share - the lab's copy of the detector and its pipeline
(`utils.py`, `utils_pooled.py`, `ddhf_config.py`, `debug_viewer_pooled.py`,
`frame_explorer.py`, `frame_features.py`, `bayes_working_point.py`, ...) - are
in `code/common/`. They are copies of the FRED side's modules with every path
pointing inside `lab/`:

| Script | For |
|---|---|
| `compare_replay.py` | play a `dual_record` run (RGB / events / overlay); pick the registration pairs |
| `export_dataset.py` | turn a run into a FRED-style dataset in `lab/runs/dataset/` |
| `gt_editor.py` | draw the ground-truth boxes on an exported run |
| `lab_viewer_pooled.py` | the lab debug viewer (webcam frame + events + detections) |
| `lab_bayes_working_point.py` | working-point search over the exported runs (`bayes_working_point.bat`) |
| `show_lab_case_graphs.py` | per-case graphs of the lab working points (`show_lab_case_graphs.bat`) |

`OLD/` keeps three tools nothing else uses any more (they still run, using
`code/` and `code/common/`): `view_dataset.py` (open an exported run in the lab viewer with the r80
working point), `replay_overlay.py` (a raw recording with detections over the
RGB video) and `event_debug_viewer.py` (events-only debugger for loose `.raw`
files from the studio).

The recordings themselves are not part of this folder (they are large). The
offline tools look for them in `lab/runs/` (exported datasets in
`lab/runs/dataset/`) and `lab/lab<N>/` (sessions from `camera/record.bat`);
record new ones with `camera/dual_record.py` or `camera/record.bat`. Feature
caches go to `lab/frame_data/`.

## Data shipped here (`data/`)

- `camera_biases/` - saved bias settings (`.bias`), loadable in the studio.
  The recordings were made with `freqcam_quadrotor.bias` (the `dual_record.py`
  default). It was not tuned here: it is the "fast (quad rotor)" setting that
  B. Pfrommer published for the same camera (CenturyArks SilkyEvCam EvC3A,
  Gen3 VGA sensor) in *Frequency Cam: Imaging Periodic Signals in Real-Time*,
  arXiv:2211.00198, Table 1. `freqcam_fast.bias` is the "fast (single pixel)"
  row of that table. Pfrommer meant that one for a one-pixel ROI only: on the
  full sensor it saturates the readout.

  | Table 1 row | diff | diff_off | diff_on | fo | hpf | pr | refr |
  |---|---|---|---|---|---|---|---|
  | default | 299 | 221 | 384 | 1477 | 1448 | 1250 | 1500 |
  | fast (LEDs) | 299 | 221 | 384 | 1430 | 1448 | 1250 | 1400 |
  | **fast (quad rotor)** = `freqcam_quadrotor.bias` | 299 | 221 | 384 | 1399 | 1250 | 1250 | 1500 |
  | fast (single pixel) = `freqcam_fast.bias` | 299 | 234 | 374 | 1250 | 1525 | 1247 | 1300 |
- `hot_pixels/cam.hotpx` - this camera's measured hot-pixel mask
- `working_points/` - lab working points and their per-case scores
- `drone_map/lab_drone_map.csv` - train/test split of the lab runs
- `bias_search/` - bias-optimisation journals
