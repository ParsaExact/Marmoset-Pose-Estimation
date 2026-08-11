# Running MarmoPose locally on macOS (Apple Silicon, CPU-only)

Verified on macOS 26.5 / arm64 / Python 3.10.13. No CUDA, no TensorRT.
The full pipeline works, including MarmoPose's own 2D detection and pose models.

## Why the documented install fails here

`marmopose/processing/prediction.py` is the only file that imports
`mmcv` / `mmdet` / `mmpose` / `mmdeploy`. Everything else (calibration, triangulation,
optimization, visualization) needs only ordinary scientific Python.

Building `mmcv` from source used to fail, but **not** because of ARM or CUDA. The real
error is in PyTorch's own header:

```
torch/include/c10/util/strong_type.h:1620:8: error: 'is_arithmetic' cannot be
specialized: Users are not allowed to specialize this standard library entity
```

Clang 21 (Xcode on macOS 26) promotes that to an error. Suppressing it is enough.

## Tier 1 — everything except 2D prediction

```bash
brew install ffmpeg                     # scikit-video shells out to the ffmpeg binary
pyenv install 3.10.13                   # if needed
~/.pyenv/versions/3.10.13/bin/python -m venv .venv
.venv/bin/pip install "numpy<2" scipy "opencv-python<5" h5py pyyaml seaborn \
                      tqdm scikit-video av torch open3d
.venv/bin/pip install -e .
```

Run it:

```bash
.venv/bin/python run_local.py --project demos/pair_mac --steps 3d,video2d,video3d
```

Timings on an M-series laptop for 525 frames x 4 cameras x 2 animals:
triangulation + optimization ~6 s, 2D videos ~100 s, 3D composite video ~6 s.

Supply your own 2D points as `<project>/points_2d/original.h5` with datasets
`<cam>_points` shaped `(n_tracks, n_frames, n_bodyparts, 3)` — last axis `(x, y, score)` —
and `<cam>_bboxes` shaped `(n_tracks, n_frames, 4)`. Cameras are matched **alphabetically**
by name to the sorted `videos_raw/*.mp4`.

## Tier 2 — add MarmoPose's own 2D models

A prebuilt wheel for this machine is in `wheels/`:

```bash
python -m venv .venv2d && .venv2d/bin/pip install --upgrade pip
.venv2d/bin/pip install "numpy==1.26.4" "torch==2.1.2" "torchvision==0.16.2"
.venv2d/bin/pip install "setuptools<70" wheel cython
.venv2d/bin/pip install wheels/mmcv-2.1.0-cp310-cp310-macosx_26_0_arm64.whl
.venv2d/bin/pip install "mmdet==3.2.0"
.venv2d/bin/pip install --no-deps "mmpose==1.3.1"
.venv2d/bin/pip install --no-build-isolation xtcocotools
.venv2d/bin/pip install munkres json_tricks "opencv-python<5" h5py pyyaml seaborn \
                        tqdm scikit-video av
.venv2d/bin/pip install -e .
```

To rebuild the mmcv wheel from scratch:

```bash
curl -sLO https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-2.1.0.tar.gz
tar xzf mmcv-2.1.0.tar.gz && cd mmcv-2.1.0
MMCV_WITH_OPS=1 MAX_JOBS=8 \
  CFLAGS="-Wno-invalid-specialization -Wno-deprecated-literal-operator" \
  CXXFLAGS="$CFLAGS" \
  python setup.py bdist_wheel        # ~10 min
```

Gotchas that cost real time:
- `setuptools>=81` removed `pkg_resources`, which mmcv's `setup.py` imports — pin `<70`
  and pass `--no-build-isolation`.
- `torch 2.1.2` breaks on numpy 2.x (`_ARRAY_API not found`). Pin `numpy==1.26.4`.
  `opencv-python>=5` forces numpy>=2, so pin `opencv-python<5` too.
- `mmpose 1.3.1` depends on `chumpy`, which cannot build on modern Python (its `setup.py`
  imports `pip`). `chumpy` is only used for SMPL human-mesh tasks, so install mmpose with
  `--no-deps` and add `xtcocotools`, `munkres`, `json_tricks` yourself.
- `mmdeploy==1.3.1` is not on PyPI (GitHub only) and is only needed for TensorRT `.engine`
  models. `prediction.py` now imports it lazily, so its absence is fine.

Then:

```bash
.venv2d/bin/python -c "
from marmopose.config import Config
from marmopose.processing.prediction import Predictor
c = Config(config_path='configs/default.yaml', project='demos/_2dtest', n_tracks=2,
           dae_enable=False, do_optimize=False)
Predictor(c, batch_size=2).predict()"
```

2D inference costs about **0.86 s/frame/camera on CPU**. A 525-frame, 4-camera session is
roughly 30 minutes. `demos/_2dtest` holds 12-frame clips as a ~40 s smoke test.

## Known config traps

- `configs/default_6pt.yaml` ends with an empty `bodypart_distance_weak:`, which YAML reads
  as `None` and crashes `parse_constraints`. `run_local.py` patches this at runtime.
- `Config.DEFAULT_CONFIG` defines `optimization.enable`, but all code reads `do_optimize`.
  A YAML omitting `do_optimize` raises `KeyError`.
- `DEFAULT_CONFIG` defaults `dae_enable` to `True`; the shipped autoencoder only accepts
  16 keypoints, so leave it `false` for the 6-point configs.
- `Reconstructor3D` hardcodes `camera_params.json`, so `camera_params_fixed.json` from
  `FixedCalibrator` is never used unless you rename it.
