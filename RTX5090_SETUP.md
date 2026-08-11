# Running MarmoPose on an RTX 5090 (Windows, CUDA 12.8)

Verified on Windows 11 / RTX 5090 (32 GB) / driver 591.86 / Python 3.10 in the
`marmopose-new` conda env. The full pipeline runs on the GPU, including
MarmoPose's own 2D detection and pose models.

## The problem

RTX 5090 is Blackwell, compute capability **sm_120**. The shipped `marmopose`
env has `torch 2.4.1+cu121`, which was built for sm_50…sm_90:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the
current PyTorch installation.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` still returns `True`, so this surfaces as a crash on
the first real kernel launch rather than a clean "no GPU" message.

sm_120 needs CUDA >= 12.8, which means **torch >= 2.7**. That is the whole fix —
except that it breaks the OpenMMLab stack in two places.

### Why you can't just install mmcv

`mmdet` / `mmpose` need `mmcv`, and OpenMMLab publishes no `mmcv` wheel for
cu128. Building from source doesn't work here either: CUDA 12.8's `nvcc` refuses
this machine's only host compiler (Visual Studio 2026, `_MSC_VER` 195x).

It turns out not to matter. MarmoPose runs **RTMDet** for detection and
**RTMPose** (SimCC/RTMCC) for keypoints, and between them the only compiled mmcv
operator ever *executed* is non-maximum suppression. Everything else that
`mmcv.ops` exports — deformable conv, RoIAlign, CARAFE, corner pooling — belongs
to architectures this project never instantiates, and is only touched because
`mmcv/ops/__init__.py` imports the whole package eagerly.

So we install `mmcv-lite` (all the `mmcv.ops` Python sources, no compiled
`mmcv._ext`) and supply the missing extension ourselves:

- `marmopose/utils/mmcv_ext_shim.py` registers a fake `mmcv._ext` that implements
  `nms`, `softnms` and `nms_match`, and returns a raising placeholder for
  anything else. NMS routes through **torchvision**'s kernel, which ships
  Blackwell binaries; mmcv's own NMS is derived from torchvision's, so the
  semantics match. Detection stays entirely on the GPU.
- The shim is a no-op when a real compiled `mmcv` is present, so an environment
  with a full mmcv build is unaffected.

### Why checkpoints fail to load

PyTorch 2.6 flipped `torch.load`'s `weights_only` default to `True`. mmengine
0.10.7 predates that and can't pass the argument, so every `.pth` raises
`UnpicklingError`. Allowlisting types via `add_safe_globals` doesn't rescue it:
these checkpoints carry a `meta` block whose reconstruction needs the builtin
`getattr`, and allowlisting *that* would make `weights_only` protect nothing.

`marmopose/utils/torch_compat.py` instead restores the pre-2.6 behaviour in a
narrow `legacy_torch_load()` context, wrapped only around the calls that load
MarmoPose's own bundled checkpoints.

### Two rendering bugs this shook out

Neither is GPU-specific, but both block the composite 3D video on Windows:

- **Odd frame height.** Open3D's offscreen window is capped by the desktop work
  area, so `create_window(height=1080)` actually returns 1061 rows here. libx264
  with `yuv420p` cannot encode odd dimensions, so ffmpeg exits before the first
  frame and skvideo reports the closed pipe as a bare
  `OSError: [Errno 22] Invalid argument` with empty stderr. `pad_to_even()` in
  `display_3d.py` now pads the frame.
- **Deadlock that hides the error.** The frame-reader threads in
  `MultiVideoCapture` used a blocking `queue.put()` and only checked
  `stop_event` at the top of the loop, so once the queue filled they could never
  be stopped. Being non-daemon, they kept the interpreter alive at shutdown — the
  process hung forever with the traceback still buffered and unprinted. Reads now
  use an interruptible `put_frame()`, and `generate_video_3d` releases the writer,
  the Open3D window and the reader threads in a `finally`.

## Install

```powershell
conda create -n marmopose-new python=3.10
conda activate marmopose-new

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "numpy==1.26.4"                 # see note below
pip install mmengine==0.10.7 "mmcv-lite==2.1.0" "mmdet==3.2.0"
pip install --no-deps "mmpose==1.3.1"
pip install xtcocotools munkres json_tricks
pip install h5py seaborn scikit-video av "opencv-python<5" open3d tqdm pyyaml
pip install -e . --no-deps
```

Pin `numpy<2`: the prebuilt `xtcocotools` 1.14.3 wheel is compiled against numpy
1.x and fails under numpy 2 with `numpy.dtype size changed`, and `scikit-video`
still calls `ndarray.tostring()`, which numpy 2 removed. torch 2.11 works fine
with numpy 1.26.4. `mmpose` needs `--no-deps` because it pulls `chumpy`, which
cannot build on modern Python and is only used for SMPL human-mesh tasks.

`ffmpeg` must be on `PATH` — `scikit-video` shells out to the binary.

**Do not install `albumentations`.** It pulls `numpy>=2` and `opencv-python-headless`,
which breaks `xtcocotools` and `scikit-video`, and having both `opencv-python` and
`opencv-python-headless` installed leaves a half-deleted `cv2` when either is
removed (`module 'cv2' has no attribute '__version__'` — fix with
`pip install --force-reinstall --no-deps "opencv-python==4.11.0.86"`). mmpose
1.3.1's `Albumentation` wrapper targets albumentations 1.x anyway, so
`configs/finetune_pose.py` drops that transform instead.

Check the GPU is actually usable:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability())"
# must contain sm_120 and print (12, 0)
```

## Run

```powershell
python -c "
from marmopose.config import Config
from marmopose.processing.prediction import Predictor
c = Config(config_path='configs/default.yaml', project='demos/pair',
           det_model='models/detection_model', pose_model='models/pose_model',
           dae_model='models/dae_model', n_tracks=2, dae_enable=False, do_optimize=False)
Predictor(c, batch_size=8).predict()"

python run_local.py --project demos/pair --steps 3d,video2d,video3d
```

## Measured on this machine

525 frames x 4 cameras x 2 animals (`demos/pair`):

| stage | time | note |
|---|---|---|
| 2D detection + pose | 43.6 s | 0.021 s/frame/camera, ~50 fps, 1.4 GB VRAM |
| triangulation | 3.3 s | ~158 frames/s |
| optimization | 3.4 s | both tracks |
| 2D labeled videos | ~110 s | CPU-bound rendering |
| 3D composite video | 23 s | 2844x1062, 525 frames |

The macOS CPU baseline in `LOCAL_SETUP.md` is 0.86 s/frame/camera, so 2D
inference is roughly **40x faster** here — a 4-camera session drops from about
30 minutes to 44 seconds.

## Correctness of the NMS shim

2D output was compared against the reference `demos/pair/points_2d/original.h5`
shipped with the repo (produced with a full compiled mmcv):

- 49,897 keypoints compared, **median difference 0.010 px**, p99 0.6 px
- 18 keypoints (0.036%) differ by more than 5 px — near-threshold detections
- 3D triangulation matches to a **median of 0.012 mm**

Differences at that scale are floating-point nondeterminism, not a behavioural
change. (`points_3d/optimized.h5` differs more, at a ~3.7 mm median; the
optimizer is a global nonlinear least-squares whose convergence amplifies tiny
input differences, and the shipped reference was generated with different
optimization settings.)

## Note on the older `marmopose` env

The py3.8 / `torch 2.4.1+cu121` env still exists and still has a fully compiled
`mmcv 2.1.0`. It cannot drive an RTX 5090 — keep it only for reference or for
running on an older GPU.
