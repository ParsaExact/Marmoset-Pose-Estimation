# MarmoPose on the SNUH data — results as of 2026-07-31

All numbers below are measured on this data, not taken from the paper.

## 1. The whole pipeline now runs locally (no CUDA)

Calibration, 2D detection + pose, 3D reconstruction, optimization and both video
renderers run on a MacBook CPU. The blocker was never CUDA: `mmcv` failed to build
because PyTorch's own header specialises `std::is_arithmetic`, which clang 21 rejects.
Suppressing that one warning built it cleanly. **Worth retrying on the RTX 5090 box —
its failure looks like the same class of problem.**

2D inference costs ~0.9-1.2 s/frame/camera on CPU (~30 min per 10 s five-camera clip).

## 2. Calibration: unusable -> working

| | reprojection error |
|---|---|
| stock MarmoPose `Calibrator` | **crashed** (assertion in `cv2.initCameraMatrix2D`) |
| after fixing partial-detection handling | 92.76 px |
| + proper ChArUco detection | 39.69 px |
| + correcting camera time offsets | **1.82 px** |

For reference, MarmoPose's own published demo data calibrates at 1.65 px, so this rig
is now equivalent.

Three separate root causes, all found by measurement:
- **The board is ChArUco, but the code treated it as a plain checkerboard**, discarding the
  marker IDs. ~7% of detections had a 180-degree flipped corner ordering as a result.
- **`detect_image` ignored `pattern_was_found`**, so failed detections returned partial
  corners that were then mislabelled as board positions 0..n-1. This both crashed
  calibration and corrupted intrinsics (D04 focal came out 489 instead of ~2054).
- **Two cameras were ~0.9 s out of sync in the calibration export** (D02 +27 frames,
  D05 -27). Located by pairwise stereo calibration: every pair among {D01, D03, D04}
  gave 2.7-4.7 px while every pair touching D02 or D05 gave 30-46 px.

Board geometry identified programmatically: **10x7 squares (9x6 interior corners),
ArUco DICT_4X4_50**. Square size in mm is still a placeholder (25 mm) — **all absolute
distances scale linearly with the true value, so this needs confirming.**

## 3. 2D: the shipped code discards one of the two animals

MarmoPose's detector has exactly two classes, `white_head_marmoset` and
`blue_head_marmoset` — **identity is the ear dye colour** — and it keeps at most one box
per class. Your session-2 animals are white and **pink**, a colour the model has never
seen, so both land on the same class and one is silently dropped.

| session 2 (white + pink) | shipped | class-agnostic | + temporal tracking |
|---|---|---|---|
| 2D keypoints valid | 42.3% | 62.0% | 62.0% |
| frames with both animals | 28.0% | 93.3% | 93.3% |
| cameras seeing both (of 5) | **1.40** | **4.67** | 4.67 |
| identity swap rate | n/a | 5.2% | **0.0%** |

The 1.40 figure is the critical one: triangulation needs two cameras, so **the second
animal could not be reconstructed in 3D at all**.

**The pretrained dye classes fail on session 1 too**, even though those animals are
blue + white — exactly the two trained classes:

| session 1 (blue + white) | dye classes | class-agnostic |
|---|---|---|
| 2D keypoints valid | 36.4% | 40.4% |
| frames with both animals | 51.0% | **76.3%** |

So the shipped identity classifier does not transfer to this cage at all. Fine-tuning is
required if identity is to come from the detector.

## 4. 3D reconstruction

| | reprojection error |
|---|---|
| session 2 (pink), 30 frames | 24.8 px |
| session 1 (blue+white), 120 frames | 9.94 px |
| session 1 after correcting frame offsets (D02 -1, D04 -2, D05 +3) | **7.51 px** |
| MarmoPose's own demo data | ~14 px |
| calibration floor on this rig | 1.82 px |

**Session 1 at 7.51 px is about half the error of MarmoPose's own published demo data.**
Caveat: 2D coverage here is ~40% versus their ~84%, so part of that is a selection effect —
only confident keypoints survive, and confident keypoints reproject well.

Two things were tested and ruled out as bottlenecks:
- **Identity is not limiting 3D.** Colour anchoring on the pink dye (implemented and
  working) left 2D unchanged and made 3D slightly worse; cross-view epipolar ID grouping
  also did not help.
- **Geometry is not limiting 3D** — calibration is at 1.82 px.

The remaining ~4x gap over the calibration floor is **2D keypoint accuracy**: an
out-of-domain model on an unfamiliar cage, lighting and lens set. That is what
fine-tuning should target, and it matters more than identity.

## 5. Synchronisation is a recording-side problem

Camera timing has needed correcting in every export examined:

| export | offsets found |
|---|---|
| calibration videos | D02 +27, D05 -27 frames (~0.9 s) |
| `marmoset_long` | D02 starts 91 s before the others |
| 10 s animal clips | spread of ~6 frames (~200 ms) |
| session-1 animal clip | D02 -1, D04 -2, D05 +3 frames |

Total frame counts also differ per camera (Cam 4 is +1721 frames vs Cam 1), so a single
offset may not hold across a long recording.

The team's own plan — an LED flash sync event, and checking whether the NVR can export
already-aligned — would remove this whole class of problem. Right now every export needs
its offsets re-derived.

## 6. About the DLC label files

`260528_test 2/Labels/` contains **DLC model predictions, not hand labels**
(`snapshot_best-950`, then ellipse-tracked, filtered and gap-interpolated). 14 clips,
5,832 frames, **cam1 only**, and **only 4 head keypoints**. Individuals are
`marm_white` / `marm_blue`, i.e. session 1.

They can't serve as 3D ground truth (single camera) or train a 16-keypoint body model
(head only). I could not match them to our footage: the coordinates reach x=2199, y=1766,
so that work used a **higher-resolution stream** than our 1440x1080 NVR export.

**Open question for whoever ran DeepLabCut: which video file do the `20m55s` clip offsets
index, and at what resolution?** With that, the comparison takes minutes.

Their `interpolation_log.csv` bounds each gap by its average step distance and records
whether it was filled — notably more careful than MarmoPose's own unbounded interpolation,
which fills any gap and writes 0 (the world origin) for an all-missing track.

## 7. Next steps

1. Confirm the ChArUco square size in mm (rescales every distance).
2. Get the 5090 building — try the `-Wno-invalid-specialization` flag. This turns
   fine-tuning from a multi-day CPU job into an iteration loop.
3. Auto-generate training labels: class-agnostic detection finds both animals in 93% of
   frames and the pink dye is visible in 71%, so labels can be minted without manual
   annotation.
4. Fine-tune for **keypoint accuracy**, not primarily identity.
5. Adopt a recording-time sync signal.

---

## Video files in this folder

All videos below are **SNUH data**.

| file | length | content |
|---|---|---|
| `01_2D_detection_fix_before_after.mp4` | 5 s | Left (red) = shipped code: only one animal detected. Right (green) = fixed: both animals boxed and posed. Session 2. |
| `02_2D_identity_swap_fix.mp4` | 5 s | Identity colours flickering (score-rank ordering) vs stable (temporal matching). Session 2. |
| `03_all5_cameras_2D.mp4` | 5 s | All five cameras, session 2 (white + pink). |
| `04_session1_2D_all5cams.mp4` | 4 s | All five cameras, session 1 (blue + white), time-aligned. |
| `05_session1_3D_composite.mp4` | 12 s | **3D reconstruction of our animals** + all five camera views. Slowed 3x for viewing; the underlying clip is 4 s of real footage. |
| `08_session1_long_2D.mp4` | 20 s | All five cameras, session 1 — the long version. |
| `09_session1_long_3D.mp4` | 20 s | **3D reconstruction, 20 s continuous.** The headline result. |

The 3D videos were rendered with `tools/render3d_composite.py`, written for this project.
MarmoPose's own `Visualizer3D` cannot produce them: it uses Open3D's interactive window,
which blocks indefinitely without a GUI session, and its composite layout only places up to
4 cameras -- a 5th is written outside the canvas. The replacement is headless and handles any
camera count.
