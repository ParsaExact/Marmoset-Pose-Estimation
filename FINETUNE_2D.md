# Getting 2D keypoint coverage from ~40% to ~84%

Everything below is measured on this machine with `tools/diagnose_2d.py` and
`tools/make_pseudo_labels.py`, not assumed.

## 1. Where the 40% actually goes

MarmoPose's headline "% valid keypoints" is the product of two independent things:
whether the detector boxed the animal at all, and whether the pose model was
confident about each keypoint inside that box. They need different fixes, so
`tools/diagnose_2d.py` reports them separately, with the keypoint threshold forced
to 0 so coverage can be swept rather than read at one arbitrary cut.

80 evenly-spaced frames per video:

| video | animals found | both present | median box | box as % frame | cover@0.5 | pose-only cover@0.5 |
|---|---|---|---|---|---|---|
| demo `bak-1-2` | 1.99 / 2 | 98.8% | 279 px | 3.76% | **76.1%** | 76.6% |
| demo `bak-3-2` | 2.00 / 2 | 100.0% | 277 px | 3.69% | 72.5% | 72.5% |
| SNUH `D01_16min` | 1.57 / 2 | 60.0% | 324 px | 6.75% | **40.7%** | 51.7% |
| SNUH `D02_16min` | 1.66 / 2 | 70.0% | 319 px | 6.54% | 31.7% | 38.1% |
| SNUH `D03_16min` | 1.62 / 2 | 70.0% | 318 px | 6.50% | 43.9% | 54.1% |

("pose-only" = coverage counting just the keypoints of animals that *were* detected.)

### Baseline on the real target data

`260528_test 2` (white + pink, session 2), the `edit_ver/1min` clips, all 5 cameras,
1440x1080 @ 30 fps, 100 frames each. **This is the number to beat.**

| camera | animals found | both present | median box | % of frame | cover@0.5 | pose-only@0.5 |
|---|---|---|---|---|---|---|
| D01 | 1.81 / 2 | 81.0% | 324 px | 6.74% | 66.2% | 73.1% |
| D02 | 1.65 / 2 | 67.0% | 251 px | 4.04% | 32.6% | 39.5% |
| D03 | 1.42 / 2 | 52.0% | 281 px | 5.07% | 36.6% | 51.5% |
| D04 | 1.73 / 2 | 73.0% | 306 px | 6.01% | 42.2% | 48.8% |
| D05 | 1.84 / 2 | 84.0% | 304 px | 5.92% | 38.5% | 41.8% |
| **mean** | 1.69 / 2 | 71.4% | | | **43.2%** | 50.9% |

The spread matters: D01 alone is already at 66.2%, so the cage and lens are not
inherently hostile -- the other four cameras are losing something D01 isn't. D02
images the animals smallest (4.04% of frame vs D01's 6.74%) and is worst on every
metric. Frames show heavy cage-mesh occlusion, which is the obvious candidate for
the low keypoint confidence.

**Resolution is not the problem — this was worth ruling out first.** The SNUH
animals occupy *more* pixels than the demo animals (324 px vs 279 px median box
side; 6.75% vs 3.76% of the frame), despite the lower 1440×1080 frame. A top-down
pose model's accuracy tracks how many pixels the animal covers, so if SNUH imaged
the animals smaller, no amount of fine-tuning would help. It doesn't. The gap is
pure domain shift — cage, lighting, viewpoint, lens — which is exactly what
fine-tuning fixes. That is the encouraging finding here.

The gap decomposes cleanly and multiplicatively (D01 vs `bak-1-2`):

```
D01   0.785 detection x 0.517 pose = 0.406   (measured 40.7%)
demo  0.995 detection x 0.766 pose = 0.762   (measured 76.1%)

fix detection only -> 0.995 x 0.517 = 51.4%   (+10.7 pts)
fix pose only      -> 0.785 x 0.766 = 60.1%   (+19.4 pts)
fix both           ->                 76.2%
```

So **detection is ~38% of the shortfall and the pose model ~62%**. Neither alone
reaches the target; both have to be fine-tuned. D02 is the worst camera on every
metric and should be looked at separately — it was also the camera with sync and
calibration problems.

### Accuracy, not just coverage: fine-tuning is genuinely required

Coverage cannot distinguish two very different failures that produce identical
numbers: the model being *wrong* on this domain (fine-tuning is the fix), or the model
being *right but under-confident* (its confidence is miscalibrated, and most of the
"missing" keypoints are recoverable by thresholding differently, with no training at
all). Before committing anyone to days of labelling, that had to be settled.

It can be, because real hand labels already exist on this exact footage. The DLC
project's `labeled-data/D0X_1min/` frames are **pixel-identical** (mean |diff| = 0.000)
to the `edit_ver/1min` clips, so its 250 labelled frames are usable ground truth --
which also answers the open question in `SUMMARY.md` section 6: these labels index
`edit_ver/1min` at 1440x1080, the same export we use.

`tools/eval_2d.py` matches DLC individuals to MarmoPose tracks by centroid and finds
keypoint correspondences by measuring cross-distance rather than trusting names:

| hand label | closest model keypoint | median |
|---|---|---|
| forehead_blaze | head | 8.3 px |
| left_ear_tuft_upper | leftear | 8.1 px |
| right_ear_tuft_upper | rightear | 8.5 px |

Error binned by the model's own predicted score, over 300 matched instances:

| score bin | n | median px | p90 | PCK@20px |
|---|---|---|---|---|
| 0.0 – 0.3 | 20 | **56.4** | 138.5 | 10.0% |
| 0.3 – 0.5 | 36 | 21.7 | 121.9 | 41.7% |
| 0.5 – 0.7 | 79 | 17.3 | 75.1 | 54.4% |
| 0.7 – 1.0 | 314 | **9.0** | 30.1 | 83.4% |

**Confidence is well calibrated — monotonic, and steeply so.** Low-confidence
predictions are genuinely wrong, not shy. So the coverage gap cannot be closed by
lowering the threshold; doing that would admit keypoints with 22–56 px median error.
Fine-tuning is required, and this is the evidence for it.

The corollary is equally useful: where the model *is* confident it is already accurate
(9 px median, 83% PCK@20px). The backbone is not broken on marmosets, it fails on this
cage's harder cases -- which is exactly the regime where fine-tuning beats retraining.

Caveat: these three keypoints are the head region, the easiest part of the animal.
Limbs are almost certainly worse, so treat this as a lower bound on the problem.

### Where the gap actually lives

Coverage@0.5 per bodypart, pooled over all 5 cameras:

| region | coverage |
|---|---|
| head / ears / neck | 54.8% |
| trunk and tail | 50.8% |
| **limbs** | **33.4%** |

Worst individually: rightfoot 25.4%, rightknee 30.7%, rightelbow 31.1%, righthand
32.5%. Best: leftear 57.0%, rightear 56.9%, tailbase 55.3%, head 55.3%.

The gap is a limb problem. That is where labelling effort will pay, and it is also the
hardest region to label through cage mesh -- worth knowing before starting rather than
after.

### One caveat on the target itself

"84%" is a *coverage at score >= 0.5* number, and coverage can be raised for free
by lowering the threshold — which improves nothing. The diagnostic therefore
reports @0.3/@0.5/@0.7. Real progress means coverage rising **at a fixed
threshold** while accuracy against held-out hand labels holds, so a small manual
validation set is required regardless of how the training labels are produced.

## 2. Decision: labels come from hand annotation

Fine-tuning and retraining are not alternatives to labelling -- they are what you do
*with* labels. So the only real question is where the labels come from, and there are
three candidates:

| source | verdict |
|---|---|
| **hand labels** | **chosen.** Trustworthy, and the only source that can also provide a held-out validation set |
| multi-view pseudo-labels | measured (section 3) -- +9.7 coverage points in this regime, 8.8 px median error, and it needs the whole calibration chain to be right first |
| the existing DLC facial labels | unusable for this: 10 *facial* keypoints, not the 16 body keypoints |

Retraining from scratch is worse than fine-tuning at this scale: it needs thousands of
images, MarmoPose's own training set is not shipped, and it discards a backbone that
already knows marmosets. The same labels go further as fine-tuning.

**Sizing.** The existing DLC project labelled 570 frames for 10 keypoints, so this is
a known quantity of work here. For 16 body keypoints, ~300-500 frames (600-1000 animal
instances) with ~20% held out should move the pose model substantially.

Two things cut the cost, both implemented:

* **Label the frames that teach the most.** `tools/select_frames_to_label.py` ranks by
  the current model's uncertainty rather than sampling uniformly -- but only 60% of the
  budget, because pure uncertainty sampling collapses onto motion blur and total
  occlusion, which a human cannot label either and which make an unrepresentative
  training set. The rest is uniform random. Frames are spaced >= 15 frames apart (30 fps),
  since consecutive frames cost double and teach once, and the budget is split evenly
  across cameras so the weak ones are actually represented.
* **Correct, don't annotate from scratch.** The same tool pre-fills each frame with the
  current predictions in DLC format, so labelling is fixing what is wrong -- typically
  several times faster. Keypoints the model missed entirely are left blank rather than
  anchored to a bad guess, since those are the cases most worth adding by hand.

### Workflow

```powershell
# 1. choose frames and pre-fill them (done: 400 frames, 61-86% pre-filled).
#    This also writes a ready DLC config.yaml, so no project setup is needed.
python tools/select_frames_to_label.py --project demos/snuh_t2 --n-frames 400

# 2. open datasets/to_label/config.yaml in DLC and CORRECT the pre-filled labels
#    (scorer "pretrained"). Prioritise limbs - that is where the gap is.

# 3. convert and fine-tune
python tools/dlc_to_coco.py --dlc-project <dlc project> --out datasets/snuh_hand
python tools/finetune.py configs/finetune_pose.py --cfg-options data_root=datasets/snuh_hand/
python tools/finetune.py configs/finetune_det.py  --cfg-options data_root=datasets/snuh_hand/

# 4. re-measure at the SAME threshold as the baseline
python tools/diagnose_2d.py --config configs/snu_charuco.yaml --frames 100 \
    --videos "demos/snuh_t2/videos_raw/*.mp4"
```

The whole chain is verified end-to-end on real SNUH images by round-tripping the
pre-filled predictions back through `dlc_to_coco.py` into a training run (230
iterations/epoch, pretrained checkpoint loaded, loss decreasing). So the pipeline is
not waiting on anything except the labels themselves.

## 3. Pseudo-labels: measured, and set aside

We need 16-keypoint body labels on SNUH footage. The DLC project can't supply them
— it has 10 *facial* keypoints. Hand-labelling thousands of frames is the obvious
route and the one to avoid.

Instead, exploit the rig: a pose model fails per-view and roughly independently, so
a keypoint lost to occlusion or blur in one camera is often clean in two others.
Calibration ties the views together, so any keypoint confident in >= 2 views can be
triangulated, and the resulting 3D point reprojects into *every* view — including
the ones where the appearance model gave nothing. Geometry supplies what appearance
missed. `tools/make_pseudo_labels.py` implements this.

Two quality gates matter more than coverage, because a wrong label is worse than a
missing one:

- `--min-views` — confident views required before a 3D point is accepted.
- `--max-reproj-error` — reject a 3D point that doesn't agree with the very
  observations that produced it. Without this the label set carries a long tail of
  gross errors: two views are enough to produce *a* point, but a cross-view
  identity swap still produces one, just in the wrong place.

The label set is then the **union** — keep confident observations as-is, use
reprojection only to fill empty slots.

### Measured yield (demo rig, 4 cameras, 15 px gate)

| source threshold | model only | + geometry | gain | of missing recovered | geometry label err (median / p90) |
|---|---|---|---|---|---|
| 0.50 | 74.3% | **83.6%** | +9.3 | 36.2% | 8.77 px / 17.71 px |
| 0.70 | 53.8% | 64.5% | +10.7 | 23.1% | 8.13 px / 16.85 px |
| 0.80 | 41.7% | 51.4% | +9.7 | 16.6% | 7.87 px / 16.38 px |

Label error is measured honestly: where a camera *did* observe a keypoint, its
observation is compared against the reprojection of the 3D point. That error is the
proxy for the labels we cannot check.

**Read this result carefully.** At the 0.80 row — per-view coverage 41.7%, i.e. the
SNUH regime — geometry adds only ~10 points, to 51%. Multi-view pseudo-labelling
alone does **not** get you from 40% to 84%, and any claim that it does is wrong.

Two reasons it still works, and one reason this demo understates it:

1. **The label set does not need 84% coverage for the model to reach 84%.** These
   labels are training fuel, not the metric. Fine-tuning on ~50% coverage of
   accurate in-domain labels teaches the model this cage's appearance; its own
   confidence then rises on the keypoints that had no label at all.
2. **It iterates.** Round 2 runs the fine-tuned model, whose higher per-view
   coverage makes more points triangulable, which yields more and better labels.
   This is self-training and it compounds — that is the actual path to 84%.
3. **This demo rig is a pessimistic testbed.** Its own reprojection error is
   ~14.5 px, so a 15 px gate sits at the noise floor and throws away half the 3D
   points. The SNUH rig is calibrated to **1.82 px** with 7.51 px 3D reprojection
   (per `results_for_meeting/SUMMARY.md`) — roughly 2x better, so SNUH can run a
   tighter gate, keep more points, and get more accurate labels than the table
   above.

### Prerequisite: cross-view identity

This is a real dependency, not a detail. Pseudo-labelling requires knowing which
detection in camera 1 is the same animal as which in camera 3. MarmoPose's dye
classes don't transfer to this cage (measured: both animals found in only 51% of
frames even on the blue+white session), so identity must come from
`Reconstructor3D.triangulate(reassign_id=True)` epipolar grouping. I found this the
hard way: running the yield measurement with the class-agnostic config, whose track
IDs are per-camera positional, triangulated animal A against animal B and produced
21 px median / 231 px p90 label error — unusable. With consistent IDs the same
measurement gives 8.8 px / 17.7 px.

SUMMARY.md notes epipolar grouping "did not help" 3D reconstruction quality. That
is compatible with it being *necessary* here: it wasn't the bottleneck for 3D, but
pseudo-labelling cannot proceed without it. **Verify grouping accuracy on SNUH
before trusting any label set.**

## 3. The plan

| stage | action | expected | cost |
|---|---|---|---|
| 0 | Hand-label 150–300 frames across all 5 cameras for **validation only** | trustworthy metric | ~1 day |
| 1 | Verify cross-view identity grouping on SNUH | unblocks stage 2 | hours |
| 2 | Generate pseudo-labels, tight gate (~8 px) | ~50–65% coverage labels @ ~4–5 px | minutes on the 5090 |
| 3 | Fine-tune RTMPose (`configs/finetune_pose.py`) | pose coverage 52% → 75%+ | ~30 min |
| 4 | Fine-tune RTMDet on the same boxes | detection 78% → 95%+ | ~1 h |
| 5 | Re-run stages 2–4 with the improved model (2–3 rounds) | compounding to ~80%+ | a few hours |
| 6 | Cheap extras: flip-test TTA (already on), larger input size, lower bbox threshold | few points | minutes |

Stage 0 is not optional. Without held-out hand labels, evaluation is circular: the
smoke test below scores **AP 0.954** against the pseudo-labels, which the model
largely generated itself. That number means the plumbing works and nothing about
accuracy.

## 4. What is already built and verified

- `tools/diagnose_2d.py` — the coverage decomposition above.
- `tools/eval_2d.py` — **accuracy** against hand labels: discovers keypoint
  correspondence by cross-distance, matches animals by centroid, and bins error by
  predicted confidence. This is the tool that settles whether a change is real; run it
  again after fine-tuning.
- `tools/select_frames_to_label.py` — uncertainty + random frame selection, exports a
  DLC `labeled-data/` tree pre-filled with the current predictions. **Run: 400 frames
  across 5 cameras, 61–86% of keypoints pre-filled**, in `datasets/to_label/`.
- `tools/dlc_to_coco.py` — hand-labelled DLC project → COCO for mmpose. Boxes come
  from the labelled keypoint extent; unlabelled keypoints get visibility 0 so the loss
  ignores them; the split is by image and grouped by camera, so the two animals in one
  frame never straddle train/val and every camera appears in both halves.
- `configs/finetune_pose.py` — RTMPose fine-tune config. Uses mmpose's generic
  `CocoDataset` plus `models/pose_model_deployed/data_meta.py` for the metainfo,
  because the shipped config names `MarmosetDataset`, which does not exist in
  mmpose 1.3.1 and would fail immediately.
- `configs/finetune_det.py` — RTMDet fine-tune config; keeps `num_classes=2` and writes
  every label as class 0 so the pretrained head loads cleanly, and drops Mosaic/MixUp.
- `tools/finetune.py` — one Runner entry point for both; installs the mmcv shim and the
  `weights_only` workaround, swaps SyncBN for BN, and resolves repo-relative paths.
- `tools/calibrate_snuh.py` — sync-corrected calibration (section 4b).
- `tools/make_pseudo_labels.py` — the set-aside path, kept because the measurement is
  the evidence for not using it.

Training is confirmed on real SNUH images, by round-tripping the pre-filled predictions
through `dlc_to_coco.py` into a run:

```
Load checkpoint from models/pose_model/best.pth
Epoch(train) [1][1/230]  lr: 2.0e-07  loss: 0.008563  acc_pose: 0.531250
```

~0.25-0.34 s/iter at batch 2; a 40-epoch fine-tune is well under an hour. Nothing in
the pipeline is now waiting on anything except the labels.

Note the earlier demo smoke test scored `coco/AP 0.954` — but that was against
pseudo-labels the model largely produced itself, so it measured plumbing, not accuracy.
The same trap applies to any evaluation without hand-labelled held-out frames.

## 4b. Calibrating the SNUH rig (and why it read 40 px first)

`FixedCalibrator` on `edit_ver/calibration` (5 x ~3,577 ChArUco frames, ~45 min of
board detection) lands at **40.00 px** with focals [887, 1098, 1134, 1103, 1163].
That is not a geometry failure -- it reproduces the "39.69 px" intermediate stage in
`results_for_meeting/SUMMARY.md` almost exactly, and the per-pair breakdown says why:

| pairs | reprojection |
|---|---|
| among {D01, D03, D04} | 8.3 – 9.5 px |
| any pair touching **D02** | 12.8 – 18.5 px |
| any pair touching **D05** | 10.8 – 18.5 px |

Exactly the signature SUMMARY.md describes: D02 and D05 are shifted in time, so the
board each reports is the board at a different instant, and bundle adjustment cannot
reconcile that. SUMMARY.md measured D02 +27, D05 -27 frames (~0.9 s at 30 fps).

`tools/calibrate_snuh.py` applies per-camera frame offsets to the *cached* detections
(`detected_boards.pickle`), so trying an offset set costs one bundle adjustment
instead of another 45 minutes of detection. Intrinsics are recomputed per camera and
are unaffected by inter-camera timing.

Applying `D02 +27, D05 -27` gives **1.82 px** (focals [887, 1098, 1134, 1103, 1163]) --
reproducing the earlier session's figure exactly and confirming those offsets hold for
this export too. Saved as `demos/snuh_t2/calibration/camera_params.json`
(`Reconstructor3D` hardcodes that filename), with `camera_params_sync.json` kept as the
named copy and `camera_params_fixed.json` the uncorrected 40 px version.

`--probe-sync N` recovers the offsets without any bundle adjustment: if two cameras are
synchronised, the relative board rotation `R_c(f) @ R_ref(f)^T` is constant across every
shared frame, and a time shift scatters it. That is one pose-estimation pass instead of a
bundle adjustment per candidate, which is what makes searching feasible -- a BA-per-
candidate search at ~30 min each is not.

This is the third independent place camera sync has bitten this project. It is worth
fixing at the recording end (an LED flash event, or an NVR export that is already
aligned) rather than re-deriving offsets for every export.

## 5. Status and what is needed

Resolved: the videos, and the calibration (1.82 px, done here).

**Blocking: the labels.** `datasets/to_label/` holds 400 frames across all 5 cameras
with predictions pre-filled. The remaining step that only you can do is correcting them
in DLC. Then `dlc_to_coco.py` → `finetune.py` → `diagnose_2d.py` gives a measured
before/after against the 43.2% baseline.

Still open, and it matters for 3D rather than 2D:

- The **ChArUco square size in mm** is still a 25 mm placeholder. Every absolute
  distance scales linearly with it, including the `bodypart_distance` constraints the
  3D optimizer enforces.
- The **animal clips may have their own frame offsets**, independent of the calibration
  videos'. `make_pseudo_labels.py --sync-search N` measures them from reprojection
  residual; worth running before any 3D reconstruction on this session.
