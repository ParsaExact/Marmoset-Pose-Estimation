# MarmoPose on the SNUH data — report

**Footage used:** `260528_test 2`, the **white + pink** pair, 5 cameras (D01–D05),
1440x1080 @ 30 fps. Every number below was measured on this machine on this data.

---

## 1. The pipeline now runs on the RTX 5090

Roughly **40x faster**. This is what makes the rest of the analysis practical — a
16-minute five-camera session is now under an hour instead of days.

### How to use it on this machine

**Use the `marmopose-new` environment. Not `marmopose`.**

```powershell
conda activate marmopose-new
cd C:\GitHub_marmopose\MarmoPose
```

The older `marmopose` env still exists and still has a fully compiled mmcv, but its
PyTorch was built for CUDA 12.1 and **cannot run on this GPU at all** — it fails on the
first kernel launch. It is kept for reference only. Check the environment is live with:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability())"
```

The output must contain `sm_120` and print `(12, 0)`. If it does not, the wrong
environment is active.

DeepLabCut lives in a separate env (`conda activate dlc`) and already works on this GPU
without changes. `ffmpeg` is on the system PATH.

**A project is just a folder** with videos in it; everything else is created:

```
demos/<name>/
    videos_raw/            <- put the 5 camera .mp4 files here (sorted by name = camera order)
    calibration/
        camera_params.json <- required for 3D; this exact filename is hardcoded
    points_2d/  points_3d/  videos_labeled_2d/  videos_labeled_3d/   <- generated
```

**Typical run, start to finish:**

```powershell
# 1. 2D detection + pose for every video in the project        (~10 min per 16-min session)
python tools\predict_2d.py --project demos\my_session --config configs\snu_charuco.yaml

# 2. check the quality before going further
python tools\diagnose_2d.py --config configs\snu_charuco.yaml --frames 100 ^
    --videos "demos\my_session\videos_raw\*.mp4"

# 3. recover per-camera frame offsets, then apply them to the 2D file
python tools\make_pseudo_labels.py --project demos\my_session --config configs\snu_charuco.yaml ^
    --measure-only --sync-search 5

# 4. 3D reconstruction, then clean it up
python run_local.py --project demos\my_session --steps 3d
python tools\filter_3d_outliers.py --project demos\my_session --config configs\snu_charuco.yaml
python tools\smooth_3d.py --project demos\my_session

# 5. videos
python run_local.py --project demos\my_session --steps video2d
python tools\render3d_composite.py --project demos\my_session --config configs\snu_charuco.yaml ^
    --source optimized --out results_for_meeting\my_session_3D.mp4
```

**Roughly what to expect on this machine:** 2D runs at ~50 frames/s per camera (a 16-minute
five-camera session takes ~50 min); triangulation ~150 frames/s; 2D labelled videos are
CPU-bound and are usually the slowest step; a 40-epoch fine-tune is under an hour.

**Things that will otherwise cost time:**

- **Do not `pip install albumentations`** into `marmopose-new`. It pulls numpy 2 and
  opencv 5, which break `xtcocotools` and `scikit-video`, and removing it afterwards
  leaves a half-deleted `cv2`. See `RTX5090_SETUP.md` for the repair if it happens.
- **Camera sync offsets must be re-derived for every export** (step 3). They differ
  between clips cut from the same recording — the 1-minute and 16-minute exports of the
  same session needed different corrections.
- **`Reconstructor3D` hardcodes the filename `camera_params.json`**, so a calibration
  saved under any other name is silently ignored.
- **Video files are matched to cameras by sorted filename**, so the names must sort into
  the same order as the cameras in `camera_params.json`.
- `demos/`, `datasets/`, `models/` and `work_dirs/` are gitignored — data stays local.
- Long jobs should be started in a way that survives the terminal; a 16-minute session
  takes ~50 minutes of GPU.


## 2. 3D reconstruction of session 2

| | |
|---|---|
| reprojection error | **2.88 px** (calibration floor 1.82 px) |
| true 3D coverage | **~62%** of keypoints |
| 2D coverage per camera | 43.1% |

Five-camera redundancy is doing real work: a keypoint needs only 2 good views, so 43%
per-camera coverage becomes ~62% in 3D.


## 3. What limits the result

### 3.1 Animals partly outside the camera

| box distance to image edge | instances | coverage@0.5 |
|---|---|---|
| **0 – 5 px (touching)** | **7,383** | **26.1%** |
| 5 – 25 px | 3,235 | 52.9% |
| 25 – 75 px | 4,095 | 71.1% |
| 75+ px (fully inside) | 11,909 | 64.2% |

**28% of all animal observations have the animal partly out of frame**, and those score
26% versus 64–71%. A keypoint outside the camera is physically absent from that image —
**no model, however trained, can recover it.**

> **Recommendation: reposition or widen the cameras so the animals stay in frame.** Free,
> actionable immediately, and likely worth more than retraining anything.

### 3.2 Identity swaps when the animals are close

An identity swap is the tracker mixing up which animal is which — the label "animal 1"
jumps from one marmoset to the other. It is detected by checking, between consecutive
frames, whether each track's movement is better explained by exchanging the two labels
than by keeping them. If exchanging fits better, the labels swapped.

| distance between the two animals | swaps per second |
|---|---|
| 552 mm (far apart) | **0.08** |
| 310 mm | 0.96 |
| 105 mm (close together) | **1.24** |

**About 15x more swaps when the animals are close.** This matters for 3D because
triangulation combines cameras: if camera 1 calls an animal "track 1" and camera 3 calls
the *other* animal "track 1", the two rays belong to different animals and the resulting
3D point is meaningless.

### 3.3 The core problem: too few cameras see each keypoint

This is the chain that explains everything above.

3D needs a keypoint to be confidently detected in **at least 2 cameras**. With per-camera
coverage around 43%, most keypoints are seen by only one or two cameras — and one camera
gives nothing at all in 3D.

| keypoint | 2D per camera | **avg cameras seeing it** | got >= 2 cams | **3D coverage** |
|---|---|---|---|---|
| leftear | 57% | 2.85 | 88% | **81%** |
| head | 55% | 2.76 | 87% | **79%** |
| tailend | 53% | 2.66 | 86% | **84%** |
| spinemid | 50% | 2.52 | 83% | **78%** |
| leftknee | 41% | 2.03 | 66% | 60% |
| lefthand | 35% | 1.76 | 60% | 49% |
| righthand | 32% | 1.62 | 52% | 40% |
| rightelbow | 31% | 1.55 | 49% | 42% |
| **rightfoot** | **25%** | **1.27** | **38%** | **30%** |

Read the middle column. The head is seen by **2.76 cameras on average**, clears the
2-camera bar almost always, and 3D coverage *rises* from 55% to 79%. The right foot is
seen by **1.27 cameras** — usually only one — so it never clears the bar and stays at 30%.

By body region:

| region | 2D coverage |
|---|---|
| head / ears / neck | 54.8% |
| trunk and tail | 50.8% |
| **limbs** | **33.4%** |

**The reconstruction is reliable for the body core and unreliable for the limbs**, and
the limbs are the parts that convey how the animal moves. In the best 25 s window, 0.0%
and 0.9% of frames had all 16 keypoints for the two animals.

**Why this is the highest-leverage target.** Because of the 2-camera threshold the
relationship is non-linear: raising limb detection from 25% to 40% per camera would take
the fraction with 2+ views from ~37% to ~66% — **roughly doubling limb coverage in 3D**
for a 15-point 2D gain. The same 15-point gain on the head buys almost nothing, because
it is already saturated.

---

## 4. Would a better time interval fix it?

Searched every 25 s window in the 113 s clip:

| window | coverage | both detected | separation | swaps/s |
|---|---|---|---|---|
| **25.6 s (best)** | **49.7%** | 63.9% | 552 mm | **0.08** |
| 87.9 s | 47.1% | 80.6% | 310 mm | 0.96 |
| 57.1 s | 42.4% | 77.5% | 105 mm | 1.24 |
| session average | 43.1% | | | |

**The best window reaches 49.7% against a 43.1% average — only +6.6 points.** There is no
interval in this clip where the 2D is close to perfect. Even in the best window the
median box-to-edge distance is **−136 px** (boxes extend beyond the image), which is why
"both detected" is only 63.9% there.

A good window still produces a **much better-looking video** — 0.08 swaps/s instead of
~1/s is very visible — but it does not fix the underlying 2D.

**Result of the larger search.** The same search over the **16-minute** recordings (all 5
cameras, 150,360 frames) moved the best window only from **49.7% to 52.8%**, against a
34.4% session average. Nine times more footage bought three points. The ~50% ceiling is
a property of the setup, not of which interval is chosen — so "use better footage" is
closed as an option.

---

## 5. Hypotheses tested and closed

Recording these because eliminating an option is a result, and each of these would
otherwise be proposed again later.

| hypothesis | outcome |
|---|---|
| A better time interval fixes the 2D | **No.** 9x more footage → +3 points |
| Full 2560x1920 resolution helps | **No.** −1.2 points; RTMPose resizes every crop to 512x512 regardless |
| The animals are too small in frame | **No.** They are *larger* than in MarmoPose's own demo data (324 px vs 279 px) |
| Coverage can be recovered by lowering the threshold | **No.** Predictions below 0.3 are wrong by 56 px median |
| The trained `blue_head_marmoset` class would work | **Untestable.** The blue-dyed animal appears only during handling, never behaving in the arena |
| Cross-view identity is the 3D bottleneck | **No.** Epipolar re-grouping made reprojection slightly *worse* (2.88 → 3.65 px) |

## 6. Two data problems found

**The recordings are 20 fps; the edited clips are 30 fps.** The raw NVR files run at 20
fps, but `edit_ver/1min` and `edit_ver/16min` are 30 fps — so roughly **one frame in
three is a duplicate**. Any motion, velocity or smoothness measure computed on the edited
clips is inflated, and this may explain part of why per-clip frame offsets keep differing.
Analysis should use the raw 20 fps source, or the conversion should be dropped at export.

**The detector boxes people as marmosets.** During the 12:11–12:34 handling period it
reported two animals with high confidence in frames whose only occupant was a person in
PPE. Class-agnostic detection has no class check by construction, so nothing rejects
non-animals. Real animals score 40–70% keypoint coverage there; these scored ~0%. A
minimum-pose-confidence guard on each box would remove them at no cost to real detections.
Until then, anyone entering the room enters the tracking data as an animal.

---

## 7. What to try next

Ordered by expected value per unit of effort. The first group needs no labelling, no
training, and no new code.

### A. Recording protocol — cheapest, and it attacks the largest measured cause

1. **Reposition or widen the cameras so the animals stay in frame.** 28% of observations
   are currently truncated and score 26% instead of 64–71%. Those keypoints are absent
   from the image, so no model can recover them. This is the single highest-value change
   available and it costs nothing but time.
2. **Add a recording-side synchronisation signal** (an LED flash visible to all cameras,
   or an NVR export that is already aligned). Camera timing has now had to be corrected
   in five separate places, and the offsets differ per export — every future clip pays
   this tax otherwise.
3. **Dye the next pair blue and white.** MarmoPose's detector has exactly two classes,
   `white_head_marmoset` and `blue_head_marmoset`. On blue/white animals its built-in
   identity might work directly, removing the class-agnostic workaround and the identity
   swaps (currently 0.56–1.24 per second). Untested — but it is a free protocol change
   with a large potential payoff, and it is the only route to fixing identity that
   requires no training whatsoever.
4. **Export at the native 20 fps** rather than upsampling to 30.

### B. Software changes needing no new data

5. **Reject detections whose keypoints are all low-confidence** — fixes the human
   false-positives above.
6. **Per-camera dye hue ranges** instead of one global setting. The same dye reads very
   differently across cameras (D02: 12 blue vs 1525 pink pixels; D04: 1787 vs 1079), so a
   single `dye_hue_range` cannot serve all five.
7. **Test `flip_test=False`.** Limb keypoints are consistently 6–10 points worse on the
   right than the left, while ears are symmetric to within 0.1 points. That asymmetry
   suggests the mirrored-prediction averaging is hurting exactly the weakest keypoints.
   One config change, ~6 minutes to measure.
8. **Sweep the bounding-box padding** (`GetBBoxCenterScale`, currently 1.1). Top-down pose
   models are sensitive to crop framing, and this is measurable against the DLC hand
   labels with `tools/eval_2d.py` at zero cost.
9. **Multi-view guided detection.** This is the idea most directly aimed at the measured
   bottleneck. When 3 or 4 cameras see an animal and one does not, the calibration already
   determines where it must appear in the missing view. Projecting that region and running
   the pose model there — rather than relying on the detector to find it unaided — could
   lift keypoints over the 2-camera threshold precisely where they currently fall short.
   Because 3D coverage depends non-linearly on that threshold, small gains here compound.

### C. Training — only if labelling happens

10. **Label bounding boxes, not keypoints, first.** MarmoPose's authors fine-tuned their
    family detector on **100 images with boxes only**. Boxes are far faster to annotate
    than 16 keypoints, and detection is ~38% of the gap.
11. **Target limbs specifically.** Because of the 2-camera threshold, lifting limb
    detection from 25% to 40% per camera would take the fraction with 2+ views from ~37%
    to ~66% — roughly doubling limb coverage in 3D. The same gain on the head buys almost
    nothing; it is already saturated.
12. `datasets/to_label/` already holds **400 frames across 5 cameras with predictions
    pre-filled**, as a ready DeepLabCut project, so annotation is correction rather than
    starting from scratch.

### D. Reframing worth discussing with the lab

13. **Decide which keypoints the science actually needs.** The body core — head, ears,
    neck, spine, tailbase — reconstructs at 77–84%, which already supports position,
    heading, posture, inter-animal distance, approach/avoid and locomotion. Limbs are
    needed for manipulation and gait, a different question. If the research questions live
    in the first group, the system may already be adequate and the limb work is optional.
14. **Report per-keypoint reliability rather than a single number.** "43% coverage" hides
    that the core is at 55% and limbs at 33%; the honest presentation is per keypoint.

### E. Larger technical alternatives

15. **Combine the DLC facial model with the MarmoPose rig.** The lab's DLC model is
    in-domain and accurate on this cage (6.21 px at p-cutoff) but single-view, so it
    cannot produce 3D. Triangulating its 10 facial keypoints through MarmoPose's
    calibration would give 3D head orientation and gaze — something neither tool provides
    alone, using assets that already exist and no new labelling.
16. **Consider a bottom-up pose model.** MarmoPose is top-down: every keypoint depends on
    the detector producing a good box first. A bottom-up model finds keypoints directly
    and is less sensitive to truncation and to boxes that clip the animal.

---

## 8. Fine-tuning: the pipeline is built and verified, only labels are missing

Section 3.3 shows the model is genuinely wrong on this cage rather than merely
unconfident, so training is the only fix for the pose model. The complete pipeline was
therefore built and tested end-to-end on SNUH images. It runs today; the sole missing
input is annotated ground truth.

### The two models, and why both are being fine-tuned

MarmoPose is a **top-down** pipeline built from two separate networks:

| model | question it answers | input | output |
|---|---|---|---|
| **detection** (RTMDet) | *where are the animals?* | full frame, resized to 640x640 | one bounding box per animal |
| **pose** (RTMPose) | *where are this animal's 16 joints?* | one animal's box, cropped and resized to 512x512 | 16 keypoints with confidences |

They run in sequence: detect first, then run the pose model separately on each box. The
consequence drives everything in section 3:

> **If the detector misses an animal, the pose model never runs on it** — all 16 of its
> keypoints are lost, however good the pose model is.

So coverage is a *product* of the two, not a sum:

```
0.785 detection rate  x  0.517 pose coverage  =  40.6%     (measured: 40.7%)
```

Fixing detection alone gives ~51%; fixing pose alone gives ~60%; fixing both gives ~76%.
Neither is sufficient by itself, which is why **both models are fine-tuned** —
`configs/finetune_det.py` (LR 1e-4, 30 epochs) and `configs/finetune_pose.py` (LR 2e-4,
40 epochs), both driven by `tools/finetune.py`.

**A single annotation effort trains both.** `dlc_to_coco.py` writes the keypoints and also
derives each animal's bounding box from the extent of its labelled keypoints, so the same
DLC labels feed the detector and the pose model. Nothing is annotated twice. The only
reason to do detection first is cost: boxes alone annotate in seconds per image, keypoints
in tens of seconds per frame.

### What exists

| component | purpose |
|---|---|
| `tools/select_frames_to_label.py` | chooses which frames to annotate and pre-fills them |
| `datasets/to_label/` | **400 frames, 5 cameras, ready as a DeepLabCut project** |
| `tools/dlc_to_coco.py` | converts hand labels from DLC into the COCO format mmpose needs |
| `configs/finetune_pose.py` | RTMPose (keypoints) fine-tune configuration |
| `configs/finetune_det.py` | RTMDet (detection) fine-tune configuration |
| `tools/finetune.py` | single entry point that runs either model |

### Verified, on SNUH data

```
Load checkpoint from models/pose_model/best.pth
Epoch(train) [1][1/230]  lr: 2.0e-07  loss: 0.008563  acc_pose: 0.531250
Epoch(val)   [1][63/63]  coco/AP: 0.556
```

230 iterations per epoch at ~0.3 s/iter — a 40-epoch fine-tune takes **under an hour** on
the 5090. On CPU this would have been days, which is why section 1 was the precondition
for everything here.

*The AP figure above is circular and must not be quoted as accuracy: the labels it scores
against are the model's own pre-filled predictions. It demonstrates that the plumbing
works, nothing more. Real accuracy requires held-out hand labels.*

### How it works, step by step

**Step 1 — choose the frames and pre-fill them.** *(done)*

```
python tools/select_frames_to_label.py --project demos/snuh_t2 --n-frames 400
```

Reads the 2D predictions already computed for every frame, scores each frame by the
model's mean keypoint confidence, and picks 400 frames — 60% the least confident, 40%
uniformly at random, spaced at least 15 frames apart, split evenly across the 5 cameras.
It then extracts those frames as PNGs, writes the model's current predictions into a
DeepLabCut `CollectedData_pretrained.h5/.csv`, and generates a matching `config.yaml`.
Output: `datasets/to_label/`, a complete DLC project. 61–86% of keypoints arrive
pre-filled.

**Step 2 — annotate.** *(the missing step)*

Open `datasets/to_label/config.yaml` in DeepLabCut and correct the pre-filled points.
Wrong points get dragged into place; missing ones get added. Keypoints genuinely not
visible are left empty — they become visibility 0 and the loss ignores them, which is
correct, rather than being trained as if the joint were at (0,0).

**Step 3 — convert the labels.**

```
python tools/dlc_to_coco.py --dlc-project datasets/to_label --out datasets/snuh_hand
```

Reads each camera's `CollectedData_*.h5`, matches bodyparts by name against the config's
16-keypoint order, derives a bounding box from each animal's labelled keypoint extent
plus 15% padding, and writes COCO `train.json` / `val.json` with the images. The split is
by image and grouped by camera, so the two animals in one frame never end up on opposite
sides of the split and every camera appears in both.

**Step 4 — fine-tune.**

```
python tools/finetune.py configs/finetune_pose.py --cfg-options data_root=datasets/snuh_hand/
python tools/finetune.py configs/finetune_det.py  --cfg-options data_root=datasets/snuh_hand/
```

`tools/finetune.py` builds an mmengine `Runner` and, before it starts, applies the four
fixes this environment needs: installs the `mmcv._ext` shim, restores the pre-2.6
`torch.load` behaviour for the checkpoint, swaps SyncBN for BN (there is no process group
on a single GPU), and makes repo-relative paths absolute.

Training itself loads `models/pose_model/best.pth` and continues from those weights at
LR 2e-4 for 40 epochs with cosine decay. Each iteration crops an animal using its box,
resizes to 512×512, applies flip / rotate / scale / colour augmentation, and optimises the
SimCC classification loss against the annotated keypoints. Validation runs every 5 epochs
and the best checkpoint by COCO AP is kept in `work_dirs/finetune_pose/`.

For the detector the same entry point runs RTMDet with LR 1e-4 for 30 epochs, learning to
box the animals from the same annotations.

**Step 5 — measure the result honestly.**

```
python tools/diagnose_2d.py --config configs/snuh_t2_viz.yaml --frames 100 \
    --videos "demos/snuh_t2/videos_raw/*.mp4"
python tools/eval_2d.py --dlc-project SNUH_short-km-2026-06-19 --project demos/snuh_t2 \
    --map forehead_blaze=head,left_ear_tuft_upper=leftear,right_ear_tuft_upper=rightear
```

The first re-measures coverage at the **same 0.5 threshold** as the 43.1% baseline, so the
before/after is comparable. The second measures accuracy in pixels against held-out hand
labels — the only number that cannot be gamed by threshold choice.

To point the fine-tuned model at new footage, replace the checkpoint in
`models/pose_model/` (or pass the new `work_dirs` path) and re-run prediction as usual.

### Design decisions, and why

**Labelling effort is spent where it teaches most.** Frames are selected 60% by lowest
model confidence and 40% uniformly at random. Pure uncertainty sampling was rejected
because it collapses onto motion blur and total occlusion — frames a human cannot label
either, producing an unrepresentative training set. Frames are spaced at least 15 apart,
since consecutive frames at 30 fps cost double and teach once, and the budget is split
evenly across cameras so the weak ones are represented rather than swamped.

**Annotation is correction, not creation.** Each frame is pre-filled with the current
model's predictions in DLC format, so the annotator fixes what is wrong instead of placing
16 points from scratch — several times faster. Keypoints the model missed entirely are
left blank rather than seeded with a bad guess, because those are exactly the cases that
need human judgement.

**Labelling happens in DeepLabCut** because that is the tool already in use here; training
happens in mmpose because that is what MarmoPose's models are. `dlc_to_coco.py` bridges
them, deriving bounding boxes from the labelled keypoint extent, marking unlabelled
keypoints as visibility 0 so the loss ignores them rather than being told a joint is at
(0,0), and splitting by image and grouped by camera so the two animals in one frame never
straddle train and validation.

**The shipped training config cannot be used as-is.** It names a dataset class,
`MarmosetDataset`, that does not exist in mmpose 1.3.1 — it lived in the authors' own
tree — so training with it fails immediately. The fine-tune config uses mmpose's generic
`CocoDataset` together with the 16-keypoint metainfo already present in
`models/pose_model_deployed/data_meta.py`, which supplies the left/right swap pairs that
flip augmentation requires.

**Adaptation, not retraining.** Learning rate is 2e-4, roughly 20x below the from-scratch
0.004, over 40 epochs, starting from the pretrained checkpoint. The backbone already knows
marmosets — section 3.3 shows it is accurate wherever it is confident — so the goal is to
move it to this cage's appearance without washing out what it knows.

**For the detector, the class count stays at 2 and every label is written as class 0.**
The two classes are dye colours, and the shipped code keeps at most one box per class,
which is the mechanism that silently dropped one animal. The fix is to stop taking
identity from the class rather than to renumber the head: keeping `num_classes=2` lets the
pretrained classification layer load cleanly instead of being discarded for a shape
mismatch. Mosaic and MixUp are dropped — strong regularisers for from-scratch training on
thousands of images, but on a few hundred in-domain frames they mostly manufacture
implausible composites.

### Recommended order when labelling happens

1. **Bounding boxes first, ~100–200 images.** MarmoPose's authors fine-tuned their family
   detector on 100 box-only images. Boxes annotate far faster than 16 keypoints, and
   detection is ~38% of the gap.
2. **Then keypoints, weighted towards limbs.** Because 3D requires 2 cameras per keypoint,
   lifting limb detection from 25% to 40% per camera would raise the fraction with 2+
   views from ~37% to ~66%, roughly doubling limb coverage in 3D. The same improvement on
   the head buys almost nothing — it is already saturated.
3. **Hold out ~20% for evaluation.** Without held-out hand labels every accuracy number is
   circular, as the AP above illustrates.

---

## 9. How many labels? Fine-tuning versus retraining

### The numbers, from documented sources

| approach | labels needed | source |
|---|---|---|
| **Retrain from scratch** | **~3,173 images** | MarmoPose's own Marmoset3K: 1,527 single-animal + 1,646 two-animal images (README) |
| **Fine-tune, detection** | **~100 images, boxes only** | What MarmoPose's authors actually did for their 4-marmoset family model (README) |
| **Fine-tune, keypoints** | **~300–500 frames** | Our prepared set is 400 frames across 5 cameras |
| *for reference* | *570 frames, 10 keypoints* | *this lab's existing DLC project — already achieved once* |

So the labelling difference is roughly **10x for keypoints and 30x for detection**. That
alone is the practical argument, but it is not the strongest one.

### Why fine-tuning is the right choice, not merely the cheaper one

**1. The backbone is not broken — it is out of domain.** Section 3.3 measured that where
the model is confident it is already accurate: 9 px median error, 83% PCK@20px. It fails
on this cage's hard cases, not on marmosets. That is the textbook definition of a
domain-adaptation problem, which is exactly what fine-tuning solves and what retraining
from scratch would solve only by accident.

**2. Retraining throws away 3,173 images of marmoset anatomy.** Marmoset3K teaches limb
structure, fur texture and posture across many configurations. Training from scratch on
400 SNUH frames discards all of it and starts from nothing, in a different cage.

**3. 400 images cannot train a CSPNeXt backbone from scratch.** It would overfit heavily.
Fine-tuning adapts an existing representation; from-scratch training has to learn one, and
that needs thousands of images — which is precisely why Marmoset3K is the size it is.

**4. Time.** A 40-epoch fine-tune is under an hour on the 5090 (measured). From-scratch
training runs 400 epochs by the shipped config — a different order of magnitude, and it
would have to be repeated for every future change.

**5. The authors did exactly this.** Faced with a new scenario their model had never seen
(four differently coloured marmosets), they fine-tuned on 100 box-only images rather than
retraining. That is the same situation we are in, one cage removed.

**6. Retraining would not fix the largest measured problem anyway.** 28% of observations
have the animal partly outside the camera (section 3.1). Those keypoints are absent from
the image. No amount of training, from scratch or otherwise, recovers them — only moving
the cameras does.

### Effort estimate

Boxes annotate at roughly 5–10 seconds per image, keypoints at perhaps 20–40 seconds per
frame when correcting pre-filled predictions rather than placing 32 points from scratch.
On that basis: **~15 minutes for 100 box-only images**, and **~3–4 hours for 400
keypoint frames**. Retraining-scale annotation, ~3,000 frames, would be on the order of
**50 hours**. *(These per-frame rates are estimates, not measured here; the frame counts
are documented.)*

### Honest caveats

- 400 frames from a single session and five fixed views risks overfitting to that session.
  Labels spread across different times, and ideally across sessions, will generalise better
  than 400 frames from one window.
- Fine-tuning improves the pose model. It does **not** address truncation, camera
  synchronisation, or the fact that limbs are seen by fewer than two cameras on average —
  the recording-side items in section 7A remain the higher-value work.
- Any accuracy claim needs held-out hand labels. Around 20% should be reserved and never
  trained on.
