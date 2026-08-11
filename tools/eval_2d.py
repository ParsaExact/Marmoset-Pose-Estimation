"""Measure 2D keypoint *accuracy* against hand labels, not just coverage.

`diagnose_2d.py` reports how many keypoints clear a confidence threshold. That is
coverage, and coverage can be raised for free by lowering the threshold. This tool
answers the question coverage cannot: when the model does predict a keypoint, is it
in the right place -- and does its confidence actually tell you so?

The last part is the one that decides the plan. Two very different failures produce
identical coverage numbers:

  * the model is **wrong** on this domain -> fine-tuning is the fix
  * the model is **right but under-confident** -> its confidence is miscalibrated for
    this cage, and most of the "missing" keypoints are recoverable by thresholding
    differently, with no training at all

So error is reported binned by confidence. If low-confidence predictions are accurate,
the coverage gap is largely a calibration artefact.

Ground truth comes from a DeepLabCut project. The 10-keypoint facial project on this
footage does not cover MarmoPose's 16 body keypoints, but several correspond
(ear tufts, forehead), and `--discover` finds which by measuring the cross-distance
between every model keypoint and every labelled keypoint rather than trusting names.

Animal identity is matched per frame by centroid, since DLC's `individualN` and
MarmoPose's track index have no reason to agree.

Usage:
    python tools/eval_2d.py --dlc-project SNUH_short-km-2026-06-19 \\
        --project demos/snuh_t2 --discover
    python tools/eval_2d.py --dlc-project SNUH_short-km-2026-06-19 \\
        --project demos/snuh_t2 --map forehead_blaze=head,left_ear_tuft=leftear
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('eval_2d')

CONFIDENCE_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def load_dlc_labels(camera_dir):
    """Return {frame_index: (n_individuals, n_parts, 2)} and the bodypart names."""
    tables = sorted(camera_dir.glob('CollectedData_*.h5'))
    if not tables:
        return {}, []
    table = pd.read_hdf(tables[0])

    parts = list(dict.fromkeys(table.columns.get_level_values('bodyparts')))
    if 'individuals' in table.columns.names:
        individuals = list(dict.fromkeys(table.columns.get_level_values('individuals')))
    else:
        individuals = [None]

    out = {}
    for row_index, row in table.iterrows():
        name = row_index[-1] if isinstance(row_index, tuple) else str(row_index)
        stem = Path(str(name)).stem
        digits = ''.join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        frame_idx = int(digits)

        coords = np.full((len(individuals), len(parts), 2), np.nan)
        for i, individual in enumerate(individuals):
            for j, part in enumerate(parts):
                try:
                    if individual is None:
                        x = row.xs((part, 'x'), level=('bodyparts', 'coords'))
                        y = row.xs((part, 'y'), level=('bodyparts', 'coords'))
                    else:
                        x = row.xs((individual, part, 'x'),
                                   level=('individuals', 'bodyparts', 'coords'))
                        y = row.xs((individual, part, 'y'),
                                   level=('individuals', 'bodyparts', 'coords'))
                except KeyError:
                    continue
                coords[i, j] = (float(np.asarray(x).ravel()[0]),
                                float(np.asarray(y).ravel()[0]))
        out[frame_idx] = coords
    return out, parts


def centroid(points):
    finite = np.isfinite(points[..., 0])
    if not finite.any():
        return None
    return np.nanmean(points[finite], axis=0)


def match_instances(label_coords, model_points):
    """Pair labelled individuals with model tracks by centroid distance.

    Returns a list of (label_index, track_index).
    """
    label_centres = [centroid(label_coords[i]) for i in range(label_coords.shape[0])]
    model_centres = [centroid(model_points[t, :, :2]) for t in range(model_points.shape[0])]

    label_ids = [i for i, c in enumerate(label_centres) if c is not None]
    track_ids = [t for t, c in enumerate(model_centres) if c is not None]
    if not label_ids or not track_ids:
        return []

    cost = np.array([[np.linalg.norm(label_centres[i] - model_centres[t])
                      for t in track_ids] for i in label_ids])
    rows, cols = linear_sum_assignment(cost)
    return [(label_ids[r], track_ids[c]) for r, c in zip(rows, cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dlc-project', required=True)
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snu_charuco.yaml')
    ap.add_argument('--discover', action='store_true', default=False,
                    help='report the closest model keypoint for each labelled keypoint')
    ap.add_argument('--map', default=None,
                    help='comma list of labelpart=modelpart to evaluate')
    ap.add_argument('--max-match-distance', type=float, default=250.0,
                    help='reject an instance pairing whose centroids are further apart')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from marmopose.config import Config
    from marmopose.utils.data_io import load_points_bboxes_2d_h5

    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'))
    model_parts = list(config.animal['bodyparts'])

    all_points, all_bboxes = load_points_bboxes_2d_h5(
        Path(config.sub_directory['points_2d']) / 'original.h5')
    cameras = sorted(p.stem for p in Path(config.sub_directory['videos_raw']).glob('*.mp4'))

    dlc_root = Path(args.dlc_project)
    if not dlc_root.is_absolute():
        dlc_root = REPO / dlc_root
    labeled_root = dlc_root / 'labeled-data'

    # (model_part, label_part) -> distances, plus per-pair confidences.
    pair_distances = {}
    matched_instances = 0
    label_parts = []

    for cam_idx, camera in enumerate(cameras):
        camera_dir = labeled_root / camera
        if not camera_dir.is_dir():
            logger.info('%s: no hand labels', camera)
            continue
        labels, parts = load_dlc_labels(camera_dir)
        if not labels:
            continue
        label_parts = parts or label_parts
        n_frames = all_points.shape[2]

        used = 0
        for frame_idx, label_coords in labels.items():
            if frame_idx >= n_frames:
                continue
            model_points = all_points[cam_idx, :, frame_idx]      # (tracks, parts, 3)
            for label_idx, track_idx in match_instances(label_coords, model_points):
                lab = label_coords[label_idx]
                mod = model_points[track_idx]
                lc, mc = centroid(lab), centroid(mod[:, :2])
                if lc is None or mc is None:
                    continue
                if np.linalg.norm(lc - mc) > args.max_match_distance:
                    continue
                matched_instances += 1
                used += 1
                for j, label_part in enumerate(parts):
                    if not np.isfinite(lab[j, 0]):
                        continue
                    for i, model_part in enumerate(model_parts):
                        if not np.isfinite(mod[i, 0]):
                            continue
                        distance = float(np.linalg.norm(lab[j] - mod[i, :2]))
                        pair_distances.setdefault((model_part, label_part), []).append(
                            (distance, float(mod[i, 2])))
        logger.info('%s: %d labelled frames, %d instances matched', camera,
                    len(labels), used)

    if not pair_distances:
        raise SystemExit('no overlap between hand labels and predictions')
    print(f'\nmatched instances: {matched_instances}\n')

    if args.discover:
        print('=' * 78)
        print('CORRESPONDENCE - closest model keypoint to each hand-labelled keypoint')
        print('=' * 78)
        print(f"{'hand label':<24}{'closest model kp':<16}{'median px':>11}{'n':>7}")
        for label_part in label_parts:
            best = None
            for model_part in model_parts:
                values = pair_distances.get((model_part, label_part))
                if not values or len(values) < 20:
                    continue
                median = float(np.median([d for d, _ in values]))
                if best is None or median < best[1]:
                    best = (model_part, median, len(values))
            if best:
                print(f'{label_part:<24}{best[0]:<16}{best[1]:>11.1f}{best[2]:>7}')
        print()

    if not args.map:
        print('Pick correspondences from the table above and re-run with e.g.')
        print('  --map forehead_blaze=head,left_ear_tuft=leftear,right_ear_tuft=rightear')
        return

    mapping = [pair.split('=') for pair in args.map.split(',')]
    print('=' * 78)
    print('ACCURACY on mapped keypoints (hand labels as ground truth)')
    print('=' * 78)
    print(f"{'pair':<34}{'n':>7}{'median px':>11}{'p90':>9}{'PCK@20px':>11}")
    for label_part, model_part in mapping:
        values = pair_distances.get((model_part, label_part))
        if not values:
            print(f'{label_part+" -> "+model_part:<34}  no data')
            continue
        distances = np.array([d for d, _ in values])
        print(f'{label_part+" -> "+model_part:<34}{len(distances):>7}'
              f'{np.median(distances):>11.1f}{np.percentile(distances, 90):>9.1f}'
              f'{100 * float(np.mean(distances <= 20)):>10.1f}%')

    print()
    print('=' * 78)
    print('IS LOW CONFIDENCE ACTUALLY WRONG?  error binned by predicted score')
    print('=' * 78)
    print(f"{'score bin':<14}{'n':>8}{'median px':>11}{'p90':>9}{'PCK@20px':>11}")
    pooled = []
    for label_part, model_part in mapping:
        pooled.extend(pair_distances.get((model_part, label_part), []))
    pooled = np.array(pooled) if pooled else np.zeros((0, 2))
    for low, high in CONFIDENCE_BINS:
        if pooled.size == 0:
            break
        sel = pooled[(pooled[:, 1] >= low) & (pooled[:, 1] < high)]
        if sel.shape[0] == 0:
            print(f'{f"{low:.1f}-{high:.1f}":<14}{0:>8}')
            continue
        distances = sel[:, 0]
        print(f'{f"{low:.1f}-{high:.1f}":<14}{sel.shape[0]:>8}'
              f'{np.median(distances):>11.1f}{np.percentile(distances, 90):>9.1f}'
              f'{100 * float(np.mean(distances <= 20)):>10.1f}%')
    print()


if __name__ == '__main__':
    main()
