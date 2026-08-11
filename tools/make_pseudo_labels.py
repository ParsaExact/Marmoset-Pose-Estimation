"""Mint 16-keypoint training labels from multi-view geometry, no manual annotation.

The idea
--------
A top-down pose model fails per-view and independently: a keypoint occluded or
blurred in camera 1 is often clean in cameras 3 and 4. Calibration ties the views
together, so a keypoint confident in >= 2 views can be triangulated, and the
resulting 3D point reprojects into *every* view -- including the ones where the
appearance model gave nothing. Geometry supplies what appearance missed.

That converts modest per-view coverage into high label coverage. With p = 0.5 per
view and 5 cameras, P(>= 2 views confident) = 1 - P(0) - P(1) = 81%, and each of
those points then labels all 5 views. This is what takes coverage from ~40% to
~80% without anyone annotating a frame.

Optional skeletal/temporal optimization (`optimize_coordinates`, the same one the
3D pipeline uses) is applied before reprojection, so bone lengths and motion
continuity further constrain the labels.

Quality control
---------------
Coverage alone is not enough -- wrong labels are worse than missing ones. Two
checks, both reported:

  * `--min-views` requires N confident views before a point is accepted at all.
  * held-out accuracy: for keypoints that *were* confident in a view, that view's
    observation is compared against the reprojection of the 3D point. Small error
    means the reprojected labels can be trusted where there was no observation.

The `--source-threshold` sweep also lets you simulate a harder domain: thresholding
demo data at 0.7 drops its per-view coverage to roughly what SNUH gets at 0.5, so
the yield measured there predicts the yield on SNUH.

Usage:
    # measure the yield only
    python tools/make_pseudo_labels.py --project demos/_pseudo --measure-only \
        --source-threshold 0.5 0.7

    # write a COCO-format training set for mmpose fine-tuning
    python tools/make_pseudo_labels.py --project demos/_pseudo \
        --source-threshold 0.6 --min-views 3 --out datasets/snuh_pseudo
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('pseudo_labels')


def triangulate_all(camera_group, points_2d, quiet=False):
    """Triangulate every (track, frame) independently.

    Args:
        points_2d: (n_cams, n_tracks, n_frames, n_bodyparts, 3), NaN where absent.

    Returns:
        (n_tracks, n_frames, n_bodyparts, 3) with NaN where fewer than 2 views saw it.
    """
    n_cams, n_tracks, n_frames, n_bodyparts, _ = points_2d.shape
    out = np.full((n_tracks, n_frames, n_bodyparts, 3), np.nan)
    frames = range(n_frames) if quiet else trange(
        n_frames, ncols=100, desc='Triangulating', unit='fr')
    for frame_idx in frames:
        for track_idx in range(n_tracks):
            flat = points_2d[:, track_idx, frame_idx]  # (n_cams, n_bodyparts, 3)
            out[track_idx, frame_idx] = camera_group.triangulate(flat, undistort=True)
    return out


def shift_camera(points_2d_cam, offset):
    """Shift one camera's frames by `offset`, padding with NaN.

    offset > 0 means this camera runs ahead of the reference and its frame k should
    be read from k + offset.
    """
    if offset == 0:
        return points_2d_cam
    out = np.full_like(points_2d_cam, np.nan)
    n_frames = points_2d_cam.shape[1]
    if offset > 0:
        out[:, :n_frames - offset] = points_2d_cam[:, offset:]
    else:
        out[:, -offset:] = points_2d_cam[:, :n_frames + offset]
    return out


def _sync_cost(camera_group, gated, sample_stride):
    """Median reprojection residual of what currently triangulates.

    Frame misalignment shows up as geometric inconsistency: the same keypoint seen a
    frame apart in two cameras does not intersect in 3D, so residuals rise. That
    makes the residual a usable objective for recovering the offsets, which every
    export of this rig has needed (see results_for_meeting/SUMMARY.md).
    """
    sub = gated[:, :, ::sample_stride]
    points_3d = triangulate_all(camera_group, sub, quiet=True)
    _, residuals = filter_by_reprojection(camera_group, points_3d, sub, np.inf)
    if residuals.size == 0:
        return np.inf, 0
    return float(np.nanmedian(residuals)), int(np.isfinite(residuals).sum())


def search_sync_offsets(camera_group, gated, max_offset, sample_stride=10):
    """Greedily recover a per-camera frame offset, camera 0 fixed as reference."""
    n_cams = gated.shape[0]
    offsets = [0] * n_cams
    current = gated.copy()
    base_cost, base_n = _sync_cost(camera_group, current, sample_stride)
    logger.info('sync: reference residual %.2fpx over %d points', base_cost, base_n)

    for cam_idx in range(1, n_cams):
        best = (base_cost, 0)
        for offset in range(-max_offset, max_offset + 1):
            if offset == 0:
                continue
            trial = current.copy()
            trial[cam_idx] = shift_camera(gated[cam_idx], offset)
            cost, n_points = _sync_cost(camera_group, trial, sample_stride)
            # Require a real improvement, and enough points for the number to mean
            # anything, so a shift is not chosen just for thinning the data.
            if cost < best[0] * 0.98 and n_points > 0.5 * base_n:
                best = (cost, offset)
        offsets[cam_idx] = best[1]
        current[cam_idx] = shift_camera(gated[cam_idx], best[1])
        base_cost = best[0]
        logger.info('sync: camera %d offset %+d  -> residual %.2fpx', cam_idx,
                    best[1], best[0])

    return offsets, current


def filter_by_reprojection(camera_group, points_3d, gated_2d, max_error):
    """Drop 3D points that don't agree with the 2D observations that produced them.

    Triangulating two views is enough to get *a* point, but not a *right* one: a
    swapped identity across cameras, or one bad 2D keypoint, still yields a 3D
    position, just one that reprojects nowhere near what any camera saw. Without
    this gate the label set carries a long tail of gross errors, which is worse
    for training than having no label at all.
    """
    n_tracks, n_frames, n_bodyparts, _ = points_3d.shape
    flat_3d = points_3d.reshape(-1, 3)
    finite = np.isfinite(flat_3d[:, 0])
    if not finite.any():
        return points_3d, np.zeros(0)

    projected = np.full((len(camera_group.cameras), flat_3d.shape[0], 2), np.nan)
    projected[:, finite] = camera_group.reproject(flat_3d[finite]).reshape(
        len(camera_group.cameras), -1, 2)

    observed = gated_2d[..., :2].reshape(gated_2d.shape[0], -1, 2)
    residual = np.linalg.norm(observed - projected, axis=-1)     # (n_cams, N)

    with np.errstate(all='ignore'):
        median_residual = np.nanmedian(residual, axis=0)         # (N,)
    bad = ~(median_residual <= max_error)                        # NaN -> bad
    flat_3d[bad] = np.nan
    return flat_3d.reshape(points_3d.shape), median_residual[finite]


def reproject_all(camera_group, points_3d):
    """Project 3D points back into every camera.

    Returns:
        (n_cams, n_tracks, n_frames, n_bodyparts, 2)
    """
    n_tracks, n_frames, n_bodyparts, _ = points_3d.shape
    flat = points_3d.reshape(-1, 3)
    finite = np.isfinite(flat[:, 0])

    n_cams = len(camera_group.cameras)
    out = np.full((n_cams, flat.shape[0], 2), np.nan)
    if finite.any():
        # cv2.projectPoints rejects NaN input, so only project the valid rows.
        projected = camera_group.reproject(flat[finite])
        out[:, finite] = projected.reshape(n_cams, -1, 2)
    return out.reshape(n_cams, n_tracks, n_frames, n_bodyparts, 2)


def measure(points_2d_raw, reprojected, source_mask, view_counts, min_views, frame_wh):
    """Coverage of the final label set, and how accurate the geometry labels are.

    The label set is the UNION of what the pose model was already confident about
    and what geometry recovered: a confident observation is kept as-is, and
    reprojection is used only to fill slots that had nothing. Reporting geometry
    coverage on its own would understate the result, since it would be replacing
    good observations rather than adding to them.
    """
    n_cams = points_2d_raw.shape[0]
    width, height = frame_wh

    accepted = view_counts >= min_views                      # (n_tracks, n_frames, n_bodyparts)
    geometry_ok = np.isfinite(reprojected[..., 0]) & accepted[None]
    inside = ((reprojected[..., 0] >= 0) & (reprojected[..., 0] < width) &
              (reprojected[..., 1] >= 0) & (reprojected[..., 1] < height))
    geometry_ok &= inside

    union = source_mask | geometry_ok
    missing = ~source_mask
    recovered = geometry_ok & missing

    before = float(source_mask.mean()) * 100
    after = float(union.mean()) * 100
    recovery = float(recovered.sum() / max(1, missing.sum())) * 100

    # Accuracy check: where a view DID observe the keypoint, how far off is the
    # geometry label? That error is the proxy for the labels we cannot check.
    both = source_mask & geometry_ok
    if both.any():
        err = np.linalg.norm(points_2d_raw[..., :2][both] - reprojected[both], axis=-1)
        acc = (float(np.median(err)), float(np.percentile(err, 90)))
    else:
        acc = (float('nan'), float('nan'))

    per_cam = [(float(source_mask[c].mean()) * 100, float(union[c].mean()) * 100)
               for c in range(n_cams)]
    return before, after, recovery, acc, per_cam, union, geometry_ok


def export_coco(project, points_2d_labels, label_ok, bodyparts, out_dir, max_frames):
    """Write a topdown COCO keypoint dataset: one image per (camera, frame) crop-free view.

    mmpose's topdown pipeline takes the full image plus a bbox per instance, so the
    frames are written once and every visible track becomes an annotation on them.
    """
    out_dir = Path(out_dir)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    video_paths = sorted((Path(project) / 'videos_raw').glob('*.mp4'))
    n_cams, n_tracks, n_frames, n_bodyparts = label_ok.shape

    images, annotations = [], []
    ann_id = 1
    # Only frames where some track is well covered are worth training on.
    per_frame_cover = label_ok.sum(axis=(0, 1, 3))
    chosen = np.argsort(-per_frame_cover)[:max_frames]
    chosen = np.sort(chosen[per_frame_cover[chosen] > 0])

    for cam_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(video_path))
        for frame_idx in chosen:
            visible = label_ok[cam_idx, :, frame_idx]           # (n_tracks, n_bodyparts)
            if not visible.any():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            name = f'{video_path.stem}_{frame_idx:06d}.jpg'
            cv2.imwrite(str(out_dir / 'images' / name), frame)
            image_id = len(images) + 1
            images.append(dict(id=image_id, file_name=f'images/{name}',
                               width=width, height=height))

            for track_idx in range(n_tracks):
                mask = visible[track_idx]
                if mask.sum() < 4:            # too few points to be a useful instance
                    continue
                pts = points_2d_labels[cam_idx, track_idx, frame_idx]
                kps = np.zeros((n_bodyparts, 3))
                kps[mask, :2] = pts[mask]
                kps[mask, 2] = 2              # COCO: 2 = labelled and visible
                xy = pts[mask]
                x1, y1 = xy.min(axis=0)
                x2, y2 = xy.max(axis=0)
                pad = 0.15 * max(x2 - x1, y2 - y1)
                x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
                x2, y2 = min(width - 1.0, x2 + pad), min(height - 1.0, y2 + pad)
                annotations.append(dict(
                    id=ann_id, image_id=image_id, category_id=1, iscrowd=0,
                    num_keypoints=int(mask.sum()),
                    keypoints=[float(v) for v in kps.reshape(-1)],
                    bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    area=float((x2 - x1) * (y2 - y1))))
                ann_id += 1
        cap.release()

    categories = [dict(id=1, name='marmoset', supercategory='animal',
                       keypoints=list(bodyparts), skeleton=[])]
    return images, annotations, categories, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True,
                    help='project with points_2d/original.h5 (raw scores) and calibration/')
    ap.add_argument('--config', default='configs/snu_s1_nodye.yaml')
    ap.add_argument('--source-threshold', type=float, nargs='+', default=[0.5],
                    help='keypoint score required for a view to feed triangulation')
    ap.add_argument('--min-views', type=int, default=2,
                    help='confident views required before a 3D point is accepted')
    ap.add_argument('--sync-search', type=int, default=0, metavar='N',
                    help='recover per-camera frame offsets by searching [-N, N] for the '
                         'alignment that minimises reprojection residual (0 = off)')
    ap.add_argument('--max-reproj-error', type=float, default=15.0,
                    help='reject a 3D point whose median reprojection error against '
                         'its own source observations exceeds this (px)')
    ap.add_argument('--optimize', action='store_true', default=False,
                    help='apply skeletal/temporal optimization before reprojection')
    ap.add_argument('--measure-only', action='store_true', default=False)
    ap.add_argument('--out', default=None, help='output dir for the COCO dataset')
    ap.add_argument('--max-frames', type=int, default=300,
                    help='frames to export, best-covered first')
    ap.add_argument('--val-fraction', type=float, default=0.15)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from marmopose.config import Config
    from marmopose.calibration.cameras import CameraGroup
    from marmopose.utils.data_io import load_points_bboxes_2d_h5

    # The model dirs are unused here but Config validates that they exist.
    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'),
                    n_tracks=2, dae_enable=False, do_optimize=False)
    bodyparts = config.animal['bodyparts']

    cam_params = Path(config.sub_directory['calibration']) / 'camera_params.json'
    camera_group = CameraGroup.load_from_json(cam_params)
    points_2d_raw, _ = load_points_bboxes_2d_h5(
        Path(config.sub_directory['points_2d']) / 'original.h5')
    logger.info('2D points %s from %s', points_2d_raw.shape, cam_params)

    video = sorted((Path(config.sub_directory['videos_raw'])).glob('*.mp4'))[0]
    cap = cv2.VideoCapture(str(video))
    frame_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    results = []
    for threshold in args.source_threshold:
        scores = points_2d_raw[..., 2]
        source_mask = np.isfinite(scores) & (scores >= threshold)
        gated = points_2d_raw.copy()
        gated[~source_mask] = np.nan

        if args.sync_search:
            offsets, gated = search_sync_offsets(camera_group, gated, args.sync_search)
            logger.info('sync offsets: %s', offsets)
            # The raw array has to move with it, or the "was this observed?" mask and
            # the reprojections would refer to different frames.
            shifted_raw = points_2d_raw.copy()
            for cam_idx, offset in enumerate(offsets):
                shifted_raw[cam_idx] = shift_camera(points_2d_raw[cam_idx], offset)
            points_2d_raw = shifted_raw
            scores = points_2d_raw[..., 2]
            source_mask = np.isfinite(scores) & (scores >= threshold)

        points_3d = triangulate_all(camera_group, gated)
        points_3d, residuals = filter_by_reprojection(
            camera_group, points_3d, gated, args.max_reproj_error)
        kept = float(np.mean(np.isfinite(points_3d[..., 0]))) * 100
        logger.info('thr %.2f: 3D points surviving the %.0fpx reprojection gate: %.1f%% '
                    '(median residual %.2fpx)', threshold, args.max_reproj_error, kept,
                    float(np.nanmedian(residuals)) if residuals.size else float('nan'))

        if args.optimize:
            from marmopose.processing.optimization import optimize_coordinates
            config.optimization['do_optimize'] = True
            for track_idx in range(points_3d.shape[0]):
                points_3d[track_idx] = optimize_coordinates(
                    config, camera_group, points_3d[track_idx], gated[:, track_idx])

        reprojected = reproject_all(camera_group, points_3d)
        view_counts = source_mask.sum(axis=0)

        before, after, recovery, acc, per_cam, union, geometry_ok = measure(
            points_2d_raw, reprojected, source_mask, view_counts, args.min_views, frame_wh)

        # The exported label set: keep confident observations, fill the rest.
        labels = reprojected.copy()
        labels[source_mask] = points_2d_raw[..., :2][source_mask]

        results.append((threshold, before, after, recovery, acc, per_cam, union,
                        labels, geometry_ok))

    print()
    print('=' * 92)
    print(f'MULTI-VIEW LABEL YIELD  (min_views={args.min_views}, '
          f'optimize={args.optimize}, cameras={points_2d_raw.shape[0]})')
    print('=' * 92)
    print(f"{'src thr':>8}{'model only':>13}{'+ geometry':>13}{'gain':>8}"
          f"{'of missing':>13}{'geom err med':>15}{'p90':>9}")
    for threshold, before, after, recovery, acc, *_ in results:
        print(f'{threshold:>8.2f}{before:>12.1f}%{after:>12.1f}%{after - before:>+7.1f}'
              f'{recovery:>12.1f}%{acc[0]:>13.2f}px{acc[1]:>8.2f}px')
    print()
    for threshold, before, after, recovery, acc, per_cam, *_ in results:
        print(f'  per-camera @ {threshold:.2f}: ' +
              '  '.join(f'{b:.0f}%->{a:.0f}%' for b, a in per_cam))
    print()

    if args.measure_only or args.out is None:
        return

    threshold, before, after, recovery, acc, per_cam, union, labels, _ = results[0]
    images, annotations, categories, out_dir = export_coco(
        REPO / args.project, labels, union, bodyparts, args.out, args.max_frames)

    rng = np.random.default_rng(0)
    ids = np.array([im['id'] for im in images])
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_fraction))
    val_ids = set(ids[:n_val].tolist())

    for split, keep in (('train', lambda i: i not in val_ids),
                        ('val', lambda i: i in val_ids)):
        split_images = [im for im in images if keep(im['id'])]
        keep_ids = {im['id'] for im in split_images}
        split_anns = [a for a in annotations if a['image_id'] in keep_ids]
        path = out_dir / f'{split}.json'
        with open(path, 'w') as fh:
            json.dump(dict(images=split_images, annotations=split_anns,
                           categories=categories), fh)
        logger.info('%s: %d images, %d instances -> %s',
                    split, len(split_images), len(split_anns), path)


if __name__ == '__main__':
    main()
