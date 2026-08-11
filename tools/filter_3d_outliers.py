"""Remove anatomically impossible 3D points, then refill short gaps.

Why this is needed
------------------
After triangulation with the full bone-length constraints, the *median* bone lengths
match their targets almost exactly (head-rightear 21.0 vs 21.0 mm target, neck-spinemid
57.5 vs 58.0). But their standard deviations run into hundreds of millimetres, which
means the skeleton is anatomically right most of the time and grossly wrong in a
minority of frames -- a keypoint flung hundreds of mm away, drawing the long stray
lines that make the reconstruction look broken.

The optimizer cannot fix these on its own: its length terms are soft penalties, so one
badly triangulated point (two cameras that agreed on the wrong animal, say) is cheaper
to keep than to move. They have to be rejected outright.

Two filters, both anatomical rather than statistical, so the threshold means something:

  * **radius** -- a marmoset is roughly 250 mm nose to tail base, so a keypoint more
    than `--max-radius` from its own body's centre cannot belong to that animal. The
    centre is a median over the body's own keypoints, so it survives a few bad ones.
  * **bone length** -- a bone stretched beyond `--length-tolerance` times its target
    is impossible; the endpoint that disagrees with the rest of the body is dropped.

Gaps are then refilled by linear interpolation, but **only across gaps shorter than
`--max-gap` frames**. MarmoPose's own interpolation is unbounded, which invents
trajectories across arbitrarily long absences and (per results_for_meeting/SUMMARY.md)
can write the world origin for an all-missing track. Bounding it keeps interpolation
honest: short gaps are genuinely predictable, long ones are not, and leaving them empty
tells you where the data really is.

Usage:
    python tools/filter_3d_outliers.py --project demos/snuh_t2 --source optimized
"""
import argparse
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('filter_3d')


def filter_by_radius(points, max_radius):
    """Drop keypoints implausibly far from their own body centre.

    Args:
        points: (n_frames, n_bodyparts, 3)
    """
    centre = np.nanmedian(points, axis=1, keepdims=True)     # robust to a few bad points
    with np.errstate(invalid='ignore'):
        distance = np.linalg.norm(points - centre, axis=-1)
    bad = distance > max_radius
    out = points.copy()
    out[bad] = np.nan
    return out, int(np.count_nonzero(bad & np.isfinite(distance)))


def filter_by_bone_length(points, bones, tolerance):
    """Drop the endpoint of any bone stretched beyond `tolerance` x its target.

    Which endpoint is wrong is decided by agreement with the rest of the body: the one
    further from the body centre is the one dropped.
    """
    out = points.copy()
    dropped = 0
    centre = np.nanmedian(points, axis=1, keepdims=True)
    for (i, j), target in bones:
        with np.errstate(invalid='ignore'):
            length = np.linalg.norm(out[:, i] - out[:, j], axis=-1)
        bad = np.isfinite(length) & (length > tolerance * target)
        if not bad.any():
            continue
        di = np.linalg.norm(out[:, i] - centre[:, 0], axis=-1)
        dj = np.linalg.norm(out[:, j] - centre[:, 0], axis=-1)
        drop_i = bad & (di >= dj)
        drop_j = bad & (dj > di)
        out[drop_i, i] = np.nan
        out[drop_j, j] = np.nan
        dropped += int(drop_i.sum() + drop_j.sum())
    return out, dropped


def interpolate_short_gaps(points, max_gap):
    """Linearly fill gaps of at most `max_gap` frames, per keypoint and axis."""
    out = points.copy()
    n_frames, n_parts, _ = points.shape
    filled = 0
    for part in range(n_parts):
        valid = np.isfinite(points[:, part, 0])
        if valid.sum() < 2:
            continue
        indices = np.flatnonzero(valid)
        for start, end in zip(indices[:-1], indices[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            for axis in range(3):
                out[start + 1:end, part, axis] = np.linspace(
                    points[start, part, axis], points[end, part, axis], gap + 2)[1:-1]
            filled += gap
    return out, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snuh_t2_viz.yaml')
    ap.add_argument('--source', default='optimized', choices=['original', 'optimized'])
    ap.add_argument('--max-radius', type=float, default=260.0,
                    help='mm from the body centre beyond which a keypoint is impossible')
    ap.add_argument('--length-tolerance', type=float, default=2.5,
                    help='a bone longer than this multiple of its target is impossible')
    ap.add_argument('--max-gap', type=int, default=10,
                    help='longest gap (frames) that may be interpolated; 0 disables')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    cfg = yaml.safe_load(open(REPO / args.config))
    parts = cfg['animal']['bodyparts']
    index = {p: i for i, p in enumerate(parts)}
    targets = {**cfg['optimization'].get('bodypart_distance', {}),
               **(cfg['optimization'].get('bodypart_distance_weak') or {})}
    bones = []
    for name, target in targets.items():
        a, b = [s.strip() for s in name.split('-')]
        if a in index and b in index:
            bones.append(((index[a], index[b]), float(target)))

    path = REPO / args.project / 'points_3d' / f'{args.source}.h5'
    backup = path.with_name(f'{args.source}_prefilter.h5')
    if not backup.exists():
        shutil.copyfile(path, backup)
        logger.info('kept the unfiltered version at %s', backup.name)

    with h5py.File(backup, 'r') as fh:
        tracks = {k: np.array(fh[k]) for k in sorted(fh.keys())}

    with h5py.File(path, 'w') as fh:
        for name, points in tracks.items():
            before = int(np.isfinite(points[..., 0]).sum())
            points, n_radius = filter_by_radius(points, args.max_radius)
            points, n_bone = filter_by_bone_length(points, bones, args.length_tolerance)
            filled = 0
            if args.max_gap > 0:
                points, filled = interpolate_short_gaps(points, args.max_gap)
            after = int(np.isfinite(points[..., 0]).sum())
            total = points[..., 0].size
            logger.info('%s: dropped %d by radius, %d by bone length, refilled %d '
                        '-> coverage %.1f%% (was %.1f%%)',
                        name, n_radius, n_bone, filled,
                        100 * after / total, 100 * before / total)
            fh.create_dataset(name, data=points)

    # Report the effect on anatomical consistency, which is the point of the exercise.
    print()
    print('%-28s %8s %10s %8s' % ('bone', 'target', 'median', 'std'))
    with h5py.File(path, 'r') as fh:
        cleaned = [np.array(fh[k]) for k in sorted(fh.keys())]
    for (i, j), target in bones[:8]:
        lengths = []
        for points in cleaned:
            length = np.linalg.norm(points[:, i] - points[:, j], axis=-1)
            lengths.append(length[np.isfinite(length)])
        lengths = np.concatenate(lengths) if lengths else np.zeros(0)
        if lengths.size:
            print('%-28s %8.1f %10.1f %8.1f' % (
                f'{parts[i]} - {parts[j]}', target,
                float(np.median(lengths)), float(np.std(lengths))))
    print()


if __name__ == '__main__':
    main()
