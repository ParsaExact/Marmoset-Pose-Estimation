"""Make 3D trajectories temporally coherent, so the reconstruction reads as motion.

The reconstruction is built frame by frame: each frame is triangulated independently,
so nothing links a keypoint at frame t to the same keypoint at t+1. Combined with
outlier removal, which leaves holes, the result is keypoints that jump and blink rather
than move -- anatomically plausible in any single frame, but not recognisable as an
animal walking.

Three stages, in order, because each depends on the previous one being done:

  1. **median filter** -- kills isolated single-frame spikes that survived the
     anatomical filter. A mean would smear a spike across its neighbours instead of
     removing it; the median simply ignores it.
  2. **gap filling** -- after outlier removal the surviving points are trustworthy, so
     gaps can be filled more generously than during cleaning. Still bounded: a long
     absence is real information about occlusion and should not be invented.
  3. **Savitzky-Golay** -- fits a low-order polynomial over a sliding window, which
     smooths noise while preserving the peaks and turns of genuine movement. A plain
     moving average would flatten exactly the fast direction changes a marmoset makes.

The window is the one real risk: too wide and true motion is smoothed away. So the tool
reports the trade-off rather than hiding it -- **jerk** (frame-to-frame change in
velocity, which is what reads as jitter) should fall a lot, while **path length** (total
distance travelled) should stay close to its original value. If path length collapses,
the window is too wide and real motion is being destroyed.

Usage:
    python tools/smooth_3d.py --project demos/snuh_t2 --source optimized
    python tools/smooth_3d.py --project demos/snuh_t2 --window 11 --max-gap 30
"""
import argparse
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import medfilt, savgol_filter

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('smooth_3d')


def valid_runs(mask, min_length):
    """Yield (start, stop) index pairs of consecutive True runs at least min_length long."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= min_length:
            yield start, stop


def fill_gaps(track, max_gap):
    """Linearly fill gaps up to max_gap frames, per keypoint and axis."""
    out = track.copy()
    filled = 0
    for part in range(track.shape[1]):
        valid = np.isfinite(track[:, part, 0])
        indices = np.flatnonzero(valid)
        if indices.size < 2:
            continue
        for start, end in zip(indices[:-1], indices[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            for axis in range(3):
                out[start + 1:end, part, axis] = np.linspace(
                    track[start, part, axis], track[end, part, axis], gap + 2)[1:-1]
            filled += gap
    return out, filled


def smooth_track(track, window, polyorder, median_kernel, max_gap):
    """Median-filter, fill gaps, then Savitzky-Golay each continuous run."""
    out = track.copy()

    if median_kernel > 1:
        for part in range(out.shape[1]):
            valid = np.isfinite(out[:, part, 0])
            for start, stop in valid_runs(valid, median_kernel):
                for axis in range(3):
                    out[start:stop, part, axis] = medfilt(
                        out[start:stop, part, axis], kernel_size=median_kernel)

    out, filled = fill_gaps(out, max_gap)

    for part in range(out.shape[1]):
        valid = np.isfinite(out[:, part, 0])
        for start, stop in valid_runs(valid, window):
            for axis in range(3):
                out[start:stop, part, axis] = savgol_filter(
                    out[start:stop, part, axis], window_length=window,
                    polyorder=polyorder, mode='interp')
    return out, filled


def motion_stats(track):
    """Jerk (jitter proxy) and path length (real-motion proxy), per keypoint."""
    velocity = np.diff(track, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    jerk = np.linalg.norm(np.diff(velocity, axis=0), axis=-1)
    return (float(np.nanmedian(jerk)) if np.isfinite(jerk).any() else float('nan'),
            float(np.nansum(speed)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--source', default='optimized', choices=['original', 'optimized'])
    ap.add_argument('--window', type=int, default=9,
                    help='Savitzky-Golay window in frames (30 fps -> 9 = 0.3 s)')
    ap.add_argument('--polyorder', type=int, default=2)
    ap.add_argument('--median-kernel', type=int, default=5,
                    help='median filter width for spike removal; must be odd')
    ap.add_argument('--max-gap', type=int, default=30,
                    help='longest gap (frames) to fill before smoothing')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.median_kernel % 2 == 0 or args.window % 2 == 0:
        raise SystemExit('median-kernel and window must both be odd')

    path = REPO / args.project / 'points_3d' / f'{args.source}.h5'
    backup = path.with_name(f'{args.source}_presmooth.h5')
    if not backup.exists():
        shutil.copyfile(path, backup)
        logger.info('kept the unsmoothed version at %s', backup.name)

    with h5py.File(backup, 'r') as fh:
        tracks = {k: np.array(fh[k]) for k in sorted(fh.keys())}

    print()
    print('%-9s %10s %10s %10s %10s %10s' % (
        'track', 'jerk before', 'jerk after', 'reduction', 'path kept', 'coverage'))

    with h5py.File(path, 'w') as fh:
        for name, track in tracks.items():
            jerk_before, path_before = motion_stats(track)
            smoothed, filled = smooth_track(track, args.window, args.polyorder,
                                            args.median_kernel, args.max_gap)
            jerk_after, path_after = motion_stats(smoothed)
            coverage = 100 * float(np.isfinite(smoothed[..., 0]).mean())
            print('%-9s %10.2f %10.2f %9.1fx %9.0f%% %9.1f%%' % (
                name, jerk_before, jerk_after,
                jerk_before / max(jerk_after, 1e-9),
                100 * path_after / max(path_before, 1e-9), coverage))
            fh.create_dataset(name, data=smoothed)

    print()
    print('jerk = median frame-to-frame change in velocity (mm); lower reads as smoother')
    print('path kept = total distance travelled vs before; should stay near 100%,')
    print('            a large drop means real motion is being smoothed away')
    print()


if __name__ == '__main__':
    main()
