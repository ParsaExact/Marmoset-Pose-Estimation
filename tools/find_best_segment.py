"""Find the stretch of footage where 2D tracking is at its best.

The failure analysis showed 2D quality is not uniform over a session: it collapses when
an animal is partly out of frame (26% coverage vs 64-71% fully inside) and degrades when
the two animals are close (40% vs 49%). Averaged over a whole clip those episodes drag
the number down and, more importantly, they are what makes a demonstration video look
broken.

So rather than judging the system on an arbitrary interval, this finds the contiguous
window where the conditions are actually favourable, and reports *why* it is favourable
so the choice is not cherry-picking dressed up as a result. A best-case window answers a
different and legitimate question -- "what does this system do when the recording
cooperates?" -- as long as it is presented as exactly that, alongside the session
average.

Scored per frame, all measurable without hand labels:

  * keypoint coverage at the operating threshold (the thing we care about)
  * whether both animals were detected at all
  * distance of each box to the image border (truncation)
  * 3D separation between the animals (proximity)
  * identity swaps, detected by checking whether the two tracks' centres would be
    better explained by exchanging their labels between consecutive frames

Usage:
    python tools/find_best_segment.py --project demos/snuh_t2 --window-seconds 25
"""
import argparse
import logging
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('best_segment')


def box_centres(bboxes):
    """(n_cams, n_tracks, n_frames, 2) centres, NaN where undetected."""
    return np.stack([(bboxes[..., 0] + bboxes[..., 2]) / 2,
                     (bboxes[..., 1] + bboxes[..., 3]) / 2], axis=-1)


def swap_flags(centres):
    """Per-frame flag: would exchanging the two track labels explain motion better?

    A genuine identity swap makes each track appear to jump to where the *other* animal
    was. Comparing the straight assignment cost against the exchanged one detects that
    without needing ground truth.
    """
    n_cams, n_tracks, n_frames, _ = centres.shape
    if n_tracks < 2:
        return np.zeros(n_frames, dtype=bool)
    previous, current = centres[:, :, :-1], centres[:, :, 1:]
    straight = (np.linalg.norm(current[:, 0] - previous[:, 0], axis=-1) +
                np.linalg.norm(current[:, 1] - previous[:, 1], axis=-1))
    exchanged = (np.linalg.norm(current[:, 0] - previous[:, 1], axis=-1) +
                 np.linalg.norm(current[:, 1] - previous[:, 0], axis=-1))
    swapped = np.isfinite(straight) & np.isfinite(exchanged) & (exchanged < straight)
    out = np.zeros(n_frames, dtype=bool)
    out[1:] = swapped.any(axis=0)          # a swap in any camera counts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snuh_t2_viz.yaml')
    ap.add_argument('--points-2d', default='original_raw.h5')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--window-seconds', type=float, default=25.0)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    import cv2
    import h5py
    from marmopose.config import Config
    from marmopose.utils.data_io import load_points_bboxes_2d_h5

    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'))

    points_2d, bboxes = load_points_bboxes_2d_h5(
        Path(config.sub_directory['points_2d']) / args.points_2d)
    n_cams, n_tracks, n_frames, n_parts, _ = points_2d.shape

    video = sorted(Path(config.sub_directory['videos_raw']).glob('*.mp4'))[0]
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scores = np.nan_to_num(points_2d[..., 2], nan=0.0)
    coverage = (scores >= args.threshold).mean(axis=(0, 1, 3))       # per frame
    detected = np.isfinite(bboxes[..., 0])
    both_detected = detected.all(axis=1).mean(axis=0)                 # per frame

    edge = np.minimum.reduce([bboxes[..., 0], bboxes[..., 1],
                              width - bboxes[..., 2], height - bboxes[..., 3]])
    with np.errstate(all='ignore'):
        edge_min = np.nanmin(np.where(detected, edge, np.nan), axis=(0, 1))

    # Separation is measured in 3D when a reconstruction exists, because that is one
    # number per frame valid for every camera. Without one, fall back to the 2D distance
    # between the two boxes, averaged over the cameras that saw both -- coarser, and in
    # pixels rather than mm, but it still ranks frames sensibly.
    separation = np.full(n_frames, np.nan)
    separation_unit = 'mm'
    points_3d_path = Path(config.sub_directory['points_3d']) / 'optimized.h5'
    if points_3d_path.exists():
        with h5py.File(points_3d_path, 'r') as fh:
            tracks_3d = np.stack([np.array(fh[k]) for k in sorted(fh.keys())])
        usable = min(n_frames, tracks_3d.shape[1])
        centre_3d = np.nanmedian(tracks_3d[:, :usable], axis=2)
        separation[:usable] = np.linalg.norm(centre_3d[0] - centre_3d[1], axis=-1)
    else:
        logger.info('no 3D reconstruction found; using 2D box separation in pixels')
        separation_unit = 'px'
        centres = box_centres(bboxes)                       # (cams, tracks, frames, 2)
        if centres.shape[1] >= 2:
            per_cam = np.linalg.norm(centres[:, 0] - centres[:, 1], axis=-1)
            with np.errstate(all='ignore'):
                separation = np.nanmean(per_cam, axis=0)

    swaps = swap_flags(box_centres(bboxes))

    window = int(round(args.window_seconds * args.fps))
    if window >= n_frames:
        raise SystemExit('window longer than the clip')

    kernel = np.ones(window) / window
    smooth_coverage = np.convolve(coverage, kernel, mode='valid')

    order = np.argsort(-smooth_coverage)
    chosen, used = [], []
    for start in order:
        if any(abs(int(start) - s) < window for s in used):
            continue          # keep the reported windows non-overlapping
        chosen.append(int(start))
        used.append(int(start))
        if len(chosen) >= args.top:
            break

    print()
    print('=' * 96)
    print(f'BEST {args.window_seconds:.0f}s WINDOWS  (session average coverage '
          f'{100 * coverage.mean():.1f}%)')
    print('=' * 96)
    print(f"{'start':>9}{'coverage':>11}{'both det':>10}{'separation':>12}"
          f"{'edge dist':>11}{'swaps/s':>9}")
    for start in chosen:
        stop = start + window
        sl = slice(start, stop)
        print(f'{start / args.fps:>7.1f}s{100 * coverage[sl].mean():>10.1f}%'
              f'{100 * both_detected[sl].mean():>9.1f}%'
              f'{np.nanmedian(separation[sl]):>10.0f}{separation_unit}'
              f'{np.nanmedian(edge_min[sl]):>9.0f}px'
              f'{swaps[sl].sum() / args.window_seconds:>9.2f}')

    best = chosen[0]
    print()
    print(f'best window: {best / args.fps:.1f}s - {(best + window) / args.fps:.1f}s '
          f'(frames {best}-{best + window})')
    print(f'coverage there {100 * coverage[best:best + window].mean():.1f}% '
          f'vs {100 * coverage.mean():.1f}% session average')
    print()
    print('Report a best-case window as a best case, next to the session average --')
    print('it shows the ceiling the method reaches when the recording cooperates.')
    print()


if __name__ == '__main__':
    main()
