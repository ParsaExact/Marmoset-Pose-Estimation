"""Quantify *when* 2D tracking fails, rather than reporting one average.

Watching the labelled videos suggests the model does well when both animals are fully
in frame and apart, and badly when they are close together or partly out of frame. A
single "43% coverage" number hides that entirely, and the two explanations imply
different fixes: proximity failures are an association/occlusion problem, truncation
failures are a field-of-view problem that no amount of training will fix.

Both conditions are measured against things we can compute without any hand labels:

  * **proximity** -- the distance between the two animals in 3D, which is available
    because the reconstruction exists. Using 3D avoids the circularity of measuring
    separation from the same 2D detections whose quality is in question, and it gives
    one number per frame that applies to all five cameras at once.
  * **truncation** -- how close the animal's bounding box sits to the image border.
    A box touching the edge means part of the animal is outside the camera.

Usage:
    python tools/analyze_failure_modes.py --project demos/snuh_t2
"""
import argparse
import logging
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('failure_modes')

PROXIMITY_BINS = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 1e9)]
EDGE_BINS = [(0, 5), (5, 25), (25, 75), (75, 1e9)]


def coverage_of(points, mask, threshold):
    """Fraction of keypoints above threshold within the selected (track, frame) slots."""
    if not mask.any():
        return float('nan'), 0
    scores = np.nan_to_num(points[..., 2], nan=0.0)
    selected = scores[mask]
    return float(np.mean(selected >= threshold)) * 100, int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snuh_t2_viz.yaml')
    ap.add_argument('--points-2d', default='original_raw.h5',
                    help='2D file with raw (unthresholded) scores')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

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

    with h5py.File(Path(config.sub_directory['points_3d']) / 'optimized.h5', 'r') as fh:
        tracks_3d = np.stack([np.array(fh[k]) for k in sorted(fh.keys())])

    # --- proximity -------------------------------------------------------------
    usable = min(n_frames, tracks_3d.shape[1])
    centre = np.nanmedian(tracks_3d[:, :usable], axis=2)          # (tracks, frames, 3)
    separation = np.linalg.norm(centre[0] - centre[1], axis=-1)   # (frames,)

    print()
    print('=' * 74)
    print('2D COVERAGE vs DISTANCE BETWEEN THE TWO ANIMALS (3D separation)')
    print('=' * 74)
    print(f"{'separation (mm)':<20}{'frames':>9}{'coverage@0.5':>15}{'both detected':>16}")
    for low, high in PROXIMITY_BINS:
        in_bin = np.isfinite(separation) & (separation >= low) & (separation < high)
        if not in_bin.any():
            continue
        sel = points_2d[:, :, :usable][:, :, in_bin]
        scores = np.nan_to_num(sel[..., 2], nan=0.0)
        detected = np.isfinite(bboxes[:, :, :usable][:, :, in_bin][..., 0])
        label = f'{low} - {high:.0f}' if high < 1e8 else f'{low}+'
        print(f'{label:<20}{int(in_bin.sum()):>9}'
              f'{float(np.mean(scores >= args.threshold)) * 100:>14.1f}%'
              f'{float(detected.mean()) * 100:>15.1f}%')

    # --- truncation ------------------------------------------------------------
    import cv2
    video = sorted(Path(config.sub_directory['videos_raw']).glob('*.mp4'))[0]
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    x1, y1, x2, y2 = (bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3])
    edge_distance = np.minimum.reduce([x1, y1, width - x2, height - y2])

    print()
    print('=' * 74)
    print('2D COVERAGE vs HOW CLOSE THE ANIMAL IS TO THE IMAGE EDGE')
    print('=' * 74)
    print(f"{'distance to edge':<20}{'instances':>11}{'coverage@0.5':>15}")
    for low, high in EDGE_BINS:
        in_bin = np.isfinite(edge_distance) & (edge_distance >= low) & (edge_distance < high)
        if not in_bin.any():
            continue
        scores = np.nan_to_num(points_2d[..., 2], nan=0.0)
        selected = scores[in_bin]
        label = f'{low} - {high:.0f} px' if high < 1e8 else f'{low}+ px (fully inside)'
        print(f'{label:<20}{int(in_bin.sum()):>11}'
              f'{float(np.mean(selected >= args.threshold)) * 100:>14.1f}%')
    print()
    print('Instances whose box touches the border are partly outside the camera, so the')
    print('missing keypoints are not recoverable from that view by any model.')
    print()


if __name__ == '__main__':
    main()
