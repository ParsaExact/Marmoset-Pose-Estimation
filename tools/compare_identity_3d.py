"""Ask what actually limits 3D reconstruction on this rig: identity, or 2D accuracy.

With calibration at 1.82 px, a triangulated point that disagrees with the cameras that
produced it by 20+ px is not a geometry failure. Two candidates remain:

  * **cross-view identity** -- track 0 in camera 1 is a different animal from track 0 in
    camera 3, so the rays never intersect. MarmoPose assigns identity per camera (dye
    colour, or positional order), which carries no guarantee of agreeing between views.
  * **2D keypoint accuracy** -- the rays are for the right animal but individually off.

They are separable: re-group the detections across views by epipolar geometry
(`reassign_id`) and see whether the residual drops. If it does, identity was the
bottleneck and the fix needs no new labels. If it does not, the limit is 2D accuracy
and only better keypoints will move it.

Usage:
    python tools/compare_identity_3d.py --project demos/snuh_t2 --offsets 0,-1,-2,1,3
"""
import argparse
import logging
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('compare_identity')


def shift_camera(points_cam, offset):
    if offset == 0:
        return points_cam
    out = np.full_like(points_cam, np.nan)
    n_frames = points_cam.shape[1]
    if offset > 0:
        out[:, :n_frames - offset] = points_cam[:, offset:]
    else:
        out[:, -offset:] = points_cam[:, :n_frames + offset]
    return out


def residuals(camera_group, points_3d, points_2d):
    """Median reprojection residual per 3D point against its own observations."""
    flat = points_3d.reshape(-1, 3)
    finite = np.isfinite(flat[:, 0])
    if not finite.any():
        return np.zeros(0)
    n_cams = len(camera_group.cameras)
    projected = np.full((n_cams, flat.shape[0], 2), np.nan)
    projected[:, finite] = camera_group.reproject(flat[finite]).reshape(n_cams, -1, 2)
    observed = points_2d[..., :2].reshape(n_cams, -1, 2)
    per_cam = np.linalg.norm(observed - projected, axis=-1)
    with np.errstate(all='ignore'):
        median = np.nanmedian(per_cam, axis=0)
    return median[finite]


def report(name, values, n_slots):
    if values.size == 0:
        print(f'{name:<34}  no 3D points')
        return
    finite = values[np.isfinite(values)]
    print(f'{name:<34}{finite.size:>9}{100 * finite.size / n_slots:>9.1f}%'
          f'{np.median(finite):>13.2f}{100 * float(np.mean(finite <= 20)):>12.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snu_charuco.yaml')
    ap.add_argument('--offsets', default=None,
                    help='per-camera frame offsets, e.g. 0,-1,-2,1,3')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--max-frames', type=int, default=1200,
                    help='frames to evaluate (epipolar grouping is slow)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from marmopose.config import Config
    from marmopose.calibration.cameras import CameraGroup
    from marmopose.utils.data_io import load_points_bboxes_2d_h5
    from marmopose.processing.triangulation import Reconstructor3D

    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'),
                    n_tracks=2, dae_enable=False, do_optimize=False)

    camera_group = CameraGroup.load_from_json(
        Path(config.sub_directory['calibration']) / 'camera_params.json')
    points_2d, _ = load_points_bboxes_2d_h5(
        Path(config.sub_directory['points_2d']) / 'original.h5')

    scores = points_2d[..., 2]
    points_2d = points_2d.copy()
    points_2d[~(np.isfinite(scores) & (scores >= args.threshold))] = np.nan

    if args.offsets:
        offsets = [int(v) for v in args.offsets.split(',')]
        for cam_idx, offset in enumerate(offsets):
            points_2d[cam_idx] = shift_camera(points_2d[cam_idx], offset)
        logger.info('applied frame offsets %s', offsets)

    points_2d = points_2d[:, :, :args.max_frames]
    n_slots = points_2d.shape[1] * points_2d.shape[2] * points_2d.shape[3]

    reconstructor = Reconstructor3D(config)

    print()
    print('=' * 78)
    print(f'WHAT LIMITS 3D  (calibration 1.82 px, {points_2d.shape[0]} cameras, '
          f'{points_2d.shape[2]} frames)')
    print('=' * 78)
    print(f"{'identity source':<34}{'3D pts':>9}{'of slots':>9}"
          f"{'median resid':>13}{'within 20px':>13}")

    as_is = reconstructor._triangulate_without_reassignment(points_2d)
    report('per-camera (dye / positional)', residuals(camera_group, as_is, points_2d),
           n_slots)

    regrouped, reassigned_2d = reconstructor._triangulate_with_reassignment(points_2d)
    report('epipolar re-grouping', residuals(camera_group, regrouped, reassigned_2d),
           n_slots)
    print()


if __name__ == '__main__':
    main()
