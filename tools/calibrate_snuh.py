"""Calibrate the SNUH rig, correcting the camera time offsets in the export.

Why this exists
---------------
Running the ChArUco-aware `FixedCalibrator` on this rig lands at **40.0 px**
reprojection error, and the per-pair breakdown shows why:

    pairs among {D01, D03, D04}   8.3 - 9.5 px
    every pair touching D02       12.8 - 18.5 px
    every pair touching D05       10.8 - 18.5 px

Geometry is not the problem -- two cameras are shifted in time relative to the
rest, so the board they each report is the board at a *different instant*, and no
amount of bundle adjustment can reconcile that. `results_for_meeting/SUMMARY.md`
found the same thing on the earlier export and measured D02 +27, D05 -27 frames
(~0.9 s at 30 fps).

Board detection is the expensive step (~45 min for 17,850 frames) and is cached in
`detected_boards.pickle`, so re-running with different offsets is cheap: this script
reuses the cache and only redoes pose estimation, extrinsic init and bundle
adjustment.

Usage:
    # try the offsets SUMMARY.md measured
    python tools/calibrate_snuh.py --offsets 0,27,0,0,-27

    # search for them instead (slow: one bundle adjustment per candidate)
    python tools/calibrate_snuh.py --search 30 --search-step 3

Offsets are in frames, one per camera in sorted-name order, positive meaning the
camera runs *ahead* of the reference.
"""
import argparse
import copy
import logging
import os
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('calibrate_snuh')


def shift_rows(rows, offset):
    """Relabel each detection's frame index so the camera lines up with the reference."""
    if offset == 0:
        return rows
    shifted = []
    for row in rows:
        row = copy.copy(row)
        video_idx, frame_idx = row['framenum']
        row['framenum'] = (video_idx, frame_idx - offset)
        shifted.append(row)
    return shifted


def calibrate_with_offsets(config, all_rows, offsets, quick=False):
    """Full extrinsic solve for one set of offsets. Returns (error, camera_group)."""
    from marmopose.calibration.calibration import Calibrator
    from marmopose.calibration.cameras import CameraGroup, get_initial_extrinsics
    from marmopose.calibration.boards import extract_points, extract_rtvecs, merge_rows
    from marmopose.calibration.calibrate_fixed import (_freeze_intrinsics,
                                                       _per_camera_intrinsics)
    import cv2

    calibrator = Calibrator(config)
    cam_names, video_list = calibrator.get_video_list(calibrator.calib_video_paths)
    board = calibrator.get_calibration_board(config)

    cgroup = CameraGroup.from_names(cam_names, config.calibration['fisheye'])
    for cam, videos in zip(cgroup.cameras, video_list):
        cap = cv2.VideoCapture(videos[0])
        cam.set_size((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()

    # Intrinsics are per-camera and independent of inter-camera timing, so they are
    # unaffected by the offsets.
    rows = [shift_rows(r, o) for r, o in zip(all_rows, offsets)]
    for cam_rows, cam in zip(rows, cgroup.cameras):
        K, dist, rms, n_views = _per_camera_intrinsics(cam_rows, board, cam.get_size())
        cam.set_camera_matrix(K)
        cam.set_distortion(dist)

    for i, (cam_rows, cam) in enumerate(zip(rows, cgroup.cameras)):
        rows[i] = board.estimate_pose_rows(cam, cam_rows)

    merged = merge_rows(rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)
    rvecs, tvecs = get_initial_extrinsics(extract_rtvecs(merged), cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    for cam in cgroup.cameras:
        _freeze_intrinsics(cam)

    if quick:
        ba = dict(n_iters=2, start_mu=15, end_mu=2, max_nfev=30, ftol=1e-3,
                  n_samp_iter=100, n_samp_full=300, error_threshold=1.0, verbose=False)
    else:
        ba = dict(n_iters=10, start_mu=15, end_mu=1, max_nfev=200, ftol=1e-5,
                  n_samp_iter=500, n_samp_full=1000, error_threshold=2.5, verbose=False)

    error = cgroup.bundle_adjust_iter(imgp, extra, **ba)
    cgroup.metadata['error'] = float(error)
    cgroup.metadata['frame_offsets'] = list(offsets)
    return float(error), cgroup


def probe_sync(config, all_rows, max_offset):
    """Recover per-camera frame offsets from board-pose consistency, without any BA.

    If two cameras are synchronised, the rigid transform between them is constant, so
    the relative board rotation `R_c(f) @ R_ref(f)^T` is the same in every frame they
    both see. Time-shifted cameras see the board in different poses, and that relative
    rotation scatters. Minimising its dispersion recovers the offset directly.

    Costs one pose estimation pass instead of a bundle adjustment per candidate, which
    is what makes searching practical at all.
    """
    import cv2
    from marmopose.calibration.calibration import Calibrator
    from marmopose.calibration.cameras import CameraGroup
    from marmopose.calibration.calibrate_fixed import _per_camera_intrinsics

    calibrator = Calibrator(config)
    cam_names, video_list = calibrator.get_video_list(calibrator.calib_video_paths)
    board = calibrator.get_calibration_board(config)
    cgroup = CameraGroup.from_names(cam_names, config.calibration['fisheye'])

    poses = []
    for cam_rows, cam, videos in zip(all_rows, cgroup.cameras, video_list):
        cap = cv2.VideoCapture(videos[0])
        cam.set_size((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
        K, dist, _, _ = _per_camera_intrinsics(cam_rows, board, cam.get_size())
        cam.set_camera_matrix(K)
        cam.set_distortion(dist)

        by_frame = {}
        for row in board.estimate_pose_rows(cam, copy.deepcopy(cam_rows)):
            if row.get('rvec') is None:
                continue
            rvec = np.asarray(row['rvec']).ravel()
            if rvec.size == 3 and np.all(np.isfinite(rvec)):
                by_frame[row['framenum'][1]] = cv2.Rodrigues(rvec)[0]
        poses.append(by_frame)
        logger.info('probe: %s has %d board poses', cam.get_name(), len(by_frame))

    def dispersion(ref, other, offset):
        rels = []
        for frame_idx, R_ref in ref.items():
            R_other = other.get(frame_idx + offset)
            if R_other is None:
                continue
            rels.append(cv2.Rodrigues(R_other @ R_ref.T)[0].ravel())
        if len(rels) < 20:
            return np.inf, len(rels)
        rels = np.asarray(rels)
        centre = np.median(rels, axis=0)
        angles = np.degrees(np.linalg.norm(rels - centre, axis=1))
        return float(np.median(angles)), len(rels)

    offsets = [0] * len(poses)
    for cam_idx in range(1, len(poses)):
        scores = []
        for offset in range(-max_offset, max_offset + 1):
            score, n = dispersion(poses[0], poses[cam_idx], offset)
            scores.append((score, offset, n))
        scores.sort()
        best_score, best_offset, best_n = scores[0]
        runner_up = next((s for s, o, _ in scores[1:] if abs(o - best_offset) > 2),
                         float('inf'))
        offsets[cam_idx] = best_offset
        logger.info('probe: cam%d offset %+4d  dispersion %.2f deg over %d frames '
                    '(next distinct candidate %.2f deg)',
                    cam_idx, best_offset, best_score, best_n, runner_up)
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snu_charuco.yaml')
    ap.add_argument('--offsets', default=None,
                    help='comma-separated per-camera frame offsets, e.g. 0,27,0,0,-27')
    ap.add_argument('--search', type=int, default=0, metavar='N',
                    help='search each camera offset over [-N, N] (greedy, one bundle '
                         'adjustment per candidate -- slow)')
    ap.add_argument('--search-step', type=int, default=3)
    ap.add_argument('--probe-sync', type=int, default=0, metavar='N',
                    help='recover offsets over [-N, N] from board-pose consistency '
                         '(seconds, no bundle adjustment) and use them')
    ap.add_argument('--probe-only', action='store_true', default=False,
                    help='report probed offsets and exit without calibrating')
    ap.add_argument('--out', default='camera_params.json',
                    help='filename written into the calibration dir')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')

    from marmopose.config import Config
    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'))

    calib_dir = Path(config.sub_directory['calibration'])
    cache = calib_dir / 'detected_boards.pickle'
    if not cache.exists():
        raise SystemExit(f'no cached detections at {cache}; run FixedCalibrator first')
    with open(cache, 'rb') as fh:
        all_rows = pickle.load(fh)
    n_cams = len(all_rows)
    logger.info('loaded detections for %d cameras: %s',
                n_cams, [len(r) for r in all_rows])

    if args.offsets:
        offsets = [int(v) for v in args.offsets.split(',')]
        assert len(offsets) == n_cams, f'need {n_cams} offsets'
    else:
        offsets = [0] * n_cams

    if args.probe_sync:
        offsets = probe_sync(config, all_rows, args.probe_sync)
        logger.info('probed offsets: %s', offsets)
        if args.probe_only:
            return

    if args.search:
        # Only needed as the starting point for the greedy search; skipping it when
        # offsets are given outright avoids a second full bundle adjustment.
        best_cost, _ = calibrate_with_offsets(config, all_rows, offsets, quick=True)
        logger.info('starting offsets %s -> %.2f px', offsets, best_cost)
        for cam_idx in range(1, n_cams):
            candidates = range(-args.search, args.search + 1, args.search_step)
            for candidate in candidates:
                if candidate == offsets[cam_idx]:
                    continue
                trial = list(offsets)
                trial[cam_idx] = candidate
                try:
                    cost, _ = calibrate_with_offsets(config, all_rows, trial, quick=True)
                except Exception as exc:            # a bad offset can starve a camera
                    logger.debug('  cam%d %+d failed: %s', cam_idx, candidate, exc)
                    continue
                logger.info('  cam%d offset %+4d -> %.2f px', cam_idx, candidate, cost)
                if cost < best_cost * 0.97:
                    best_cost, offsets[cam_idx] = cost, candidate
            logger.info('cam%d chose %+d (%.2f px)', cam_idx, offsets[cam_idx], best_cost)
        logger.info('searched offsets: %s', offsets)

    error, cgroup = calibrate_with_offsets(config, all_rows, offsets, quick=False)
    focals = [round(float(c.get_focal_length())) for c in cgroup.cameras]
    logger.info('FINAL offsets %s | reprojection error %.2f px | focals %s',
                offsets, error, focals)

    if config.triangulation['user_define_axes']:
        from marmopose.calibration.calibration import Calibrator
        Calibrator(config).update_extrinsics_by_user_define_axes(cgroup)

    out_path = calib_dir / args.out
    cgroup.save_to_json(out_path)
    logger.info('saved -> %s', out_path)


if __name__ == '__main__':
    main()
