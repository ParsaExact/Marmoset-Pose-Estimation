
import os
import logging
import pickle

import numpy as np
import cv2

from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
from marmopose.calibration.cameras import CameraGroup, get_initial_extrinsics
from marmopose.calibration.boards import extract_points, extract_rtvecs, merge_rows

logger = logging.getLogger(__name__)


def _freeze_intrinsics(cam):
    """bundle adjustment ê°€ ì´ ì¹´ë©”ë¼ì˜ rvec/tvec(6ê°œ)ë§Œ ìµœì í™”í•˜ë„ë¡ ë§Œë“¤ê³ ,
    camera matrix ì™€ distortion ì€ ê³ ì •í•œë‹¤."""
    def get_params():
        p = np.zeros(6)
        p[0:3] = cam.get_rotation()
        p[3:6] = cam.get_translation()
        return p

    def set_params(params):
        cam.set_rotation(params[0:3])
        cam.set_translation(params[3:6])

    cam.get_params = get_params
    cam.set_params = set_params


def _per_camera_intrinsics(rows, board, size, max_views=150, min_points=12):
    """Calibrate one camera's intrinsics with the standard cv2.calibrateCamera.

    Partially detected boards are kept. Every corner carries its own id, so the
    object points can be subset to match, and cv2.calibrateCamera accepts views
    of differing length. This matters for ChArUco boards, where a fully complete
    detection is the exception rather than the rule.
    """
    objP_all = board.objPoints.astype(np.float32).reshape(-1, 3)
    objs, imgs = [], []
    for r in rows:
        c = np.array(r['filled']).reshape(-1, 2)
        valid = ~np.isnan(c).any(axis=1)
        if valid.sum() < min_points:
            continue
        objs.append(objP_all[valid].reshape(-1, 1, 3))
        imgs.append(c[valid].astype(np.float32).reshape(-1, 1, 2))
    if len(imgs) < 8:
        raise RuntimeError('usable board detections=%d (need >=8) for intrinsics' % len(imgs))
    step = max(1, len(imgs) // max_views)        # subsample evenly if there are many
    imgs = imgs[::step]
    objs = objs[::step]
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT       # principal point pinned to image centre
             | cv2.CALIB_ZERO_TANGENT_DIST       # no tangential distortion
             | cv2.CALIB_FIX_K3)                 # radial k1, k2 only
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, tuple(size), None, None, flags=flags)
    return K, dist.ravel(), rms, len(imgs)


class FixedCalibrator(Calibrator):
    """ì›ë³¸ Calibrator ì™€ ë™ìž‘ì€ ê°™ë˜, intrinsic ì¶”ì •ë§Œ êµì²´í•˜ê³  ê²°ê³¼ë¥¼
    camera_params_fixed.json ìœ¼ë¡œ ì €ìž¥í•œë‹¤."""

    def __init__(self, config):
        super().__init__(config)
        # ì›ë³¸ camera_params.json ì„ ë®ì–´ì“°ì§€ ì•Šë„ë¡ ì¶œë ¥ íŒŒì¼ëª…ì„ ë¶„ë¦¬
        self.output_path = self.calibration_path / 'camera_params_fixed.json'

    def calibrate(self):
        cam_names, video_list = self.get_video_list(self.calib_video_paths)
        board = self.get_calibration_board(self.config)

        if not self.output_path.exists():
            detected_file = self.calibration_path / 'detected_boards.pickle'
            if detected_file.exists():
                logger.info('Loading detected boards from: %s', detected_file)
                with open(detected_file, 'rb') as f:
                    all_rows = pickle.load(f)
            else:
                logger.info('Detecting boards in videos...')
                all_rows = self.get_rows_videos(video_list, board)
                with open(detected_file, 'wb') as f:
                    pickle.dump(all_rows, f)

            cgroup = CameraGroup.from_names(cam_names, self.config.calibration['fisheye'])
            # ì›ë³¸ì€ set_camera_sizes_videos(ì˜ìƒ ë¼ì´ë¸ŒëŸ¬ë¦¬ ì˜ì¡´)ë¥¼ ì“°ì§€ë§Œ, ì—¬ê¸°ì„œëŠ” cv2 ë¡œ ëŒ€ì²´
            for cam, videos in zip(cgroup.cameras, video_list):
                cap = cv2.VideoCapture(videos[0])
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                cam.set_size((w, h))

            self._calibrate_fixed(cgroup, all_rows, board)
        else:
            logger.info('Fixed calibration already exists in: %s', self.output_path)
            cgroup = CameraGroup.load_from_json(str(self.output_path))

        if self.config.triangulation['user_define_axes']:
            self.update_extrinsics_by_user_define_axes(cgroup)

        cgroup.save_to_json(self.output_path)
        logger.info('Calibration done! Result stored in: %s', self.output_path)

    def _calibrate_fixed(self, cgroup, all_rows, board):
        # 1) ì¹´ë©”ë¼ë³„ robust intrinsic (í‘œì¤€ calibrateCamera)
        logger.info('[fixed] per-camera intrinsics via cv2.calibrateCamera')
        for rows, cam in zip(all_rows, cgroup.cameras):
            K, dist, rms, nv = _per_camera_intrinsics(rows, board, cam.get_size())
            cam.set_camera_matrix(K)
            cam.set_distortion(dist)
            logger.info('    %s: f=%.1f  k1=%.3f k2=%.3f  (views=%d, RMS=%.2fpx)',
                        cam.get_name(), K[0, 0], dist[0], dist[1], nv, rms)

        # 2) extrinsic ì´ˆê¸°í™” (ì›ë³¸ê³¼ ë™ì¼í•œ ë¡œì§ ìž¬ì‚¬ìš©)
        for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
            all_rows[i] = board.estimate_pose_rows(cam, row)
        merged = merge_rows(all_rows)
        imgp, extra = extract_points(merged, board, min_cameras=2)
        rtvecs = extract_rtvecs(merged)
        rvecs, tvecs = get_initial_extrinsics(rtvecs, cgroup.get_names())
        cgroup.set_rotations(rvecs)
        cgroup.set_translations(tvecs)

        # 3) intrinsic ê³ ì • í›„ bundle adjustment (extrinsic + 3Dì ë§Œ ìµœì í™”)
        logger.info('[fixed] bundle adjustment (intrinsics frozen)')
        for cam in cgroup.cameras:
            _freeze_intrinsics(cam)

        if os.environ.get('CALIB_QUICK') == '1':
            ba = dict(n_iters=3, start_mu=15, end_mu=1, max_nfev=40, ftol=1e-3,
                      n_samp_iter=150, n_samp_full=400, error_threshold=1.0, verbose=True)
        else:
            ba = dict(n_iters=10, start_mu=15, end_mu=1, max_nfev=200, ftol=1e-5,
                      n_samp_iter=500, n_samp_full=1000, error_threshold=2.5, verbose=True)

        error = cgroup.bundle_adjust_iter(imgp, extra, **ba)
        cgroup.metadata['error'] = error
        focals = [round(float(c.get_focal_length())) for c in cgroup.cameras]
        logger.info('[fixed] reprojection error = %.2f px  |  focals = %s', error, focals)
