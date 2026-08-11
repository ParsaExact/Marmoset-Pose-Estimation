"""
Run the MarmoPose pipeline locally on macOS (CPU, no CUDA, no mmcv).

Covers everything except MarmoPose's own 2D detector: 3D reconstruction,
skeleton optimization, and 2D/3D video rendering. Supply 2D points yourself
(e.g. from DeepLabCut) as <project>/points_2d/original.h5.

Usage:
    .venv/bin/python run_local.py --project demos/pair_mac
    .venv/bin/python run_local.py --project demos/pair_mac --steps 3d,video2d,video3d
"""
import argparse
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True, help='project dir, e.g. demos/pair_mac')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--steps', default='3d,video2d,video3d',
                    help='comma list: 3d, video2d, video3d')
    ap.add_argument('--n-tracks', type=int, default=2)
    ap.add_argument('--optimize', action='store_true', default=True)
    ap.add_argument('--no-optimize', dest='optimize', action='store_false')
    ap.add_argument('--dae', action='store_true', default=False,
                    help='enable the denoising autoencoder (16 keypoints only)')
    ap.add_argument('--reassign-id', action='store_true', default=False,
                    help='group detections across views by epipolar geometry. Required when '
                         '2D was run with dye_identity: false, since track ids are then '
                         'per-camera positional rather than a real identity.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log = logging.getLogger('run_local')

    from marmopose.config import Config
    config = Config(
        config_path=str(REPO / args.config),
        project=str(REPO / args.project),
        det_model=str(REPO / 'models/detection_model'),
        pose_model=str(REPO / 'models/pose_model'),
        dae_model=str(REPO / 'models/dae_model'),
        n_tracks=args.n_tracks,
        dae_enable=args.dae,
        do_optimize=args.optimize,
    )
    # default_6pt.yaml leaves this key empty, which YAML parses as None
    if config.optimization.get('bodypart_distance_weak') is None:
        config.optimization['bodypart_distance_weak'] = {}
        log.info('bodypart_distance_weak was empty -> using {}')

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]

    if '3d' in steps:
        from marmopose.processing.triangulation import Reconstructor3D
        log.info('=== 3D reconstruction (reassign_id=%s) ===', args.reassign_id)
        Reconstructor3D(config).triangulate(reassign_id=args.reassign_id)

    if 'video2d' in steps:
        from marmopose.visualization.display_2d import Visualizer2D
        log.info('=== 2D labeled videos ===')
        Visualizer2D(config).generate_videos_2d()

    if 'video3d' in steps:
        from marmopose.visualization.display_3d import Visualizer3D
        log.info('=== 3D video ===')
        source = 'optimized' if args.optimize else 'original'
        Visualizer3D(config).generate_video_3d(source_3d=source, video_type='composite')

    log.info('Done. Outputs under %s', REPO / args.project)


if __name__ == '__main__':
    main()
