"""Run 2D detection + pose over every video in a project.

`run_local.py` covers the 3D and rendering steps but not this one, which previously had
to be invoked as an inline Python snippet. This is the same call, as a command.

The project must contain `videos_raw/*.mp4`. Results are written to
`points_2d/original.h5`, keyed by video name, which is what the 3D stage reads.

Note on `--keypoint-threshold`: keypoints scoring below it are stored as NaN, so the
choice is baked into the file. The pipeline default is 0.5. Pass 0 to keep the raw scores
when the output will be analysed (`tools/diagnose_2d.py`, `tools/find_best_segment.py`)
rather than fed straight to triangulation, since a threshold cannot be raised later but
can always be applied afterwards.

Usage:
    python tools/predict_2d.py --project demos/snuh_t2 --config configs/snu_charuco.yaml
    python tools/predict_2d.py --project demos/snuh_t2 --keypoint-threshold 0 --batch-size 8
"""
import argparse
import logging
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True, help='e.g. demos/snuh_t2')
    ap.add_argument('--config', default='configs/snu_charuco.yaml')
    ap.add_argument('--det-model', default='models/detection_model')
    ap.add_argument('--pose-model', default='models/pose_model')
    ap.add_argument('--n-tracks', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--keypoint-threshold', type=float, default=None,
                    help='override the config threshold; 0 keeps raw scores')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from marmopose.config import Config
    from marmopose.processing.prediction import Predictor

    overrides = dict(
        project=str(REPO / args.project),
        det_model=str(REPO / args.det_model),
        pose_model=str(REPO / args.pose_model),
        dae_model=str(REPO / 'models/dae_model'),
        n_tracks=args.n_tracks,
        dae_enable=False,
        do_optimize=False,
    )
    if args.keypoint_threshold is not None:
        overrides['keypoint'] = args.keypoint_threshold

    config = Config(config_path=str(REPO / args.config), **overrides)
    if args.keypoint_threshold is not None:
        assert config.threshold['keypoint'] == args.keypoint_threshold, \
            'keypoint threshold override did not take effect'

    videos = sorted((Path(config.sub_directory['videos_raw'])).glob('*.mp4'))
    if not videos:
        raise SystemExit(f"no videos in {config.sub_directory['videos_raw']}")
    logging.info('%d videos, keypoint threshold %.2f, batch %d',
                 len(videos), config.threshold['keypoint'], args.batch_size)

    start = time.time()
    Predictor(config, batch_size=args.batch_size).predict()
    elapsed = time.time() - start
    logging.info('done in %.1f min -> %s/original.h5',
                 elapsed / 60, config.sub_directory['points_2d'])


if __name__ == '__main__':
    main()
