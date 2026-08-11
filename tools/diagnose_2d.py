"""Measure where 2D keypoint coverage is lost, per video.

MarmoPose reports "% valid keypoints" after thresholding keypoint scores at
`threshold.keypoint` (0.5 by default). That single number folds together two
independent failures:

  1. the detector never boxed the animal, so its keypoints are missing outright
  2. the animal was boxed, but the pose model was not confident about the point

Fixing them needs different work, so this script reports them separately. It
also runs with the keypoint threshold forced to 0, keeping the raw scores, so
coverage can be swept across thresholds instead of being read at one arbitrary
cut -- otherwise "coverage" can be raised for free by lowering the threshold,
which improves nothing.

Bounding-box size is reported alongside, because a top-down pose model's accuracy
tracks how many pixels the animal actually occupies. If the target domain images
the animal smaller than the training domain, that is a resolution problem, not a
domain-shift problem, and fine-tuning will not fix it.

Usage:
    python tools/diagnose_2d.py --videos demos/pair/videos_raw/bak-1-2.mp4 --frames 120
    python tools/diagnose_2d.py --videos "SNUH_short-km-2026-06-19/videos/*.mp4" --frames 100
"""
import argparse
import glob
import logging
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent

THRESHOLDS = (0.3, 0.5, 0.7)


def sample_frame_indices(n_frames: int, n_wanted: int) -> np.ndarray:
    """Evenly spaced frame indices, so a long video isn't judged on its first seconds."""
    if n_frames <= n_wanted:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, n_wanted).astype(int)


def measure_video(predictor, video_path: str, n_wanted: int, batch_size: int, n_tracks: int):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = sample_frame_indices(n_frames, n_wanted)

    scores, box_areas, n_detected = [], [], []

    batch = []
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        batch.append(frame)
        if len(batch) == batch_size:
            _accumulate(predictor, batch, scores, box_areas, n_detected)
            batch = []
    if batch:
        _accumulate(predictor, batch, scores, box_areas, n_detected)
    cap.release()

    scores = np.concatenate(scores) if scores else np.zeros((0, 0))
    return {
        'video': Path(video_path).name,
        'resolution': f'{width}x{height}',
        'frames_sampled': len(n_detected),
        'n_tracks': n_tracks,
        'scores': scores,
        'box_areas': np.array(box_areas, dtype=float),
        'n_detected': np.array(n_detected, dtype=float),
        'frame_area': float(width * height),
    }


def _accumulate(predictor, frames, scores, box_areas, n_detected):
    """Run one batch and collect raw scores, box areas and detection counts.

    `predict_image_batch` returns arrays shaped (batch, n_tracks, n_bodyparts, 3)
    and (batch, n_tracks, 4), with NaN in the slots of tracks it did not detect.
    """
    points, bboxes = predictor.predict_image_batch(frames)
    for frame_points, frame_boxes in zip(points, bboxes):
        detected = ~np.isnan(frame_boxes[:, 0])
        n_detected.append(int(detected.sum()))
        for box in frame_boxes[detected]:
            box_areas.append(abs((box[2] - box[0]) * (box[3] - box[1])))
        # NaN marks an undetected track; keep it as a score of 0 so coverage is
        # measured against every slot that should have been filled.
        frame_scores = np.nan_to_num(frame_points[..., 2], nan=0.0)
        scores.append(frame_scores.reshape(1, -1))


def report(results):
    print()
    print('=' * 108)
    print('DETECTION -- how often each animal was boxed at all')
    print('=' * 108)
    print(f"{'video':<22}{'res':<11}{'frames':>7}{'mean animals':>14}"
          f"{'all present':>13}{'median box':>12}{'% of frame':>12}")
    for r in results:
        expected = r['n_tracks']
        all_present = float(np.mean(r['n_detected'] >= expected)) * 100
        med_area = float(np.median(r['box_areas'])) if r['box_areas'].size else float('nan')
        print(f"{r['video']:<22}{r['resolution']:<11}{r['frames_sampled']:>7}"
              f"{np.mean(r['n_detected']):>10.2f}/{expected:<3}"
              f"{all_present:>12.1f}%{np.sqrt(med_area):>10.0f}px"
              f"{100 * med_area / r['frame_area']:>11.2f}%")

    print()
    print('=' * 108)
    print('KEYPOINTS -- coverage over every slot (undetected animals counted as missing)')
    print('=' * 108)
    header = f"{'video':<22}" + ''.join(f'{f"cover@{t}":>12}' for t in THRESHOLDS)
    print(header + f"{'mean score':>12}{'median':>10}")
    for r in results:
        s = r['scores']
        row = f"{r['video']:<22}"
        for t in THRESHOLDS:
            row += f'{100 * float(np.mean(s >= t)):>11.1f}%'
        print(row + f'{float(np.mean(s)):>12.3f}{float(np.median(s)):>10.3f}')

    print()
    print('=' * 108)
    print('POSE MODEL ALONE -- coverage among keypoints of animals that WERE detected')
    print('=' * 108)
    print(f"{'video':<22}" + ''.join(f'{f"cover@{t}":>12}' for t in THRESHOLDS)
          + f"{'mean score':>12}")
    for r in results:
        s = r['scores']
        detected_only = s[s > 0]  # slots of undetected tracks were zero-filled
        row = f"{r['video']:<22}"
        if detected_only.size == 0:
            print(row + '   no detections')
            continue
        for t in THRESHOLDS:
            row += f'{100 * float(np.mean(detected_only >= t)):>11.1f}%'
        print(row + f'{float(np.mean(detected_only)):>12.3f}')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='+', required=True,
                    help='video paths or globs')
    ap.add_argument('--config', default='configs/snu_s1_nodye.yaml',
                    help='config to take bodyparts/thresholds from; should be '
                         'class-agnostic (dye_identity: false) for a fair comparison')
    ap.add_argument('--frames', type=int, default=100, help='frames sampled per video')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-tracks', type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    paths = []
    for pattern in args.videos:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    if not paths:
        raise SystemExit('no videos matched')

    from marmopose.config import Config
    from marmopose.processing.prediction import Predictor

    scratch = REPO / 'demos' / '_diag'
    (scratch / 'videos_raw').mkdir(parents=True, exist_ok=True)
    (scratch / 'calibration').mkdir(parents=True, exist_ok=True)

    config = Config(
        config_path=str(REPO / args.config),
        project=str(scratch),
        det_model=str(REPO / 'models/detection_model'),
        pose_model=str(REPO / 'models/pose_model'),
        dae_model=str(REPO / 'models/dae_model'),
        n_tracks=args.n_tracks,
        dae_enable=False,
        do_optimize=False,
        keypoint=0.0,  # keep raw scores so coverage can be swept, not fixed at 0.5
    )
    assert config.threshold['keypoint'] == 0.0, 'keypoint threshold override failed'

    predictor = Predictor(config, batch_size=args.batch_size)

    results = []
    for path in paths:
        print(f'measuring {path} ...', flush=True)
        results.append(measure_video(predictor, path, args.frames,
                                     args.batch_size, args.n_tracks))
    report(results)


if __name__ == '__main__':
    main()
