"""Pick the frames worth hand-labelling, and pre-fill them with the current predictions.

Labelling is the expensive resource, so spend it where it buys the most:

* **Uncertainty.** Frames the current model is already confident about teach it very
  little. Ranking by low mean keypoint score concentrates effort where the model is
  failing -- which, per FINETUNE_2D.md, is cameras D02-D05 and occluded poses.

* **Not only uncertainty.** Pure uncertainty sampling collapses onto degenerate
  frames: motion blur, an animal half out of frame, total occlusion. Those are often
  unlabelable by a human too, and a training set made of them is unrepresentative.
  So a fraction of the budget is drawn uniformly at random, which keeps the set
  honest and gives an unbiased slice for evaluation. `--uncertain-fraction` controls
  the mix.

* **No near-duplicates.** Consecutive frames at 30 fps are almost the same image, so
  two of them cost double and teach once. `--min-gap` enforces temporal spacing
  within each camera.

* **Balance across cameras.** The budget is split evenly, so the weak cameras get
  represented rather than swamped by whichever camera happens to score lowest.

Output is a DeepLabCut-style `labeled-data/<camera>/` tree plus a
`CollectedData_<scorer>.csv/.h5` holding the model's current predictions as the
starting labels. Labelling then means *correcting* rather than clicking 16 points
from scratch, which is several times faster. Frames the model missed entirely are
written with empty labels, since those are exactly the cases worth adding by hand.

Usage:
    python tools/select_frames_to_label.py --project demos/snuh_t2 --n-frames 400
    # then create a DLC project with the bodyparts it prints, drop the tree in,
    # label, and convert back with tools/dlc_to_coco.py
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('select_frames')


def frame_quality(points, bboxes):
    """Per-frame mean keypoint score and whether anything was detected.

    Args:
        points: (n_tracks, n_frames, n_bodyparts, 3)
        bboxes: (n_tracks, n_frames, 4)

    Returns:
        mean_score: (n_frames,) mean keypoint confidence over detected animals
        n_detected: (n_frames,) how many animals were boxed
    """
    scores = np.nan_to_num(points[..., 2], nan=0.0)          # (tracks, frames, parts)
    n_detected = np.isfinite(bboxes[..., 0]).sum(axis=0)     # (frames,)
    with np.errstate(all='ignore'):
        mean_score = scores.mean(axis=(0, 2))
    return mean_score, n_detected


def pick_indices(mean_score, n_detected, budget, min_gap, uncertain_fraction, rng):
    """Choose frame indices: hardest-first plus a random slice, spaced apart."""
    usable = np.flatnonzero(n_detected > 0)
    if usable.size == 0:
        return np.zeros(0, dtype=int)

    n_uncertain = int(round(budget * uncertain_fraction))
    chosen = []

    def take(candidates, how_many):
        for idx in candidates:
            if len(chosen) >= budget or how_many <= 0:
                break
            if all(abs(int(idx) - c) >= min_gap for c in chosen):
                chosen.append(int(idx))
                how_many -= 1

    # Hardest first: ascending mean score among frames with a detection.
    take(usable[np.argsort(mean_score[usable])], n_uncertain)
    # Then a uniform random slice for representativeness.
    take(rng.permutation(usable), budget - len(chosen))
    return np.array(sorted(chosen), dtype=int)


def build_dlc_frame(points, indices, bodyparts, scorer, individuals, image_names):
    """A DLC multi-animal CollectedData table pre-filled with the predictions."""
    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts, ['x', 'y']],
        names=['scorer', 'individuals', 'bodyparts', 'coords'])
    data = np.full((len(indices), len(columns)), np.nan)

    for row, frame_idx in enumerate(indices):
        col = 0
        for track_idx in range(len(individuals)):
            for part_idx in range(len(bodyparts)):
                x, y, score = points[track_idx, frame_idx, part_idx]
                # NaN score means the model produced nothing here; leave it blank so
                # the annotator adds it rather than being anchored to a bad guess.
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(score):
                    data[row, col] = x
                    data[row, col + 1] = y
                col += 2
    return pd.DataFrame(data, columns=columns, index=image_names)


def write_dlc_config(out_root, bodyparts, individuals, scorer, skeleton, video_paths):
    """Write a DLC config.yaml so the export opens directly as a labelling project.

    Creating a multi-animal project by hand is easy to get subtly wrong -- the scorer
    has to match the CollectedData filenames and the bodypart list has to match the
    order the model was trained on, or the labels convert back misaligned.
    """
    lines = [
        'Task: marmopose_body',
        f'scorer: {scorer}',
        'date: auto',
        'multianimalproject: true',
        'identity: false',
        '',
        f'project_path: {out_root.parent.as_posix()}',
        '',
        'video_sets:',
    ]
    for path in video_paths:
        lines.append(f'  {Path(path).as_posix()}:')
        lines.append('    crop: 0, 0, 0, 0')
    lines.append('individuals:')
    lines += [f'- {name}' for name in individuals]
    lines.append('uniquebodyparts: []')
    lines.append('multianimalbodyparts:')
    lines += [f'- {part}' for part in bodyparts]
    lines.append('bodyparts: MULTI!')
    lines += ['', 'start: 0', 'stop: 1', 'numframes2pick: 50', '']
    lines.append('skeleton:')
    for a, b in skeleton:
        lines += [f'- - {a}', f'  - {b}']
    lines += [
        'skeleton_color: white',
        'pcutoff: 0.6',
        'dotsize: 5',
        'alphavalue: 0.7',
        'colormap: rainbow',
        '',
        'TrainingFraction:',
        '- 0.95',
        'iteration: 0',
        'default_net_type: dlcrnet_ms5',
        'default_augmenter: multi-animal-imgaug',
        'default_track_method: ellipse',
        'snapshotindex: -1',
        'batch_size: 8',
        '',
        'cropping: false',
        'x1: 0', 'x2: 640', 'y1: 277', 'y2: 624',
        '',
        'corner2move2:', '- 50', '- 50', 'move2corner: true',
        '',
    ]
    (out_root.parent / 'config.yaml').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snu_charuco.yaml')
    ap.add_argument('--out', default='datasets/to_label')
    ap.add_argument('--n-frames', type=int, default=400,
                    help='total frames across all cameras')
    ap.add_argument('--min-gap', type=int, default=15,
                    help='minimum frame spacing within a camera (30 fps -> 15 = 0.5 s)')
    ap.add_argument('--uncertain-fraction', type=float, default=0.6,
                    help='share drawn hardest-first; the rest uniformly at random')
    ap.add_argument('--scorer', default='pretrained')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    rng = np.random.default_rng(args.seed)

    from marmopose.config import Config
    from marmopose.utils.data_io import load_points_bboxes_2d_h5

    config = Config(config_path=str(REPO / args.config), project=str(REPO / args.project),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'))
    bodyparts = list(config.animal['bodyparts'])
    n_tracks = config.animal['n_tracks']
    individuals = [f'individual{i + 1}' for i in range(n_tracks)]

    all_points, all_bboxes = load_points_bboxes_2d_h5(
        Path(config.sub_directory['points_2d']) / 'original.h5')
    video_paths = sorted(Path(config.sub_directory['videos_raw']).glob('*.mp4'))
    assert len(video_paths) == all_points.shape[0], 'video/points camera count mismatch'

    out_root = REPO / args.out / 'labeled-data'
    per_camera = max(1, args.n_frames // len(video_paths))
    manifest = []

    for cam_idx, video_path in enumerate(video_paths):
        points, bboxes = all_points[cam_idx], all_bboxes[cam_idx]
        mean_score, n_detected = frame_quality(points, bboxes)
        indices = pick_indices(mean_score, n_detected, per_camera, args.min_gap,
                               args.uncertain_fraction, rng)

        camera = video_path.stem
        cam_dir = out_root / camera
        cam_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        image_names = []
        written = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            name = f'img{int(frame_idx):06d}.png'
            cv2.imwrite(str(cam_dir / name), frame)
            # DLC indexes images by this 3-part relative path.
            image_names.append(('labeled-data', camera, name))
            written.append(int(frame_idx))
            manifest.append(dict(camera=camera, frame=int(frame_idx),
                                 mean_score=round(float(mean_score[frame_idx]), 4),
                                 n_detected=int(n_detected[frame_idx]),
                                 image=f'labeled-data/{camera}/{name}'))
        cap.release()

        if not written:
            logger.warning('%s: nothing selected', camera)
            continue

        index = pd.MultiIndex.from_tuples(image_names)
        table = build_dlc_frame(points, written, bodyparts, args.scorer,
                                individuals, index)
        table.to_csv(cam_dir / f'CollectedData_{args.scorer}.csv')
        table.to_hdf(cam_dir / f'CollectedData_{args.scorer}.h5',
                     key='df_with_missing', mode='w')

        filled = int(np.isfinite(table.to_numpy()).sum() // 2)
        total = len(written) * n_tracks * len(bodyparts)
        logger.info('%s: %d frames, mean score %.3f, %d/%d keypoints pre-filled (%.0f%%)',
                    camera, len(written), float(mean_score[written].mean()),
                    filled, total, 100 * filled / max(1, total))

    manifest_path = REPO / args.out / 'manifest.csv'
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    skeleton = []
    for chain in config.visualization.get('skeleton', []):
        skeleton += list(zip(chain[:-1], chain[1:]))
    write_dlc_config(out_root, bodyparts, individuals, args.scorer, skeleton,
                     [str(p) for p in video_paths])

    logger.info('%d frames total -> %s', len(manifest), out_root)
    logger.info('manifest -> %s', manifest_path)
    logger.info('DLC project ready -> %s', out_root.parent / 'config.yaml')
    print()
    print('Open this as a DLC project and label:')
    print(f'  {out_root.parent / "config.yaml"}')
    print(f'  scorer = "{args.scorer}", {len(bodyparts)} bodyparts, '
          f'{len(individuals)} individuals')
    print('Predictions are pre-filled, so correct rather than annotate from scratch.')
    print()


if __name__ == '__main__':
    main()
