"""Convert a hand-labelled DeepLabCut project into the COCO set mmpose fine-tunes on.

Labelling happens in DLC because that is the tool already in use here; training
happens in mmpose because that is what MarmoPose's models are. This bridges them.

Reads every `labeled-data/<video>/CollectedData_<scorer>.h5` (falling back to the
CSV), and writes `train.json` / `val.json` plus a self-contained `images/` directory
in the layout `configs/finetune_pose.py` and `configs/finetune_det.py` expect.

Details that matter:

* **Bounding boxes come from the labelled keypoint extent**, padded, because DLC
  multi-animal projects store keypoints, not boxes. That is also what the detector
  fine-tune trains on, so boxes and keypoints stay consistent by construction.
* **Unlabelled keypoints get visibility 0**, so mmpose's loss ignores them rather
  than being told the animal's elbow is at (0, 0). A partially labelled animal is
  still a useful training instance.
* **The split is by image, not by instance**, so the two animals in one frame never
  straddle train and val -- otherwise val would be scored on a frame the model
  trained on, and the metric would flatter itself.
* **Splitting is grouped by camera** so every camera appears in both halves; with a
  per-camera split you could not tell whether a gain was real or just easier cameras
  landing in val.

Usage:
    python tools/dlc_to_coco.py --dlc-project datasets/to_label --out datasets/snuh_hand
    python tools/dlc_to_coco.py --dlc-project "C:/path/SNUH_body-km-2026" --out datasets/snuh_hand
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('dlc_to_coco')


def load_collected_data(camera_dir):
    """Load a DLC CollectedData table, preferring the h5 over the csv."""
    h5_files = sorted(camera_dir.glob('CollectedData_*.h5'))
    for path in h5_files:
        try:
            return pd.read_hdf(path), path
        except Exception as exc:
            logger.warning('could not read %s (%s); trying csv', path.name, exc)
    csv_files = sorted(camera_dir.glob('CollectedData_*.csv'))
    for path in csv_files:
        return pd.read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1, 2]), path
    return None, None


def image_relpath(index_entry):
    """DLC indexes rows either as a 3-tuple path or as a single string path."""
    if isinstance(index_entry, tuple):
        return '/'.join(str(part) for part in index_entry)
    return str(index_entry).replace('\\', '/')


def extract_instances(table, bodyparts):
    """Yield (image_relpath, {individual: (n_bodyparts, 2) array}) per labelled frame.

    Bodyparts are looked up by name so a DLC project whose ordering differs from the
    config still lines up; a missing bodypart stays NaN rather than shifting the rest.
    """
    levels = table.columns.names
    if 'individuals' in levels:
        individuals = list(dict.fromkeys(table.columns.get_level_values('individuals')))
    else:
        individuals = [None]

    available = set(table.columns.get_level_values('bodyparts'))
    missing = [part for part in bodyparts if part not in available]
    if missing:
        logger.warning('bodyparts absent from the DLC project: %s', missing)

    for row_index, row in table.iterrows():
        per_individual = {}
        for individual in individuals:
            points = np.full((len(bodyparts), 2), np.nan)
            for part_idx, part in enumerate(bodyparts):
                if part not in available:
                    continue
                try:
                    if individual is None:
                        x = row.xs((part, 'x'), level=('bodyparts', 'coords'))
                        y = row.xs((part, 'y'), level=('bodyparts', 'coords'))
                    else:
                        x = row.xs((individual, part, 'x'),
                                   level=('individuals', 'bodyparts', 'coords'))
                        y = row.xs((individual, part, 'y'),
                                   level=('individuals', 'bodyparts', 'coords'))
                except KeyError:
                    continue
                points[part_idx] = (float(np.asarray(x).ravel()[0]),
                                    float(np.asarray(y).ravel()[0]))
            if np.isfinite(points[:, 0]).any():
                per_individual[individual or 'individual1'] = points
        if per_individual:
            yield image_relpath(row_index), per_individual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dlc-project', required=True,
                    help='DLC project root (the dir containing labeled-data/)')
    ap.add_argument('--out', required=True, help='output dataset dir')
    ap.add_argument('--config', default='configs/snu_charuco.yaml',
                    help='config supplying the bodypart order to train on')
    ap.add_argument('--min-keypoints', type=int, default=4,
                    help='drop an instance with fewer labelled keypoints than this')
    ap.add_argument('--bbox-pad', type=float, default=0.15,
                    help='padding around the keypoint extent, as a fraction of its size')
    ap.add_argument('--val-fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    import cv2
    from marmopose.config import Config

    config = Config(config_path=str(REPO / args.config),
                    project=str(REPO / 'demos/snuh_t2'),
                    det_model=str(REPO / 'models/detection_model'),
                    pose_model=str(REPO / 'models/pose_model'),
                    dae_model=str(REPO / 'models/dae_model'))
    bodyparts = list(config.animal['bodyparts'])

    project = Path(args.dlc_project)
    if not project.is_absolute():
        project = REPO / project
    labeled_root = project / 'labeled-data'
    if not labeled_root.is_dir():
        raise SystemExit(f'no labeled-data/ under {project}')

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    skipped_sparse = skipped_missing = 0

    for camera_dir in sorted(p for p in labeled_root.iterdir() if p.is_dir()):
        table, source = load_collected_data(camera_dir)
        if table is None:
            logger.warning('%s: no CollectedData file', camera_dir.name)
            continue

        n_before = len(images)
        for rel_path, per_individual in extract_instances(table, bodyparts):
            image_path = project / rel_path
            if not image_path.exists():
                image_path = camera_dir / Path(rel_path).name
            if not image_path.exists():
                skipped_missing += 1
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped_missing += 1
                continue
            height, width = frame.shape[:2]

            usable = {name: pts for name, pts in per_individual.items()
                      if int(np.isfinite(pts[:, 0]).sum()) >= args.min_keypoints}
            skipped_sparse += len(per_individual) - len(usable)
            if not usable:
                continue

            name = f'{camera_dir.name}_{Path(rel_path).stem}.png'
            shutil.copyfile(image_path, out_dir / 'images' / name)
            image_id = len(images) + 1
            images.append(dict(id=image_id, file_name=f'images/{name}',
                               width=width, height=height, camera=camera_dir.name))

            for points in usable.values():
                visible = np.isfinite(points[:, 0])
                keypoints = np.zeros((len(bodyparts), 3))
                keypoints[visible, :2] = points[visible]
                keypoints[visible, 2] = 2          # COCO: labelled and visible

                xy = points[visible]
                x1, y1 = xy.min(axis=0)
                x2, y2 = xy.max(axis=0)
                pad = args.bbox_pad * max(x2 - x1, y2 - y1, 1.0)
                x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
                x2, y2 = min(width - 1.0, x2 + pad), min(height - 1.0, y2 + pad)
                annotations.append(dict(
                    id=ann_id, image_id=image_id, category_id=1, iscrowd=0,
                    num_keypoints=int(visible.sum()),
                    keypoints=[float(v) for v in keypoints.reshape(-1)],
                    bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    area=float((x2 - x1) * (y2 - y1))))
                ann_id += 1

        logger.info('%s: %d images from %s', camera_dir.name,
                    len(images) - n_before, source.name if source else '?')

    if not images:
        raise SystemExit('no labelled images found')

    # Split within each camera so both halves span all cameras.
    rng = np.random.default_rng(args.seed)
    val_ids = set()
    by_camera = {}
    for image in images:
        by_camera.setdefault(image['camera'], []).append(image['id'])
    for camera, ids in by_camera.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * args.val_fraction)))
        val_ids.update(int(i) for i in ids[:n_val])

    categories = [dict(id=1, name='marmoset', supercategory='animal',
                       keypoints=bodyparts, skeleton=[])]

    for split in ('train', 'val'):
        want_val = split == 'val'
        split_images = [dict(im) for im in images
                        if (im['id'] in val_ids) == want_val]
        keep = {im['id'] for im in split_images}
        for im in split_images:
            im.pop('camera', None)
        split_anns = [a for a in annotations if a['image_id'] in keep]
        path = out_dir / f'{split}.json'
        with open(path, 'w') as fh:
            json.dump(dict(images=split_images, annotations=split_anns,
                           categories=categories), fh)
        logger.info('%s: %d images, %d instances -> %s',
                    split, len(split_images), len(split_anns), path)

    if skipped_sparse:
        logger.info('dropped %d instances with < %d labelled keypoints',
                    skipped_sparse, args.min_keypoints)
    if skipped_missing:
        logger.warning('%d rows had no readable image', skipped_missing)
    logger.info('train with:  python tools/finetune.py configs/finetune_pose.py '
                '--cfg-options data_root=%s/', args.out)


if __name__ == '__main__':
    main()
