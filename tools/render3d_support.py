"""Render the 3D reconstruction coloured by how many cameras actually saw each point.

Triangulation is per keypoint: each one is solved independently from whichever cameras
saw it that frame. So the reconstruction is not one quality but sixteen. The head is
seen by 2.76 cameras on average and lands at 79% 3D coverage; the right foot is seen by
1.27 and lands at 30%. Drawing both in the same colour presents a well-measured point
and a barely-supported one as equally trustworthy.

This colours every keypoint by its camera support in that frame:

    4+ cameras  strong    - heavily over-determined, the most reliable points
    3  cameras  good      - over-determined, one bad view cannot dominate
    2  cameras  minimum   - solvable but unverifiable; no redundancy to detect an error
    <2 cameras  not drawn - not measured at all

Two cameras is the minimum that yields a point, but it gives no way to check it: any
two rays meet somewhere. Only at three or more does disagreement between views become
visible, so the 2-camera points are exactly the ones that carry unnoticed error.

Bones are drawn only where both endpoints were measured, so a limb that was never seen
is absent rather than implied by a line.

Usage:
    python tools/render3d_support.py --project demos/snuh_t2 --still 900 1200 1500
    python tools/render3d_support.py --project demos/snuh_t2 --out results/support.mp4 \
        --start 767 --end 1517
"""
import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger('render3d_support')

# support level -> (colour, label)
SUPPORT_STYLE = {
    4: ('#1a9850', '4+ cameras (strong)'),
    3: ('#91cf60', '3 cameras (good)'),
    2: ('#fdae61', '2 cameras (minimum)'),
}


def support_colour(n_views):
    if n_views >= 4:
        return SUPPORT_STYLE[4][0]
    if n_views == 3:
        return SUPPORT_STYLE[3][0]
    return SUPPORT_STYLE[2][0]


def draw_frame(ax, points_3d, views, chains, index, limits):
    """One frame: points coloured by camera support, bones only between measured points."""
    for track_idx in range(points_3d.shape[0]):
        pts = points_3d[track_idx]
        seen = views[track_idx]
        measured = np.isfinite(pts[:, 0]) & (seen >= 2)

        for chain in chains:
            ids = [index[b] for b in chain if b in index]
            for a, b in zip(ids[:-1], ids[1:]):
                if measured[a] and measured[b]:
                    ax.plot(*zip(pts[a], pts[b]), color='0.55', lw=1.1, zorder=1)

        for part in np.flatnonzero(measured):
            ax.scatter(*pts[part], s=44, color=support_colour(seen[part]),
                       edgecolors='white', linewidths=0.6, depthshade=False, zorder=2)

    (lo, hi) = limits
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='demos/snuh_t2')
    ap.add_argument('--config', default='configs/snuh_t2_viz.yaml')
    ap.add_argument('--source', default='original',
                    help="use 'original' (only measured points) not 'optimized' (filled)")
    ap.add_argument('--still', type=int, nargs='+', default=None,
                    help='render these frames to a PNG instead of a video')
    ap.add_argument('--out', default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--elev', type=float, default=20)
    ap.add_argument('--azim', type=float, default=-60)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    cfg = yaml.safe_load(open(REPO / args.config))
    parts = cfg['animal']['bodyparts']
    chains = cfg['visualization']['skeleton']
    index = {p: i for i, p in enumerate(parts)}

    project = REPO / args.project
    with h5py.File(project / 'points_3d' / f'{args.source}.h5', 'r') as fh:
        points_3d = np.stack([np.array(fh[k]) for k in sorted(fh.keys())])

    # Camera support: how many cameras contributed a keypoint that frame. Taken from the
    # sync-corrected, thresholded 2D -- the same input triangulation consumed.
    from marmopose.utils.data_io import load_points_bboxes_2d_h5
    points_2d, _ = load_points_bboxes_2d_h5(project / 'points_2d' / 'original.h5')
    views = np.isfinite(points_2d[..., 0]).sum(axis=0)          # (tracks, frames, parts)

    n_frames = min(points_3d.shape[1], views.shape[1])
    end = min(args.end or n_frames, n_frames)

    measured = np.isfinite(points_3d[:, :end, :, 0]) & (views[:, :end] >= 2)
    total = measured.size
    strong = int((views[:, :end][measured] >= 3).sum())
    logger.info('measured points %.1f%% of slots; %.1f%% of those have 3+ cameras',
                100 * measured.sum() / total, 100 * strong / max(1, measured.sum()))

    good = points_3d[:, :end][np.isfinite(points_3d[:, :end, :, 0])]
    lo = np.percentile(good, 1, axis=0) - 60
    hi = np.percentile(good, 99, axis=0) + 60
    limits = (lo, hi)

    handles = [Line2D([], [], marker='o', linestyle='', color=c, label=l)
               for c, l in SUPPORT_STYLE.values()]

    if args.still:
        fig = plt.figure(figsize=(5.2 * len(args.still), 5))
        for j, frame_idx in enumerate(args.still):
            ax = fig.add_subplot(1, len(args.still), j + 1, projection='3d')
            draw_frame(ax, points_3d[:, frame_idx], views[:, frame_idx], chains, index, limits)
            n_ok = int((np.isfinite(points_3d[:, frame_idx, :, 0])
                        & (views[:, frame_idx] >= 2)).sum())
            ax.set_title(f'frame {frame_idx}  |  {n_ok} measured keypoints')
            ax.view_init(elev=args.elev, azim=args.azim)
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        out = REPO / (args.out or 'results_for_meeting/_support_still.png')
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        plt.savefig(out, dpi=85)
        logger.info('wrote %s', out)
        return

    # Show the labelled camera views beside the 3D. Without them the reconstruction
    # cannot be judged at all -- there is nothing to compare it against.
    import cv2
    camera_videos = sorted((project / 'videos_labeled_2d').glob('*.mp4'))
    if not camera_videos:
        camera_videos = sorted((project / 'videos_raw').glob('*.mp4'))
    captures = [cv2.VideoCapture(str(p)) for p in camera_videos]
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    logger.info('camera panels: %s', [p.stem for p in camera_videos])

    tmp = REPO / 'work_dirs' / '_support_frames'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    n_cams = len(captures)
    for n, frame_idx in enumerate(range(args.start, end)):
        frames = []
        for cap in captures:
            ok, image = cap.read()
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if ok else None)

        fig = plt.figure(figsize=(10.5, 7.6))
        grid = fig.add_gridspec(2, max(n_cams, 3), height_ratios=[1.0, 3.4],
                                hspace=0.16, wspace=0.03,
                                left=0.01, right=0.99, top=0.95, bottom=0.05)
        for cam_idx, image in enumerate(frames):
            ax = fig.add_subplot(grid[0, cam_idx])
            if image is not None:
                ax.imshow(image)
            ax.set_title(camera_videos[cam_idx].stem, fontsize=8)
            ax.axis('off')

        ax = fig.add_subplot(grid[1, :], projection='3d')
        draw_frame(ax, points_3d[:, frame_idx], views[:, frame_idx], chains, index, limits)
        n_ok = int((np.isfinite(points_3d[:, frame_idx, :, 0])
                    & (views[:, frame_idx] >= 2)).sum())
        ax.set_title(f'measured 3D keypoints: {n_ok}/32   frame {frame_idx}', fontsize=10)
        ax.view_init(elev=args.elev, azim=args.azim)

        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
        # No bbox_inches='tight' -- it crops to content, so every frame comes out a
        # different size and libx264 refuses the sequence.
        plt.savefig(tmp / f'{n:06d}.png', dpi=80)
        plt.close(fig)
        if n % 200 == 0:
            logger.info('rendered %d/%d', n, end - args.start)

    for cap in captures:
        cap.release()

    out = REPO / (args.out or 'results_for_meeting/support.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                    '-i', str(tmp / '%06d.png'),
                    # libx264 + yuv420p needs even dimensions.
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '23', '-pix_fmt', 'yuv420p', str(out)], check=True)
    shutil.rmtree(tmp)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
