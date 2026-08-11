"""Headless 3D + multi-camera composite renderer.

MarmoPose's own Visualizer3D uses Open3D's interactive window, which blocks when there
is no GUI session, and its composite layout only lays out up to 4 cameras (a 5th writes
outside the canvas). This renders with matplotlib's Agg backend instead: no display
needed, and any number of cameras.

Usage:
    python tools/render3d_composite.py --project demos/snu_s1 --config configs/snu_s1_viz.yaml \
        --source optimized --out results_for_meeting/session1_3D.mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import h5py

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--source', default='optimized', choices=['original', 'optimized'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--elev', type=float, default=22)
    ap.add_argument('--azim0', type=float, default=-60)
    ap.add_argument('--spin', type=float, default=0.25, help='degrees of azimuth per frame')
    ap.add_argument('--slow', type=int, default=1, help='repeat each frame N times')
    args = ap.parse_args()

    proj = (REPO / args.project) if not Path(args.project).is_absolute() else Path(args.project)
    cfg = yaml.safe_load(open(REPO / args.config if not Path(args.config).is_absolute() else args.config))
    bodyparts = cfg['animal']['bodyparts']
    chains = cfg['visualization']['skeleton']

    with h5py.File(proj / 'points_3d' / f'{args.source}.h5') as h:
        p3d = np.stack([h[k][:] for k in sorted(h.keys())])          # (T,F,K,3)
    T, F, K, _ = p3d.shape
    print(f'3D: {T} tracks, {F} frames, {K} keypoints from {args.source}.h5')

    lab2d = sorted((proj / 'videos_labeled_2d').glob('*.mp4'))
    caps = [cv2.VideoCapture(str(p)) for p in lab2d]
    print(f'2D overlay videos: {[p.stem for p in lab2d]}')

    idx = {b: i for i, b in enumerate(bodyparts)}
    chain_idx = [[idx[b] for b in ch if b in idx] for ch in chains]
    track_colors = [(1.0, 0.45, 0.1), (0.1, 0.75, 0.7),
                    (0.45, 0.35, 0.85), (0.85, 0.25, 0.5),
                    (0.35, 0.65, 0.2), (0.9, 0.7, 0.1)]

    good = p3d[~np.isnan(p3d[..., 0])]
    lo, hi = np.percentile(good, 1, axis=0), np.percentile(good, 99, axis=0)
    pad = 0.12 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad
    span = float(np.max(hi - lo))

    tmp = Path('/tmp/_r3d_frames')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    STRIP_H = 260
    for f in range(F):
        frames_2d = []
        for cap in caps:
            ok, fr = cap.read()
            frames_2d.append(fr if ok else None)

        fig = plt.figure(figsize=(13.2, 7.4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        for t in range(T):
            P = p3d[t, f]
            col = track_colors[t % len(track_colors)]
            for ch in chain_idx:
                pts = P[ch]
                m = ~np.isnan(pts[:, 0])
                if m.sum() >= 2:
                    seg = pts[m]
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-', color=col, lw=2.2, alpha=0.9)
            m = ~np.isnan(P[:, 0])
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], P[m, 2], color=col, s=26,
                           edgecolors='k', linewidths=0.4, depthshade=False)
        mid = (lo + hi) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        ax.view_init(elev=args.elev, azim=args.azim0 + args.spin * f)
        ax.set_title(f'3D reconstruction  |  frame {f+1}/{F}  |  {args.source}', fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        img3d = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
        plt.close(fig)

        W = img3d.shape[1]
        n = max(1, len(frames_2d))
        tw = W // n
        strip = np.full((STRIP_H, W, 3), 255, np.uint8)
        for i, fr in enumerate(frames_2d):
            if fr is None:
                continue
            th = STRIP_H
            scale = min(tw / fr.shape[1], th / fr.shape[0])
            rs = cv2.resize(fr, (int(fr.shape[1] * scale), int(fr.shape[0] * scale)))
            y0 = (th - rs.shape[0]) // 2
            x0 = i * tw + (tw - rs.shape[1]) // 2
            strip[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
            cv2.putText(strip, lab2d[i].stem, (i * tw + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        combined = np.vstack([strip, img3d])
        if combined.shape[0] % 2:
            combined = combined[:-1]
        if combined.shape[1] % 2:
            combined = combined[:, :-1]
        for rep in range(args.slow):
            cv2.imwrite(str(tmp / f'{f * args.slow + rep:05d}.png'), combined)
        if (f + 1) % 20 == 0 or f == F - 1:
            print(f'  rendered {f+1}/{F}', flush=True)

    for cap in caps:
        cap.release()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(args.fps),
                    '-i', str(tmp / '%05d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(out)], check=True)
    shutil.rmtree(tmp)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
