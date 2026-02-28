"""
Figures.py
Generates supplementary figures for DATA.tex that are not produced by
Part1Final.py, part2.py, or part3.py.

Run after Part1Final.py (tracking_results.npz must exist).
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Part1Final import (rgb_to_brightness, apply_threshold,
                        morphological_cleanup, ROIS, THRESHOLDS,
                        MIN_BLOB_SIZE, MAX_BLOB_SIZE, VID_KEYS)


def fig_pipeline_cam1():
    """
    2x3 diagnostic panel for Camera 1, Frame 0 showing all six pipeline steps.
    Output: fig_pipeline_cam1.png
    """
    cam_id = 1
    data = sio.loadmat(f'cam{cam_id}.mat')
    vid = data[VID_KEYS[cam_id]]

    # Frame 0 is the stabilisation reference, so stabilised == original
    frame0 = vid[:, :, :, 0].astype(np.float64)

    y1, y2, x1, x2 = ROIS[cam_id]
    threshold = THRESHOLDS[cam_id]

    roi = frame0[y1:y2, x1:x2, :]
    bright = rgb_to_brightness(roi)
    binary = apply_threshold(bright, threshold)
    cleaned, blob_info = morphological_cleanup(binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE)

    cx, cy = None, None
    if blob_info:
        best = max(blob_info, key=lambda b: b[2])
        cx, cy = best[0], best[1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # (a) Original frame with ROI box
    axes[0, 0].imshow(frame0.astype(np.uint8))
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor='lime', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('(a) Original frame + ROI')
    axes[0, 0].axis('off')

    # (b) ROI crop
    axes[0, 1].imshow(roi.astype(np.uint8))
    axes[0, 1].set_title('(b) ROI crop')
    axes[0, 1].axis('off')

    # (c) Brightness channel
    axes[0, 2].imshow(bright, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(r'(c) Brightness $\max(R,G,B)$')
    axes[0, 2].axis('off')

    # (d) Histogram with threshold line
    axes[1, 0].hist(bright.ravel(), bins=100, color='steelblue', alpha=0.8)
    axes[1, 0].axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                       label=f'$\\theta = {threshold}$')
    axes[1, 0].set_xlabel('Pixel intensity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('(d) Intensity histogram + threshold')
    axes[1, 0].legend()

    # (e) Binary mask
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title('(e) Binary mask')
    axes[1, 1].axis('off')

    # (f) Cleaned mask with centroid
    axes[1, 2].imshow(cleaned, cmap='gray')
    if cx is not None:
        axes[1, 2].plot(cx, cy, 'r+', markersize=14, markeredgewidth=2)
    axes[1, 2].set_title('(f) Cleaned blob + centroid')
    axes[1, 2].axis('off')

    plt.suptitle('Camera 1 — Image Processing Pipeline (Frame 0)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_pipeline_cam1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: fig_pipeline_cam1.png')


if __name__ == '__main__':
    fig_pipeline_cam1()
