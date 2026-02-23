"""
Animate the tracking pipeline for a chosen camera.

Shows a 4-panel (1×4) animation flipping through every frame:
    Panel 1: Stabilised RGB frame with ROI rectangle
    Panel 2: Brightness channel (cropped ROI)
    Panel 3: Binary threshold (white threshold)
    Panel 4: Cleaned blobs + red dot at detected centroid

Usage:
    python animate_pipeline.py          # default Camera 1
    python animate_pipeline.py 2        # Camera 2
    python animate_pipeline.py 3        # Camera 3
"""

import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from chat1 import (
    VID_KEYS, DATA_PATH, ROIS, THRESHOLDS, CHANNEL_MODE,
    MIN_BLOB_SIZE, MAX_BLOB_SIZE,
    rotate_cam3_video, stabilize_video,
    rgb_to_brightness, rgb_to_grayscale,
    rgb_to_yellow_channel, rgb_to_pink_channel,
    rgb_to_dark_yellowgreen_channel,
    apply_threshold, morphological_cleanup, select_best_blob,
)

# ---- Choose camera from CLI arg or default to 1 -------------------------
CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f'Animating Camera {CAM} pipeline...')

# ---- Load & prepare video -----------------------------------------------
data = sio.loadmat(DATA_PATH.format(CAM))
vid = data[VID_KEYS[CAM]]
if CAM == 3:
    vid = rotate_cam3_video(vid)
    print('  Rotated Camera 3')

vid, stab_shifts = stabilize_video(vid)
print(f'  Stabilised (max shift = {np.max(np.abs(stab_shifts)):.1f} px)')

nframes = vid.shape[3]
y1, y2, x1, x2 = ROIS[CAM]
threshold = THRESHOLDS[CAM]
mode = CHANNEL_MODE[CAM]

print(f'  {nframes} frames  |  ROI: y[{y1}:{y2}] x[{x1}:{x2}]  |  T={threshold}')

# ---- Pre-process all frames (fast enough for ~200-400 frames) ------------
print('  Pre-processing frames...')
prev_cx, prev_cy = None, None
frames_rgb = []        # stabilised full frame (uint8 for imshow)
frames_bright = []     # brightness channel (ROI)
frames_binary = []     # binary threshold (ROI)
frames_cleaned = []    # cleaned blobs (ROI)
centroids_roi = []     # (cx, cy) in ROI coords, or (None, None)

for i in range(nframes):
    rgb = vid[:, :, :, i]
    frames_rgb.append(rgb)

    roi = rgb[y1:y2, x1:x2, :].astype(np.float64)

    # Brightness channel
    if mode == 'brightness':
        gray = rgb_to_brightness(roi)
    elif mode == 'yellow':
        gray = rgb_to_yellow_channel(roi)
    elif mode == 'pink':
        gray = rgb_to_pink_channel(roi)
    elif mode == 'dark_yellowgreen':
        gray = rgb_to_dark_yellowgreen_channel(roi)
    else:
        gray = rgb_to_grayscale(roi)
    frames_bright.append(gray)

    # Threshold
    binary = apply_threshold(gray, threshold)
    frames_binary.append(binary)

    # Blob cleanup
    cleaned, blob_info = morphological_cleanup(binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE)
    frames_cleaned.append(cleaned)

    # Centroid
    cx, cy = select_best_blob(blob_info, prev_cx, prev_cy)
    if cx is not None:
        prev_cx, prev_cy = cx, cy
    centroids_roi.append((cx, cy))

print('  Done. Launching animation...')

# ---- Set up figure -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f'Camera {CAM} — Frame 0/{nframes}', fontsize=13, fontweight='bold')
plt.subplots_adjust(wspace=0.08)

# Panel 1: Stabilised RGB + ROI
im0 = axes[0].imshow(frames_rgb[0])
rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      linewidth=2, edgecolor='lime', facecolor='none')
axes[0].add_patch(rect)
axes[0].set_title('Stabilised + ROI', fontsize=10)
axes[0].axis('off')

# Panel 2: Brightness channel
bmax = max(np.max(frames_bright[0]), 1)
im1 = axes[1].imshow(frames_bright[0], cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Brightness', fontsize=10)
axes[1].axis('off')

# Panel 3: Binary threshold
im2 = axes[2].imshow(frames_binary[0], cmap='gray', vmin=0, vmax=1)
axes[2].set_title(f'Threshold (T={threshold})', fontsize=10)
axes[2].axis('off')

# Panel 4: Cleaned blobs + centroid
im3 = axes[3].imshow(frames_cleaned[0], cmap='gray', vmin=0, vmax=1)
dot, = axes[3].plot([], [], 'ro', markersize=8, markeredgewidth=2)
axes[3].set_title('Blobs + Centroid', fontsize=10)
axes[3].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.93])


def update(frame_idx):
    """Update all four panels for the given frame."""
    im0.set_data(frames_rgb[frame_idx])
    im1.set_data(frames_bright[frame_idx])
    im2.set_data(frames_binary[frame_idx])
    im3.set_data(frames_cleaned[frame_idx])

    cx, cy = centroids_roi[frame_idx]
    if cx is not None:
        dot.set_data([cx], [cy])
    else:
        dot.set_data([], [])

    fig.suptitle(f'Camera {CAM} — Frame {frame_idx}/{nframes}',
                 fontsize=13, fontweight='bold')
    return im0, im1, im2, im3, dot


anim = FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)
plt.show()
