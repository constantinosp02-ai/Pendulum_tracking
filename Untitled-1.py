"""
MECH0107 Coursework — Part 1: Motion Tracking

Extracts the position of an oscillating mass from video frames
captured by three cameras using image processing techniques.

Pipeline:
    1. Video stabilisation via phase correlation (FFT-based)
    2. ROI cropping
    3. Brightness channel: max(R, G, B)
    4. Binary thresholding
    5. Morphological cleanup (connected component labelling)
    6. Centre-of-mass computation
"""

import numpy as np
import scipy.io as sio
from scipy.ndimage import label, rotate
from scipy.ndimage import shift as ndi_shift
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────────────────────

VID_KEYS = {1: 'vidFrames1_4', 2: 'vidFrames2_4', 3: 'vidFrames3_4'}
DATA_PATH = 'cam{}.mat'

# Region of interest per camera: (y_min, y_max, x_min, x_max)
ROIS = {
    1: (150, 430, 280, 500),
    2: (80,  410, 180, 480),
    3: (200, 520, 129, 349),
}

# Brightness thresholds — chosen from the intensity histogram of each camera
THRESHOLDS = {1: 230, 2: 230, 3: 230}

MIN_BLOB_SIZE = 20
MAX_BLOB_SIZE = 8000


# ── Helper functions ───────────────────────────────────────────────────

def rotate_cam3_video(vid):
    """Rotate Camera 3 frames: 90° CW then 16° CW to correct orientation."""
    vid = np.rot90(vid, k=-1, axes=(0, 1))
    for i in range(vid.shape[3]):
        vid[:, :, :, i] = rotate(vid[:, :, :, i], angle=-16,
                                  reshape=False, order=1)
    return vid


def stabilize_video(vid):
    """
    Stabilise a video using phase correlation.

    Phase correlation estimates the translational shift between frames
    using the normalised cross-power spectrum of their 2D FFTs. Each
    frame is aligned to frame 0 by accumulating frame-to-frame shifts.
    """
    nframes = vid.shape[3]
    H, W = vid.shape[0], vid.shape[1]

    stabilised = vid.copy()
    shifts = np.zeros((nframes, 2))

    # grayscale of first frame as initial reference
    prev_gray = (0.2989 * stabilised[:, :, 0, 0] +
                 0.5870 * stabilised[:, :, 1, 0] +
                 0.1140 * stabilised[:, :, 2, 0])

    for i in range(1, nframes):
        curr = stabilised[:, :, :, i].astype(np.float64)
        curr_gray = 0.2989 * curr[:, :, 0] + 0.5870 * curr[:, :, 1] + 0.1140 * curr[:, :, 2]

        # cross-power spectrum
        F1 = np.fft.fft2(prev_gray)
        F2 = np.fft.fft2(curr_gray)
        cross_power = F1 * np.conj(F2)
        mag = np.abs(cross_power)
        mag[mag == 0] = 1
        correlation = np.real(np.fft.ifft2(cross_power / mag))

        # peak location gives the shift
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        dy, dx = peak[0], peak[1]
        if dy > H // 2: dy -= H
        if dx > W // 2: dx -= W

        shifts[i] = shifts[i - 1] + [dy, dx]

        # apply cumulative shift to align with frame 0
        for c in range(3):
            stabilised[:, :, c, i] = ndi_shift(
                vid[:, :, c, i].astype(np.float64),
                shift=-shifts[i], order=1, mode='constant', cval=0
            ).astype(vid.dtype)

        prev_gray = (0.2989 * stabilised[:, :, 0, i] +
                     0.5870 * stabilised[:, :, 1, i] +
                     0.1140 * stabilised[:, :, 2, i])

    return stabilised, shifts


def rgb_to_brightness(rgb_image):
    """Brightness = max(R, G, B) per pixel. Isolates the brightest spots."""
    return np.max(rgb_image.astype(np.float64), axis=2)


def apply_threshold(image, threshold):
    """Binary threshold: pixels above threshold → 1, else → 0."""
    return (image > threshold).astype(np.float64)


def morphological_cleanup(binary_image, min_size, max_size):
    """
    Connected component labelling followed by area filtering.
    Removes blobs smaller than min_size or larger than max_size.
    Returns the cleaned mask and a list of (cx, cy, area) for valid blobs.
    """
    labelled, num_features = label(binary_image)
    cleaned = np.zeros_like(binary_image)
    blob_info = []

    rows, cols = binary_image.shape
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    for k in range(1, num_features + 1):
        mask = (labelled == k)
        area = int(np.sum(mask))
        if area < min_size or area > max_size:
            continue
        cx = float(np.sum(xx * mask)) / area
        cy = float(np.sum(yy * mask)) / area
        cleaned[mask] = 1
        blob_info.append((cx, cy, area))

    return cleaned, blob_info


def select_best_blob(blob_info, prev_cx, prev_cy):
    """
    Pick the best blob: closest to previous position if available,
    otherwise the largest one.
    """
    if len(blob_info) == 0:
        return prev_cx, prev_cy
    if len(blob_info) == 1:
        return blob_info[0][0], blob_info[0][1]
    if prev_cx is not None:
        best = min(blob_info, key=lambda b: (b[0]-prev_cx)**2 + (b[1]-prev_cy)**2)
    else:
        best = max(blob_info, key=lambda b: b[2])
    return best[0], best[1]


# ── Main tracking pipeline for one camera ──────────────────────────────

def track_camera(cam_id):
    """
    Full tracking pipeline for a single camera:
      stabilise → crop ROI → brightness → threshold → cleanup → centroid
    """
    filepath = DATA_PATH.format(cam_id)
    print(f'  Loading {filepath}...')
    data = sio.loadmat(filepath)
    vid = data[VID_KEYS[cam_id]]

    if cam_id == 3:
        vid = rotate_cam3_video(vid)

    vid, _ = stabilize_video(vid)
    nframes = vid.shape[3]

    y1, y2, x1, x2 = ROIS[cam_id]
    threshold = THRESHOLDS[cam_id]

    x_pos = np.full(nframes, np.nan)
    y_pos = np.full(nframes, np.nan)
    prev_cx, prev_cy = None, None

    for i in range(nframes):
        rgb_frame = vid[:, :, :, i].astype(np.float64)
        roi = rgb_frame[y1:y2, x1:x2, :]

        bright = rgb_to_brightness(roi)
        binary = apply_threshold(bright, threshold)
        cleaned, blob_info = morphological_cleanup(binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE)
        cx, cy = select_best_blob(blob_info, prev_cx, prev_cy)

        if cx is not None:
            x_pos[i] = cx + x1
            y_pos[i] = cy + y1
            prev_cx, prev_cy = cx, cy

        if (i + 1) % 100 == 0 or i == nframes - 1:
            print(f'    Frame {i+1:>4d}/{nframes}')

    # interpolate through any NaN gaps
    valid = ~np.isnan(x_pos)
    if np.sum(~valid) > 0 and np.sum(valid) >= 2:
        idx = np.arange(nframes)
        x_pos = np.interp(idx, idx[valid], x_pos[valid])
        y_pos = np.interp(idx, idx[valid], y_pos[valid])

    return x_pos, y_pos


# ── Main execution ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print('MECH0107 CW — Part 1: Motion Tracking')
    print('=' * 50)

    results = {}
    for cam in [1, 2, 3]:
        print(f'\nCamera {cam}')
        print('-' * 30)
        if not os.path.exists(DATA_PATH.format(cam)):
            print(f'  WARNING: {DATA_PATH.format(cam)} not found.')
            continue
        x, y = track_camera(cam)
        results[cam] = {
            'x_raw': x, 'y_raw': y,
            'dx': x - np.nanmean(x),
            'dy': -(y - np.nanmean(y)),   # negate because image y-axis is inverted
        }

    # save tracking data
    if results:
        np.savez('tracking_results.npz',
                 **{f'cam{c}_{ax}': results[c][f'd{ax}']
                    for c in results for ax in ['x', 'y']})
        print('\nSaved: tracking_results.npz')

    # ── Figure 1: Displacement time series ─────────────────────────────
    print('\nGenerating displacement figure...')
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        frames = np.arange(len(results[cam]['dx']))
        axes[i, 0].plot(frames, results[cam]['dx'], color='#2166ac', linewidth=0.8)
        axes[i, 0].set_ylabel('Displacement (px)')
        axes[i, 0].set_title(f'Camera {cam} — Horizontal')
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(0, color='k', lw=0.5, ls='--')

        axes[i, 1].plot(frames, results[cam]['dy'], color='#b2182b', linewidth=0.8)
        axes[i, 1].set_ylabel('Displacement (px)')
        axes[i, 1].set_title(f'Camera {cam} — Vertical')
        axes[i, 1].set_xlabel('Frame')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(0, color='k', lw=0.5, ls='--')

    plt.suptitle('Tracked Displacement of the Oscillating Mass',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_displacement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig_displacement.png')

    # ── Figure 2: 2D trajectories ──────────────────────────────────────
    print('Generating trajectory figure...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        dx, dy = results[cam]['dx'], results[cam]['dy']
        v = ~(np.isnan(dx) | np.isnan(dy))
        sc = axes[i].scatter(dx[v], dy[v], c=np.arange(np.sum(v)),
                             cmap='viridis', s=4, alpha=0.7)
        axes[i].set_xlabel('Horizontal (px)')
        axes[i].set_ylabel('Vertical (px)')
        axes[i].set_title(f'Camera {cam}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
        plt.colorbar(sc, ax=axes[i], label='Frame', shrink=0.8)

    plt.suptitle('2D Trajectory (colour = time)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_trajectory.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig_trajectory.png')

    print('\nPart 1 complete.')