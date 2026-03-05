"""
MECH0107 Coursework
Part 1

Extracts the position of an oscillating mass from video frames
captured by three cameras using image processing techniques.

Methodology:
    1. Video stabilisation via phase correlation (FFT-based)
    2. ROI cropping
    3. Brightness channel: max(R, G, B)
    4. Binary thresholding
    5. Morphological cleanup (connected component labelling)
    6. Centre-of-mass computation
"""

import numpy as np                          # numerical arrays and math
import scipy.io as sio                      # load .mat files
from scipy.ndimage import label, rotate     # blob labelling + image rotation
from scipy.ndimage import shift as ndi_shift  # sub-pixel frame shifting
import matplotlib.pyplot as plt             # plotting
import os                                   # file existence checks

# Configuration

VID_KEYS = {1: 'vidFrames1_4', 2: 'vidFrames2_4', 3: 'vidFrames3_4'}  # .mat variable names per camera
DATA_PATH = 'cam{}.mat'  # filename template for camera data

# Region of interest per camera: (y_min, y_max, x_min, x_max)
ROIS = {
    1: (150, 430, 280, 500),   # cam 1 crop bounds
    2: (80,  410, 180, 480),   # cam 2 crop bounds
    3: (200, 520, 129, 349),   # cam 3 crop bounds (after rotation)
}

# Brightness thresholds: chosen from the intensity histogram of each camera
THRESHOLDS = {1: 240, 2: 245, 3: 200}

MIN_BLOB_SIZE = 20    # ignore blobs smaller than this (noise)
MAX_BLOB_SIZE = 8000  # ignore blobs larger than this (background)


# Helper functions

def rotate_cam3_video(vid):
    """Rotate Camera 3 frames: 90° CW then 16° CW so the can base sits horizontal."""
    vid = np.rot90(vid, k=-1, axes=(0, 1))       # 90 deg clockwise
    for i in range(vid.shape[3]):                 # loop over each frame
        vid[:, :, :, i] = rotate(vid[:, :, :, i], angle=-16,
                                  reshape=False, order=1)  # extra 16 deg CW correction
    return vid


def stabilize_video(vid):
    """
    Stabilise a video using phase correlation.

    Phase correlation estimates the translational shift between frames
    using the normalised cross-power spectrum of their 2D FFTs. Each
    frame is aligned to frame 0 by accumulating frame-to-frame shifts.
    """
    nframes = vid.shape[3]            # total number of frames
    H, W = vid.shape[0], vid.shape[1] # frame height and width

    stabilised = vid.copy()                    # work on a copy so we keep the original
    shifts = np.zeros((nframes, 2))            # cumulative (dy, dx) for each frame

    # grayscale of first frame as initial reference
    prev_gray = (0.2989 * stabilised[:, :, 0, 0] +
                 0.5870 * stabilised[:, :, 1, 0] +
                 0.1140 * stabilised[:, :, 2, 0])  # standard luma weights

    for i in range(1, nframes):                     # start from frame 1
        curr = stabilised[:, :, :, i].astype(np.float64)  # current frame as float
        curr_gray = 0.2989 * curr[:, :, 0] + 0.5870 * curr[:, :, 1] + 0.1140 * curr[:, :, 2]  # grayscale

        # cross-power spectrum
        F1 = np.fft.fft2(prev_gray)                # FFT of previous frame
        F2 = np.fft.fft2(curr_gray)                # FFT of current frame
        cross_power = F1 * np.conj(F2)             # element-wise cross-power
        mag = np.abs(cross_power)                  # magnitude for normalisation
        mag[mag == 0] = 1                          # avoid divide-by-zero
        correlation = np.real(np.fft.ifft2(cross_power / mag))  # inverse FFT gives correlation surface

        # peak location gives the shift
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)  # find peak coords
        dy, dx = peak[0], peak[1]                  # raw shift values
        if dy > H // 2: dy -= H                    # wrap-around correction for negative shifts
        if dx > W // 2: dx -= W                    # same for horizontal

        shifts[i] = shifts[i - 1] + [dy, dx]       # accumulate shifts relative to frame 0

        # apply cumulative shift to align with frame 0
        for c in range(3):                          # loop R, G, B channels
            stabilised[:, :, c, i] = ndi_shift(
                vid[:, :, c, i].astype(np.float64),
                shift=-shifts[i], order=1, mode='constant', cval=0  # shift back by accumulated offset
            ).astype(vid.dtype)

        prev_gray = (0.2989 * stabilised[:, :, 0, i] +
                     0.5870 * stabilised[:, :, 1, i] +
                     0.1140 * stabilised[:, :, 2, i])  # update reference to stabilised frame

    return stabilised, shifts


def rgb_to_brightness(rgb_image):
    """Brightness = max(R, G, B) per pixel. Isolates the brightest spots."""
    return np.max(rgb_image.astype(np.float64), axis=2)  # take max across colour channels


def apply_threshold(image, threshold):
    """Binary threshold: pixels above threshold → 1, else → 0."""
    return (image > threshold).astype(np.float64)  # simple binary mask


def morphological_cleanup(binary_image, min_size, max_size):
    """
    Connected component labelling followed by area filtering.
    Removes blobs smaller than min_size or larger than max_size.
    Returns the cleaned mask and a list of (cx, cy, area) for valid blobs.
    """
    labelled, num_features = label(binary_image)   # label connected regions
    cleaned = np.zeros_like(binary_image)           # output mask starts empty
    blob_info = []                                  # store centroid + area of valid blobs

    rows, cols = binary_image.shape                 # image dimensions
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')  # coordinate grids for centroid calc

    for k in range(1, num_features + 1):            # loop over each labelled blob
        mask = (labelled == k)                      # boolean mask for this blob
        area = int(np.sum(mask))                    # number of pixels in blob
        if area < min_size or area > max_size:      # skip if too small or too large
            continue
        cx = float(np.sum(xx * mask)) / area        # x centroid (weighted mean)
        cy = float(np.sum(yy * mask)) / area        # y centroid
        cleaned[mask] = 1                           # add blob to cleaned mask
        blob_info.append((cx, cy, area))            # record blob info

    return cleaned, blob_info


def select_best_blob(blob_info, prev_cx, prev_cy):
    """
    Pick the best blob: closest to previous position if available,
    otherwise the largest one.
    """
    if len(blob_info) == 0:                         # no blobs found
        return prev_cx, prev_cy                     # keep previous position
    if len(blob_info) == 1:                         # only one blob, use it
        return blob_info[0][0], blob_info[0][1]
    if prev_cx is not None:                         # multiple blobs and we have history
        best = min(blob_info, key=lambda b: (b[0]-prev_cx)**2 + (b[1]-prev_cy)**2)  # nearest to last known pos
    else:
        best = max(blob_info, key=lambda b: b[2])   # no history, pick the biggest blob
    return best[0], best[1]


# Main tracking pipeline for one camera

def track_camera(cam_id):
    """
    Full tracking pipeline for a single camera:
      stabilise → crop ROI → brightness → threshold → cleanup → centroid
    """
    filepath = DATA_PATH.format(cam_id)             # build filename
    print(f'  Loading {filepath}...')
    data = sio.loadmat(filepath)                    # load .mat file
    vid = data[VID_KEYS[cam_id]]                    # extract video array

    # Camera 2 starts ~5 frames late therefore trim to synchronise with other cameras
    if cam_id == 2:
        vid = vid[:, :, :, 5:]                      # drop first 5 frames

    if cam_id == 3:
        vid = rotate_cam3_video(vid)                # fix camera 3 orientation

    vid, _ = stabilize_video(vid)                   # stabilise all frames to frame 0
    nframes = vid.shape[3]                          # number of frames after trimming

    y1, y2, x1, x2 = ROIS[cam_id]                  # unpack ROI bounds
    threshold = THRESHOLDS[cam_id]                  # brightness threshold for this camera

    x_pos = np.full(nframes, np.nan)                # horizontal position array (fill NaN)
    y_pos = np.full(nframes, np.nan)                # vertical position array
    prev_cx, prev_cy = None, None                   # no previous position yet

    for i in range(nframes):                        # process each frame
        rgb_frame = vid[:, :, :, i].astype(np.float64)  # get frame as float
        roi = rgb_frame[y1:y2, x1:x2, :]           # crop to region of interest

        bright = rgb_to_brightness(roi)             # max(R,G,B) brightness
        binary = apply_threshold(bright, threshold) # threshold to binary
        cleaned, blob_info = morphological_cleanup(binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE)  # filter blobs
        cx, cy = select_best_blob(blob_info, prev_cx, prev_cy)  # pick best blob centroid

        if cx is not None:                          # valid detection
            x_pos[i] = cx + x1                      # convert ROI coords back to full frame
            y_pos[i] = cy + y1                      # same for y
            prev_cx, prev_cy = cx, cy               # update tracking history

        if (i + 1) % 100 == 0 or i == nframes - 1:  # progress print every 100 frames
            print(f'    Frame {i+1:>4d}/{nframes}')

    # interpolate through any NaN gaps
    valid = ~np.isnan(x_pos)                        # mask of frames with valid detections
    if np.sum(~valid) > 0 and np.sum(valid) >= 2:   # only interpolate if there are gaps and enough good points
        idx = np.arange(nframes)                    # frame indices
        x_pos = np.interp(idx, idx[valid], x_pos[valid])  # linear interp over gaps
        y_pos = np.interp(idx, idx[valid], y_pos[valid])  # same for y

    return x_pos, y_pos


# Main execution

if __name__ == '__main__':
    print('MECH0107 CW — Part 1: Motion Tracking')
    print('=' * 50)

    results = {}                                    # dict to hold results per camera
    for cam in [1, 2, 3]:                           # process all three cameras
        print(f'\nCamera {cam}')
        print('-' * 30)
        if not os.path.exists(DATA_PATH.format(cam)):  # check data file exists
            print(f'  WARNING: {DATA_PATH.format(cam)} not found.')
            continue
        x, y = track_camera(cam)                    # run tracking pipeline
        results[cam] = {
            'x_raw': x, 'y_raw': y,                # raw pixel positions
            'dx': x - np.nanmean(x),                # mean-centred horizontal displacement
            'dy': -(y - np.nanmean(y)),             # mean-centred vertical (negated because image y is flipped)
        }

    # save tracking data
    if results:                                     # only save if we got results
        np.savez('tracking_results.npz',
                 **{f'cam{c}_{ax}': results[c][f'd{ax}']
                    for c in results for ax in ['x', 'y']})  # save dx, dy for each camera
        print('\nSaved: tracking_results.npz')

    # Figure 1: Displacement time series
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))  # 3 cameras x 2 directions
    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        frames = np.arange(len(results[cam]['dx']))    # frame numbers for x-axis
        axes[i, 0].plot(frames, results[cam]['dx'], color='#2166ac', linewidth=0.8)  # horizontal
        axes[i, 0].set_ylabel('Displacement (px)')
        axes[i, 0].set_title(f'Camera {cam} — Horizontal')
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(0, color='k', lw=0.5, ls='--')  # zero reference line

        axes[i, 1].plot(frames, results[cam]['dy'], color='#b2182b', linewidth=0.8)  # vertical
        axes[i, 1].set_ylabel('Displacement (px)')
        axes[i, 1].set_title(f'Camera {cam} — Vertical')
        axes[i, 1].set_xlabel('Frame')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(0, color='k', lw=0.5, ls='--')  # zero line

    plt.suptitle('Tracked Displacement of the Oscillating Mass',
                 fontsize=13, fontweight='bold', y=1.01)  # overall title
    plt.tight_layout()
    plt.savefig('fig_displacement.png', dpi=200, bbox_inches='tight')  # save figure
    plt.close()
    print('  Saved fig_displacement.png')

    # Figure 2: 2D trajectories
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))    # one subplot per camera
    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        dx, dy = results[cam]['dx'], results[cam]['dy']
        v = ~(np.isnan(dx) | np.isnan(dy))              # valid (non-NaN) frames
        sc = axes[i].scatter(dx[v], dy[v], c=np.arange(np.sum(v)),
                             cmap='viridis', s=4, alpha=0.7)  # colour by time
        axes[i].set_xlabel('Horizontal (px)')
        axes[i].set_ylabel('Vertical (px)')
        axes[i].set_title(f'Camera {cam}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')                      # equal axis scaling
        plt.colorbar(sc, ax=axes[i], label='Frame', shrink=0.8)  # colourbar showing frame number

    plt.suptitle('2D Trajectory (colour = time)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_trajectory.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig_trajectory.png')
