"""
Diagnostic: Show frames where the tracker makes suspicious jumps.
Run this AFTER motion_tracking_v2.py (in the same folder as the cam*.mat files).

This script re-runs the tracking to get the raw pixel positions, then
identifies and visualises the frames where large jumps occur.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# -- Import the tracking functions from the main script --------------------
from chat1 import (
    track_camera, ROIS, VID_KEYS, CHANNEL_MODE
)

# -- Settings --------------------------------------------------------------
JUMP_THRESHOLD = 12      # pixels per frame — anything above this is suspicious
CONTEXT = 2              # show this many frames before and after each jump
CAMERAS = [1, 2, 3]      # which cameras to check

# -- Run tracking and find jumps -------------------------------------------
for cam in CAMERAS:
    print(f'\n{"="*50}')
    print(f'Camera {cam}')
    print(f'{"="*50}')

    # run the tracker to get raw pixel positions
    x_raw, y_raw = track_camera(cam)

    # compute frame-to-frame jump sizes
    jx = np.diff(np.nan_to_num(x_raw))
    jy = np.diff(np.nan_to_num(y_raw))
    jump_size = np.sqrt(jx**2 + jy**2)

    # find frames where the jump exceeds our threshold
    bad_frames = np.where(jump_size > JUMP_THRESHOLD)[0]

    if len(bad_frames) == 0:
        print(f'  No suspicious jumps (threshold = {JUMP_THRESHOLD} px)')
        continue

    print(f'  {len(bad_frames)} suspicious jumps at frames: {bad_frames}')

    # load the video for visualisation
    vid = sio.loadmat(f'cam{cam}.mat')[VID_KEYS[cam]]
    nframes = vid.shape[3]
    y1, y2, x1, x2 = ROIS[cam]

    # group nearby bad frames so we don't make redundant plots
    groups = []
    current_group = [bad_frames[0]]
    for bf in bad_frames[1:]:
        if bf - current_group[-1] <= 2 * CONTEXT + 1:
            current_group.append(bf)
        else:
            groups.append(current_group)
            current_group = [bf]
    groups.append(current_group)

    print(f'  {len(groups)} distinct jump events')

    # for each group, show a strip of frames around the jump
    for g_idx, group in enumerate(groups[:10]):     # limit to first 10
        f_start = max(0, group[0] - CONTEXT)
        f_end = min(nframes - 1, group[-1] + CONTEXT + 1)
        frame_indices = list(range(f_start, f_end + 1))

        ncols = len(frame_indices)
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4.5))
        if ncols == 1:
            axes = [axes]

        for j, fidx in enumerate(frame_indices):
            # show the frame
            axes[j].imshow(vid[:, :, :, fidx])

            # draw ROI box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor='lime',
                                  facecolor='none', linestyle='--')
            axes[j].add_patch(rect)

            # draw tracked centroid (red +)
            if not np.isnan(x_raw[fidx]):
                axes[j].plot(x_raw[fidx], y_raw[fidx], 'r+',
                             markersize=18, markeredgewidth=3)

            # red border on the actual jump frame(s)
            is_jump = fidx in group or (fidx - 1) in group
            for spine in axes[j].spines.values():
                spine.set_edgecolor('red' if is_jump else 'white')
                spine.set_linewidth(4 if is_jump else 1)

            # title: frame number + jump size
            title = f'F{fidx}'
            title_col = 'black'
            if fidx > 0:
                jmp = jump_size[fidx - 1] if fidx - 1 < len(jump_size) else 0
                if jmp > JUMP_THRESHOLD:
                    title += f'\njump = {jmp:.0f} px'
                    title_col = 'red'
            axes[j].set_title(title, fontsize=9, color=title_col)
            axes[j].axis('off')

        mode_label = CHANNEL_MODE[cam]
        plt.suptitle(
            f'Camera {cam} (mode: {mode_label}) — Jump Event at Frame(s) {group}',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        fname = f'debug_cam{cam}_jump{g_idx}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')

print('\nDone! Check the debug_cam*_jump*.png files to see what went wrong.')
print('Look for: bright objects that are NOT the can (shoes, reflections, etc.)')