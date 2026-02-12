"""
Show Camera 1, frames 300–350, all in one figure with tracked centroid overlay.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from chat1 import track_camera, ROIS, VID_KEYS, CHANNEL_MODE

# -- Settings --------------------------------------------------------------
CAM = 1
F_START, F_END = 300, 350          # inclusive range

# -- Load video & run tracker ---------------------------------------------
vid = sio.loadmat(f'cam{CAM}.mat')[VID_KEYS[CAM]]   # (480, 640, 3, nframes)
x_raw, y_raw = track_camera(CAM)
y1, y2, x1, x2 = ROIS[CAM]

frames = list(range(F_START, F_END + 1))
ncols = 10
nrows = int(np.ceil(len(frames) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.2 * nrows))
axes = axes.flatten()

for i, fidx in enumerate(frames):
    ax = axes[i]
    ax.imshow(vid[:, :, :, fidx])

    # ROI box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=1, edgecolor='lime',
                          facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # tracked centroid
    if not np.isnan(x_raw[fidx]):
        ax.plot(x_raw[fidx], y_raw[fidx], 'r+', markersize=14, markeredgewidth=2)

    ax.set_title(f'F{fidx}', fontsize=7)
    ax.axis('off')

# hide unused subplots
for i in range(len(frames), len(axes)):
    axes[i].axis('off')

plt.suptitle(
    f'Camera {CAM} — Frames {F_START}–{F_END}  (mode: {CHANNEL_MODE[CAM]})',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
fname = f'debug_cam{CAM}_frames_{F_START}_{F_END}.png'
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {fname}')