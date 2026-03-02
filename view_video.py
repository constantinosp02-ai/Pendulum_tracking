"""Quick viewer for the .mat camera videos with tracking overlay."""
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os

from Part1Final import track_camera, VID_KEYS, rotate_cam3_video, stabilize_video

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# --- Load video (apply same preprocessing as Part1Final) ---
mat = scipy.io.loadmat(f'cam{cam}.mat')
vid = mat[VID_KEYS[cam]]
if cam == 3:
    vid = rotate_cam3_video(vid)
vid, _ = stabilize_video(vid)
nframes = vid.shape[3]
print(f'Camera {cam}: {nframes} frames')

# --- Get raw tracking positions (cache to avoid re-running) ---
cache = f'tracking_raw_cam{cam}.npz'
if os.path.exists(cache):
    print(f'  Loading cached positions from {cache}')
    d = np.load(cache)
    x_pos, y_pos = d['x'], d['y']
else:
    print(f'  Running tracker (first time — will cache result)...')
    x_pos, y_pos = track_camera(cam)
    np.savez(cache, x=x_pos, y=y_pos)
    print(f'  Saved {cache}')

# --- Set up figure ---
fig, ax = plt.subplots()
im = ax.imshow(vid[:, :, :, 0])
ax.set_title(f'Camera {cam}')
ax.axis('off')

frame_text = ax.text(0.02, 0.95, f'Frame 0 / {nframes-1}', transform=ax.transAxes,
                     fontsize=12, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                     verticalalignment='top')

# trail line and current-position dot
trail_line, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.8)
dot, = ax.plot([], [], 'ro', markersize=8)

def update(i):
    im.set_data(vid[:, :, :, i])
    frame_text.set_text(f'Frame {i} / {nframes-1}')

    # draw trail from frame 0 to current frame
    trail_x = x_pos[:i+1]
    trail_y = y_pos[:i+1]
    # skip NaNs for clean line
    valid = ~np.isnan(trail_x) & ~np.isnan(trail_y)
    trail_line.set_data(trail_x[valid], trail_y[valid])

    # current dot
    if not np.isnan(x_pos[i]) and not np.isnan(y_pos[i]):
        dot.set_data([x_pos[i]], [y_pos[i]])
    else:
        dot.set_data([], [])

    return [im, frame_text, trail_line, dot]

ani = animation.FuncAnimation(fig, update, frames=nframes, interval=50, blit=True)
plt.show()
