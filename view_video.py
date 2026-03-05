"""Side-by-side viewer: first 50 frames of all 3 cameras in slow motion."""
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from Part1Final import VID_KEYS, rotate_cam3_video

N_FRAMES = 50
INTERVAL = 200  # ms per frame (slow motion)

# --- Load all 3 cameras (raw, no stabilisation for speed) ---
vids = {}
for cam in [1, 2, 3]:
    print(f'Loading cam{cam}.mat ...')
    mat = scipy.io.loadmat(f'cam{cam}.mat')
    vid = mat[VID_KEYS[cam]]
    if cam == 2:
        vid = vid[:, :, :, 5:]  # trim first 5 frames to sync
    if cam == 3:
        vid = rotate_cam3_video(vid)
    vids[cam] = vid

# --- Set up side-by-side figure ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ims = {}
texts = {}
for idx, cam in enumerate([1, 2, 3]):
    axes[idx].set_title(f'Camera {cam}')
    axes[idx].axis('off')
    ims[cam] = axes[idx].imshow(vids[cam][:, :, :, 0])
    texts[cam] = axes[idx].text(
        0.02, 0.95, 'Frame 0', transform=axes[idx].transAxes,
        fontsize=11, color='white', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
        verticalalignment='top')

plt.tight_layout()
fig.suptitle('Press SPACE to pause/resume', fontsize=10, color='gray', y=0.02)

paused = [False]

def on_key(event):
    if event.key == ' ':
        if paused[0]:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        paused[0] = not paused[0]

fig.canvas.mpl_connect('key_press_event', on_key)

def update(i):
    out = []
    for cam in [1, 2, 3]:
        ims[cam].set_data(vids[cam][:, :, :, i])
        texts[cam].set_text(f'Frame {i}')
        out += [ims[cam], texts[cam]]
    return out

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=INTERVAL, blit=True)
plt.show()
