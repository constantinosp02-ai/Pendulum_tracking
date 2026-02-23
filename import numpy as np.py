import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

cam = 3

data = sio.loadmat(f'cam{cam}.mat')
keys = {1: 'vidFrames1_4', 2: 'vidFrames2_4', 3: 'vidFrames3_4'}
vid = data[keys[cam]]

frame = vid[:, :, :, 0]

# rotate Camera 3: 90° CW then 18° CW
if cam == 3:
    frame = np.rot90(frame, k=-1, axes=(0, 1))
    frame = rotate(frame, angle=-18, reshape=False, order=1)

y1, y2, x1, x2 = 258, 520, 129, 349
plt.imshow(frame)
plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime', facecolor='none'))
plt.title(f'Camera {cam} — ROI check')
plt.show()