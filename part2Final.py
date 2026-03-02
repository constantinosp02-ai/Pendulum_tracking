"""
MECH0107 Coursework
Part 2: Time-Frequency Analysis

Performs Gabor transform (spectrogram) analysis on the displacement
data extracted in Part 1, using Camera 1.
"""

import numpy as np
import matplotlib.pyplot as plt

#Configuration

CAM = 1

# Three window widths to demonstrate the time-frequency trade-off:
# narrow a = sharp time localisation, poor frequency resolution
# wide  a = sharp frequency resolution, poor time localisation
WIDTHS = [5, 15, 40]

TSLIDE_STEP = 2   # frames between consecutive window positions
MAX_FREQ = 0.15   # cycles/frame — upper limit for display


# Load data

data = np.load('tracking_results.npz')
dx = data[f'cam{CAM}_x']
dy = data[f'cam{CAM}_y']

# mean-subtract so the signal oscillates around zero
dx = dx - np.mean(dx)
dy = dy - np.mean(dy)

n = len(dx)
t = np.arange(n)

# frequency vector in cycles/frame
k = np.arange(-n//2, n//2) / n

print(f'Camera {CAM}: {n} frames, freq resolution = {1/n:.5f} cyc/frame')


#Functions

def compute_fft(signal):
    """FFT magnitude spectrum, normalised to peak = 1."""
    mag = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    return mag / np.max(mag)


def gabor_spectrogram(signal, width):
    """
    Slide a Gaussian window g(t) = exp(-(t-tau)^2 / a^2) across the signal,
    take the FFT at each position, and stack the magnitudes into a spectrogram.
    """
    tslide = np.arange(0, n, TSLIDE_STEP)
    Sgt_spec = np.zeros((len(tslide), n))

    for j, tau in enumerate(tslide):
        g = np.exp(-(t - tau)**2 / width**2)
        Sgt = np.fft.fft(g * signal)
        mag = np.abs(np.fft.fftshift(Sgt))
        Sgt_spec[j, :] = mag

    # normalise by global max so amplitude decay is visible across time
    if Sgt_spec.max() > 0:
        Sgt_spec /= Sgt_spec.max()

    return tslide, Sgt_spec


# Figure 1: Signal + global FFT

mag_x = compute_fft(dx)
mag_y = compute_fft(dy)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0, 0].plot(t, dx, 'k', linewidth=0.7)
axes[0, 0].set_title('Horizontal displacement')
axes[0, 0].set_xlabel('Frame number')
axes[0, 0].set_ylabel('Displacement (px)')
axes[0, 0].set_xlim([0, n])

axes[0, 1].plot(t, dy, 'k', linewidth=0.7)
axes[0, 1].set_title('Vertical displacement')
axes[0, 1].set_xlabel('Frame number')
axes[0, 1].set_ylabel('Displacement (px)')
axes[0, 1].set_xlim([0, n])

axes[1, 0].plot(k, mag_x, 'b', linewidth=0.7)
axes[1, 0].set_title('FFT — Horizontal')
axes[1, 0].set_xlabel('Frequency (cycles/frame)')
axes[1, 0].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[1, 0].set_xlim([-MAX_FREQ, MAX_FREQ])

axes[1, 1].plot(k, mag_y, 'b', linewidth=0.7)
axes[1, 1].set_title('FFT — Vertical')
axes[1, 1].set_xlabel('Frequency (cycles/frame)')
axes[1, 1].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[1, 1].set_xlim([-MAX_FREQ, MAX_FREQ])

plt.tight_layout()
plt.savefig('fig_part2_signal_fft.png', dpi=200, bbox_inches='tight')
plt.close()


#Figure 2: Gabor illustration at a single time instant

tau_example = n // 2
w_example = WIDTHS[1]
g_example = np.exp(-(t - tau_example)**2 / w_example**2)
Sg_example = g_example * dx
mag_example = np.abs(np.fft.fftshift(np.fft.fft(Sg_example)))
mag_example = mag_example / mag_example.max()

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t, dx, 'k', linewidth=0.7, label='Signal $S(t)$')
axes[0].plot(t, g_example * np.max(np.abs(dx)), 'r', linewidth=1.2,
             label=f'Gabor window ($\\tau$={tau_example}, a={w_example})')
axes[0].set_title('Signal with Gabor function in time domain')
axes[0].set_xlabel('Frame number')
axes[0].set_ylabel('$S(t)$')
axes[0].legend()
axes[0].set_xlim([0, n])

axes[1].plot(t, Sg_example, 'm', linewidth=0.7)
axes[1].set_title('Windowed signal in time domain')
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('$S(t) \\cdot g(t)$')
axes[1].set_xlim([0, n])

axes[2].plot(k, mag_example, 'b', linewidth=0.7)
axes[2].set_title('Fourier transform of the windowed signal')
axes[2].set_xlabel('Frequency (cycles/frame)')
axes[2].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[2].set_xlim([-MAX_FREQ, MAX_FREQ])

plt.tight_layout()
plt.savefig('fig_part2_gabor_illustration.png', dpi=200, bbox_inches='tight')
plt.close()


# Compute spectrograms for all widths

print('Computing spectrograms...')
results_x = {}
results_y = {}
for w in WIDTHS:
    print(f'  a = {w}')
    tslide, spec_x = gabor_spectrogram(dx, w)
    _, spec_y = gabor_spectrogram(dy, w)
    results_x[w] = (tslide, spec_x)
    results_y[w] = (tslide, spec_y)

freq_mask = (k >= 0) & (k <= MAX_FREQ)


#Figure 3: Width comparison — horizontal

fig, axes = plt.subplots(len(WIDTHS) + 1, 1, figsize=(13, 3 * (len(WIDTHS) + 1)))

axes[0].plot(t, dx, 'k', linewidth=0.7)
axes[0].set_ylabel('Disp. (px)')
axes[0].set_title(f'Camera {CAM} — Horizontal displacement')
axes[0].set_xlim([0, n])

for i, w in enumerate(WIDTHS):
    tslide, spec = results_x[w]
    axes[i+1].pcolormesh(tslide, k[freq_mask], spec[:, freq_mask].T,
                          shading='auto', cmap='jet')
    axes[i+1].set_ylabel('Freq\n(cyc/frame)')
    axes[i+1].set_title(f'Gabor width a = {w} frames')

axes[-1].set_xlabel('Frame number')
plt.tight_layout()
plt.savefig('fig_part2_width_comparison_horiz.png', dpi=200, bbox_inches='tight')
plt.close()


#Figure 4: Width comparison - vertical

fig, axes = plt.subplots(len(WIDTHS) + 1, 1, figsize=(13, 3 * (len(WIDTHS) + 1)))

axes[0].plot(t, dy, 'k', linewidth=0.7)
axes[0].set_ylabel('Disp. (px)')
axes[0].set_title(f'Camera {CAM} — Vertical displacement')
axes[0].set_xlim([0, n])

for i, w in enumerate(WIDTHS):
    tslide, spec = results_y[w]
    axes[i+1].pcolormesh(tslide, k[freq_mask], spec[:, freq_mask].T,
                          shading='auto', cmap='jet')
    axes[i+1].set_ylabel('Freq\n(cyc/frame)')
    axes[i+1].set_title(f'Gabor width a = {w} frames')

axes[-1].set_xlabel('Frame number')
plt.tight_layout()
plt.savefig('fig_part2_width_comparison_vert.png', dpi=200, bbox_inches='tight')
plt.close()


#Figure 5: Spectrograms + dominant frequency tracking 

w_mid = WIDTHS[1]
tslide, spec_x = results_x[w_mid]
_, spec_y = results_y[w_mid]

freqs_pos = k[freq_mask]
spec_x_pos = spec_x[:, freq_mask]
spec_y_pos = spec_y[:, freq_mask]

dom_fx = freqs_pos[np.argmax(spec_x_pos, axis=1)]
dom_fy = freqs_pos[np.argmax(spec_y_pos, axis=1)]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].pcolormesh(tslide, freqs_pos, spec_x_pos.T, shading='auto', cmap='jet')
axes[0, 0].plot(tslide, dom_fx, 'w--', linewidth=0.8, alpha=0.7)
axes[0, 0].set_title(f'Spectrogram — Horizontal (a={w_mid})')
axes[0, 0].set_ylabel('Frequency (cyc/frame)')

axes[0, 1].pcolormesh(tslide, freqs_pos, spec_y_pos.T, shading='auto', cmap='jet')
axes[0, 1].plot(tslide, dom_fy, 'w--', linewidth=0.8, alpha=0.7)
axes[0, 1].set_title(f'Spectrogram — Vertical (a={w_mid})')
axes[0, 1].set_ylabel('Frequency (cyc/frame)')

axes[1, 0].plot(tslide, dom_fx, 'b.-', markersize=2, linewidth=0.6)
axes[1, 0].set_title('Dominant frequency — Horizontal')
axes[1, 0].set_xlabel('Frame number')
axes[1, 0].set_ylabel('Frequency (cyc/frame)')
axes[1, 0].set_ylim([0, MAX_FREQ])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(tslide, dom_fy, 'r.-', markersize=2, linewidth=0.6)
axes[1, 1].set_title('Dominant frequency — Vertical')
axes[1, 1].set_xlabel('Frame number')
axes[1, 1].set_ylabel('Frequency (cyc/frame)')
axes[1, 1].set_ylim([0, MAX_FREQ])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part2_dominant_freq.png', dpi=200, bbox_inches='tight')
plt.close()
