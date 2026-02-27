"""
MECH0107 CW1 — Part 2: Time-Frequency Analysis
Gabor Transform (Lecture 3, Sec. 3.2-3.3)
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# Configuration
# =========================================================================
CAM = 1                          # camera to analyse

# Gabor window widths to compare (in frames)
# small = good time resolution, poor freq resolution
# large = good freq resolution, poor time resolution
WIDTHS = [5, 15, 40]

TSLIDE_STEP = 2                  # step between window positions (frames)
MAX_FREQ = 0.15                  # max frequency to display (cycles/frame)

# =========================================================================
# Load displacement data from Part 1
# =========================================================================
print('Loading tracking data...')
data = np.load('tracking_results.npz')
dx = data[f'cam{CAM}_x']        # horizontal displacement (pixels)
dy = data[f'cam{CAM}_y']        # vertical displacement (pixels)

# subtract mean so signal oscillates around zero
dx = dx - np.mean(dx)
dy = dy - np.mean(dy)

n = len(dx)                      # number of frames
t = np.arange(n)                 # time axis in frame numbers

# frequency vector in cycles per frame (as per Lama's email)
# k = (-n/2 : n/2-1) / n — already centred, no fftshift needed
k = np.arange(-n//2, n//2) / n

print(f'Camera {CAM}: {n} frames')
print(f'Frequency resolution: {1/n:.5f} cycles/frame')


# =========================================================================
# Global Fourier Transform (Lecture 3, Sec. 3.1)
# =========================================================================
def compute_fft(signal):
    """Compute FFT and return normalised magnitude spectrum."""
    S_hat = np.fft.fft(signal)
    # fftshift to match the centred frequency vector k
    mag = np.abs(np.fft.fftshift(S_hat))
    mag = mag / np.max(mag)
    return mag


# =========================================================================
# Gabor Transform / Spectrogram (Lecture 3, Sec. 3.2-3.3)
# =========================================================================
def gabor_spectrogram(signal, width):
    """
    Compute spectrogram via Gabor transform.

    For each time position tau, we:
      1. Create Gaussian window: g(t) = exp(-(t - tau)^2 / a^2)
      2. Multiply signal by the window
      3. Take FFT of the result
      4. Store the magnitude as one row of the spectrogram

    Parameters
    ----------
    signal : array, the displacement signal
    width  : float, Gabor window width 'a' (in frames)

    Returns
    -------
    tslide   : array, window centre positions (frame numbers)
    Sgt_spec : 2D array, spectrogram matrix (rows=time, cols=freq)
    """
    tslide = np.arange(0, n, TSLIDE_STEP)
    Sgt_spec = np.zeros((len(tslide), n))

    for j, tau in enumerate(tslide):
        # Gabor window function (Gaussian centred at tau)
        g = np.exp(-(t - tau)**2 / width**2)

        # multiply signal by window
        Sg = g * signal

        # FFT of windowed signal
        Sgt = np.fft.fft(Sg)

        # store normalised magnitude (fftshift to match k)
        mag = np.abs(np.fft.fftshift(Sgt))
        m = np.max(mag)
        if m > 0:
            Sgt_spec[j, :] = mag / m

    return tslide, Sgt_spec


# =========================================================================
# Plotting
# =========================================================================

# --- Figure: Signal + global FFT for both components ---
print('\nComputing global FFT...')
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
print('  Saved fig_part2_signal_fft.png')


# --- Figure: Gabor illustration at one time instant ---
print('\nGenerating Gabor illustration...')
tau_example = n // 2
w_example = WIDTHS[1]  # use the middle width
g_example = np.exp(-(t - tau_example)**2 / w_example**2)
Sg_example = g_example * dx
Sgt_example = np.fft.fft(Sg_example)
mag_example = np.abs(np.fft.fftshift(Sgt_example))
mag_example = mag_example / np.max(mag_example)

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
axes[1].set_title('Convolution of the signal with Gabor transform in time domain')
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('$S(t) \\cdot g(t)$')
axes[1].set_xlim([0, n])

axes[2].plot(k, mag_example, 'b', linewidth=0.7)
axes[2].set_title('Fourier transform of the new signal in frequency domain')
axes[2].set_xlabel('Frequency (cycles/frame)')
axes[2].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[2].set_xlim([-MAX_FREQ, MAX_FREQ])

plt.tight_layout()
plt.savefig('fig_part2_gabor_illustration.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved fig_part2_gabor_illustration.png')


# --- Compute spectrograms for all widths, both components ---
print('\nComputing spectrograms...')
results_x = {}
results_y = {}
for w in WIDTHS:
    print(f'  width a = {w} frames ...')
    tslide, spec_x = gabor_spectrogram(dx, w)
    _, spec_y = gabor_spectrogram(dy, w)
    results_x[w] = (tslide, spec_x)
    results_y[w] = (tslide, spec_y)


# --- Figure: Width comparison (horizontal) ---
print('\nPlotting width comparison — horizontal...')
fig, axes = plt.subplots(len(WIDTHS) + 1, 1, figsize=(13, 3 * (len(WIDTHS) + 1)))

axes[0].plot(t, dx, 'k', linewidth=0.7)
axes[0].set_ylabel('Disp. (px)')
axes[0].set_title(f'Camera {CAM} — Horizontal displacement')
axes[0].set_xlim([0, n])

freq_mask = (k >= 0) & (k <= MAX_FREQ)
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
print('  Saved fig_part2_width_comparison_horiz.png')


# --- Figure: Width comparison (vertical) ---
print('Plotting width comparison — vertical...')
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
print('  Saved fig_part2_width_comparison_vert.png')


# --- Figure: Spectrogram + dominant frequency tracking (middle width) ---
print('\nTracking dominant frequencies...')
w_mid = WIDTHS[1]
tslide, spec_x = results_x[w_mid]
_, spec_y = results_y[w_mid]

freqs_pos = k[freq_mask]
spec_x_pos = spec_x[:, freq_mask]
spec_y_pos = spec_y[:, freq_mask]

# dominant frequency at each time step
dom_fx = freqs_pos[np.argmax(spec_x_pos, axis=1)]
dom_fy = freqs_pos[np.argmax(spec_y_pos, axis=1)]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# horizontal spectrogram
axes[0, 0].pcolormesh(tslide, freqs_pos, spec_x_pos.T,
                       shading='auto', cmap='jet')
axes[0, 0].plot(tslide, dom_fx, 'w--', linewidth=0.8, alpha=0.7)
axes[0, 0].set_title(f'Spectrogram — Horizontal (a={w_mid})')
axes[0, 0].set_ylabel('Frequency (cyc/frame)')

# vertical spectrogram
axes[0, 1].pcolormesh(tslide, freqs_pos, spec_y_pos.T,
                       shading='auto', cmap='jet')
axes[0, 1].plot(tslide, dom_fy, 'w--', linewidth=0.8, alpha=0.7)
axes[0, 1].set_title(f'Spectrogram — Vertical (a={w_mid})')
axes[0, 1].set_ylabel('Frequency (cyc/frame)')

# dominant freq evolution
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
print('  Saved fig_part2_dominant_freq.png')


# =========================================================================
# Summary
# =========================================================================
print(f'\n{"="*55}')
print('Part 2 complete.')
print(f'{"="*55}')
print('Figures:')
print('  fig_part2_signal_fft.png          — signal + global FFT')
print('  fig_part2_gabor_illustration.png  — Gabor transform steps')
print('  fig_part2_width_comparison_horiz  — window width comparison (x)')
print('  fig_part2_width_comparison_vert   — window width comparison (y)')
print('  fig_part2_dominant_freq.png       — spectrograms + dom. freq.')