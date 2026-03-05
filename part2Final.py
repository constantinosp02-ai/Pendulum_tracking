"""
Part 2

Performs Gabor transform analysis on the displacement
data extracted in Part 1, using Camera 1.
"""

import numpy as np                          # numerical arrays and math
import matplotlib.pyplot as plt             # plotting

#Configuration

CAM = 1                                     # which camera to analyse

# Three window widths to demonstrate the time frequency trade off:
# narrow a = sharp time localisation, poor frequency resolution
# wide  a = sharp frequency resolution, poor time localisation
WIDTHS = [5, 15, 40]

TSLIDE_STEP = 2   # frames between consecutive window positions
MAX_FREQ = 0.15   # cycles/frame — upper limit for display


# Load data

data = np.load('tracking_results.npz')      # load Part 1 output
dx = data[f'cam{CAM}_x']                    # horizontal displacement
dy = data[f'cam{CAM}_y']                    # vertical displacement

# mean-subtract so the signal oscillates around zero
dx = dx - np.mean(dx)                       # remove DC offset from horizontal
dy = dy - np.mean(dy)                       # remove DC offset from vertical

n = len(dx)                                 # total number of frames
t = np.arange(n)                            # time axis (frame indices)

# frequency vector in cycles/frame
k = np.arange(-n//2, n//2) / n             # centred frequency axis for FFT plots

print(f'Camera {CAM}: {n} frames, freq resolution = {1/n:.5f} cyc/frame')


#Functions

def compute_fft(signal):
    """FFT magnitude spectrum, normalised to peak = 1."""
    mag = np.abs(np.fft.fftshift(np.fft.fft(signal)))  # FFT, shift zero-freq to centre, take magnitude
    return mag / np.max(mag)                             # normalise so peak = 1


def gabor_spectrogram(signal, width):
    """
    Slide a Gaussian window g(t) = exp(-(t-tau)^2 / a^2) across the signal,
    take the FFT at each position, and stack the magnitudes into a spectrogram.
    """
    tslide = np.arange(0, n, TSLIDE_STEP)              # window centre positions
    Sgt_spec = np.zeros((len(tslide), n))               # spectrogram matrix (time x freq)

    for j, tau in enumerate(tslide):                    # slide window across signal
        g = np.exp(-(t - tau)**2 / width**2)            # Gaussian window centred at tau
        Sgt = np.fft.fft(g * signal)                    # FFT of windowed signal
        mag = np.abs(np.fft.fftshift(Sgt))              # magnitude, shifted to centre
        Sgt_spec[j, :] = mag                            # store this time slice

    # normalise by global max so amplitude decay is visible across time
    if Sgt_spec.max() > 0:                              # avoid divide by zero
        Sgt_spec /= Sgt_spec.max()                      # scale to [0, 1]

    return tslide, Sgt_spec


# Figure 1: Signal + global FFT

mag_x = compute_fft(dx)                                # FFT of horizontal signal
mag_y = compute_fft(dy)                                # FFT of vertical signal

fig, axes = plt.subplots(2, 2, figsize=(13, 7))        # 2x2 grid: signals on top, FFTs below

axes[0, 0].plot(t, dx, 'k', linewidth=0.7)             # horizontal displacement vs time
axes[0, 0].set_title('Horizontal displacement')
axes[0, 0].set_xlabel('Frame number')
axes[0, 0].set_ylabel('Displacement (px)')
axes[0, 0].set_xlim([0, n])                            # full time range

axes[0, 1].plot(t, dy, 'k', linewidth=0.7)             # vertical displacement vs time
axes[0, 1].set_title('Vertical displacement')
axes[0, 1].set_xlabel('Frame number')
axes[0, 1].set_ylabel('Displacement (px)')
axes[0, 1].set_xlim([0, n])

axes[1, 0].plot(k, mag_x, 'b', linewidth=0.7)          # horizontal FFT spectrum
axes[1, 0].set_title('FFT — Horizontal')
axes[1, 0].set_xlabel('Frequency (cycles/frame)')
axes[1, 0].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[1, 0].set_xlim([-MAX_FREQ, MAX_FREQ])             # zoom to relevant freq range

axes[1, 1].plot(k, mag_y, 'b', linewidth=0.7)          # vertical FFT spectrum
axes[1, 1].set_title('FFT — Vertical')
axes[1, 1].set_xlabel('Frequency (cycles/frame)')
axes[1, 1].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[1, 1].set_xlim([-MAX_FREQ, MAX_FREQ])

plt.tight_layout()                                      # fix subplot spacing
plt.savefig('fig_part2_signal_fft.png', dpi=200, bbox_inches='tight')  # save figure
plt.close()


#Figure 2: Gabor illustration at a single time instant

tau_example = n // 2                                    # place window at the middle of the signal
w_example = WIDTHS[1]                                   # use medium width for illustration
g_example = np.exp(-(t - tau_example)**2 / w_example**2)  # Gaussian window
Sg_example = g_example * dx                             # windowed signal
mag_example = np.abs(np.fft.fftshift(np.fft.fft(Sg_example)))  # FFT of windowed signal
mag_example = mag_example / mag_example.max()           # normalise

fig, axes = plt.subplots(3, 1, figsize=(12, 8))        # 3 vertically stacked subplots

axes[0].plot(t, dx, 'k', linewidth=0.7, label='Signal $S(t)$')  # original signal
axes[0].plot(t, g_example * np.max(np.abs(dx)), 'r', linewidth=1.2,
             label=f'Gabor window ($\\tau$={tau_example}, a={w_example})')  # Gaussian window scaled for visibility
axes[0].set_title('Signal with Gabor function in time domain')
axes[0].set_xlabel('Frame number')
axes[0].set_ylabel('$S(t)$')
axes[0].legend()
axes[0].set_xlim([0, n])

axes[1].plot(t, Sg_example, 'm', linewidth=0.7)        # product of signal and window
axes[1].set_title('Windowed signal in time domain')
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('$S(t) \\cdot g(t)$')
axes[1].set_xlim([0, n])

axes[2].plot(k, mag_example, 'b', linewidth=0.7)       # frequency content at this time slice
axes[2].set_title('Fourier transform of the windowed signal')
axes[2].set_xlabel('Frequency (cycles/frame)')
axes[2].set_ylabel('$|F(\\omega)|$ (normalised)')
axes[2].set_xlim([-MAX_FREQ, MAX_FREQ])

plt.tight_layout()
plt.savefig('fig_part2_gabor_illustration.png', dpi=200, bbox_inches='tight')
plt.close()


# Compute spectrograms for all widths

print('Computing spectrograms...')
results_x = {}                                          # store horizontal spectrograms
results_y = {}                                          # store vertical spectrograms
for w in WIDTHS:                                        # loop over each window width
    print(f'  a = {w}')
    tslide, spec_x = gabor_spectrogram(dx, w)           # horizontal spectrogram
    _, spec_y = gabor_spectrogram(dy, w)                 # vertical spectrogram (same time axis)
    results_x[w] = (tslide, spec_x)                     # store result
    results_y[w] = (tslide, spec_y)

freq_mask = (k >= 0) & (k <= MAX_FREQ)                  # only positive freqs up to MAX_FREQ


#Figure 3: Width comparison — horizontal

fig, axes = plt.subplots(len(WIDTHS) + 1, 1, figsize=(13, 3 * (len(WIDTHS) + 1)))  # signal + 3 spectrograms

axes[0].plot(t, dx, 'k', linewidth=0.7)                 # raw horizontal signal on top
axes[0].set_ylabel('Disp. (px)')
axes[0].set_title(f'Camera {CAM} — Horizontal displacement')
axes[0].set_xlim([0, n])

for i, w in enumerate(WIDTHS):                           # one spectrogram per width
    tslide, spec = results_x[w]                          # get precomputed spectrogram
    axes[i+1].pcolormesh(tslide, k[freq_mask], spec[:, freq_mask].T,
                          shading='auto', cmap='jet')    # plot as heatmap
    axes[i+1].set_ylabel('Freq\n(cyc/frame)')
    axes[i+1].set_title(f'Gabor width a = {w} frames')

axes[-1].set_xlabel('Frame number')                      # x-label on bottom subplot only
plt.tight_layout()
plt.savefig('fig_part2_width_comparison_horiz.png', dpi=200, bbox_inches='tight')
plt.close()


#Figure 4: Width comparison - vertical

fig, axes = plt.subplots(len(WIDTHS) + 1, 1, figsize=(13, 3 * (len(WIDTHS) + 1)))  # same layout as fig 3

axes[0].plot(t, dy, 'k', linewidth=0.7)                 # raw vertical signal on top
axes[0].set_ylabel('Disp. (px)')
axes[0].set_title(f'Camera {CAM} — Vertical displacement')
axes[0].set_xlim([0, n])

for i, w in enumerate(WIDTHS):                           # one spectrogram per width
    tslide, spec = results_y[w]                          # get precomputed spectrogram
    axes[i+1].pcolormesh(tslide, k[freq_mask], spec[:, freq_mask].T,
                          shading='auto', cmap='jet')    # heatmap
    axes[i+1].set_ylabel('Freq\n(cyc/frame)')
    axes[i+1].set_title(f'Gabor width a = {w} frames')

axes[-1].set_xlabel('Frame number')
plt.tight_layout()
plt.savefig('fig_part2_width_comparison_vert.png', dpi=200, bbox_inches='tight')
plt.close()


#Figure 5: Spectrograms + dominant frequency tracking

w_mid = WIDTHS[1]                                        # use medium width for final analysis
tslide, spec_x = results_x[w_mid]                       # horizontal spectrogram
_, spec_y = results_y[w_mid]                             # vertical spectrogram

freqs_pos = k[freq_mask]                                 # positive frequency values only
spec_x_pos = spec_x[:, freq_mask]                        # crop spectrogram to positive freqs
spec_y_pos = spec_y[:, freq_mask]                        # same for vertical

dom_fx = freqs_pos[np.argmax(spec_x_pos, axis=1)]       # dominant freq at each time step (horizontal)
dom_fy = freqs_pos[np.argmax(spec_y_pos, axis=1)]       # dominant freq at each time step (vertical)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))         # 2x2: spectrograms on top, freq tracks below

axes[0, 0].pcolormesh(tslide, freqs_pos, spec_x_pos.T, shading='auto', cmap='jet')  # horiz spectrogram
axes[0, 0].plot(tslide, dom_fx, 'w--', linewidth=0.8, alpha=0.7)  # overlay dominant freq line
axes[0, 0].set_title(f'Spectrogram — Horizontal (a={w_mid})')
axes[0, 0].set_ylabel('Frequency (cyc/frame)')

axes[0, 1].pcolormesh(tslide, freqs_pos, spec_y_pos.T, shading='auto', cmap='jet')  # vert spectrogram
axes[0, 1].plot(tslide, dom_fy, 'w--', linewidth=0.8, alpha=0.7)  # dominant freq overlay
axes[0, 1].set_title(f'Spectrogram — Vertical (a={w_mid})')
axes[0, 1].set_ylabel('Frequency (cyc/frame)')

axes[1, 0].plot(tslide, dom_fx, 'b.-', markersize=2, linewidth=0.6)  # horiz dominant freq over time
axes[1, 0].set_title('Dominant frequency — Horizontal')
axes[1, 0].set_xlabel('Frame number')
axes[1, 0].set_ylabel('Frequency (cyc/frame)')
axes[1, 0].set_ylim([0, MAX_FREQ])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(tslide, dom_fy, 'r.-', markersize=2, linewidth=0.6)  # vert dominant freq over time
axes[1, 1].set_title('Dominant frequency — Vertical')
axes[1, 1].set_xlabel('Frame number')
axes[1, 1].set_ylabel('Frequency (cyc/frame)')
axes[1, 1].set_ylim([0, MAX_FREQ])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part2_dominant_freq.png', dpi=200, bbox_inches='tight')
plt.close()
