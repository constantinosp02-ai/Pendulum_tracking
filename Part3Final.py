"""
MECH0107 Coursework — Part 3: Dimensionality Reduction

Applies SVD to the multi-camera displacement data from Part 1 to
identify the dominant modes of motion of the spring-pendulum system.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────

data = np.load('tracking_results.npz')

cam1_x = data['cam1_x'];  cam1_y = data['cam1_y']
cam2_x = data['cam2_x'];  cam2_y = data['cam2_y']
cam3_x = data['cam3_x'];  cam3_y = data['cam3_y']

# truncate to the shortest camera recording
n = min(len(cam1_x), len(cam2_x), len(cam3_x))
cam1_x, cam1_y = cam1_x[:n], cam1_y[:n]
cam2_x, cam2_y = cam2_x[:n], cam2_y[:n]
cam3_x, cam3_y = cam3_x[:n], cam3_y[:n]
t = np.arange(n)


# ── Step 1: Build data matrix — rows = measurements, cols = frames ─────

X = np.vstack([cam1_x, cam1_y,
               cam2_x, cam2_y,
               cam3_x, cam3_y])

m = X.shape[0]
print(f'Data matrix: {m} x {n}')


# ── Step 2 & 3: Mean-subtract each row ────────────────────────────────

mn = np.mean(X, axis=1, keepdims=True)
B = X - mn


# ── Step 4: SVD ────────────────────────────────────────────────────────
# Dividing by sqrt(n-1) so that S^2 gives eigenvalues of the covariance
# matrix directly — equivalent to MATLAB: [U,S,V] = svd(X/sqrt(n-1), 'econ')

U, S, Vt = np.linalg.svd(B / np.sqrt(n - 1), full_matrices=False)

variance = S**2 / np.sum(S**2)
Y = U.T @ B   # project data onto modes

print('\nVariance explained:')
for i in range(m):
    print(f'  Mode {i+1}: {variance[i]*100:.1f}%')


# ── Figure 1: Singular value spectrum + first 3 modes ─────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(np.arange(1, m+1), variance * 100, 'bo-', linewidth=1.2, markersize=8)
axes[0].set_xlabel('Mode')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('SVD Singular Value Spectrum')
axes[0].set_xticks(np.arange(1, m+1))
axes[0].grid(True, alpha=0.3)

for i, (col, label) in enumerate(zip(['b', 'r', 'g'], ['Mode 1', 'Mode 2', 'Mode 3'])):
    axes[1].plot(t, Y[i, :], col, linewidth=0.9, label=label)
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('SVD Modes (first 3)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part3_svd.png', dpi=200, bbox_inches='tight')
plt.close()


# ── Figure 2: Cumulative variance ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
cumvar = np.cumsum(variance) * 100
ax.plot(np.arange(1, m+1), cumvar, 'ro-', markersize=8, linewidth=1.2)
ax.axhline(y=95, color='k', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Number of Modes')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('Cumulative Variance — SVD')
ax.set_xticks(np.arange(1, m+1))
ax.set_ylim([0, 105])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_part3_cumulative_variance.png', dpi=200, bbox_inches='tight')
plt.close()


# ── Figure 3: Individual modes ─────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i, col in enumerate(['b', 'r', 'g']):
    axes[i].plot(t, Y[i, :], col, linewidth=0.9)
    axes[i].set_ylabel('Amplitude')
    axes[i].set_title(f'Mode {i+1} — {variance[i]*100:.1f}% of variance')
    axes[i].grid(True, alpha=0.3)
axes[2].set_xlabel('Frame number')
plt.tight_layout()
plt.savefig('fig_part3_modes_individual.png', dpi=200, bbox_inches='tight')
plt.close()


# ── Figure 4: Spatial structure (left singular vectors) ───────────────

fig, ax = plt.subplots(figsize=(10, 5))
row_labels = ['Cam1 x', 'Cam1 y', 'Cam2 x', 'Cam2 y', 'Cam3 x', 'Cam3 y']
x_pos = np.arange(m)
bar_w = 0.25
for i in range(3):
    ax.bar(x_pos + i * bar_w, U[:, i], bar_w, label=f'Mode {i+1}', alpha=0.8)
ax.set_xticks(x_pos + bar_w)
ax.set_xticklabels(row_labels)
ax.set_ylabel('Weight (left singular vector)')
ax.set_title('Spatial Structure of SVD Modes')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig_part3_mode_structure.png', dpi=200, bbox_inches='tight')
plt.close()


# ── Figure 5: Low-rank reconstruction ─────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for rank, ax_col in zip([1, 2], [0, 1]):
    X_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :] * np.sqrt(n - 1) + mn

    axes[0, ax_col].plot(t, X[0, :], 'k', linewidth=0.7, label='Original')
    axes[0, ax_col].plot(t, X_approx[0, :], 'b--', linewidth=0.9, label=f'Rank-{rank} approx')
    axes[0, ax_col].set_title(f'Cam1 horizontal — rank {rank}')
    axes[0, ax_col].set_ylabel('Position (px)')
    axes[0, ax_col].legend(fontsize=9)
    axes[0, ax_col].grid(True, alpha=0.3)

    axes[1, ax_col].plot(t, X[1, :], 'k', linewidth=0.7, label='Original')
    axes[1, ax_col].plot(t, X_approx[1, :], 'r--', linewidth=0.9, label=f'Rank-{rank} approx')
    axes[1, ax_col].set_title(f'Cam1 vertical — rank {rank}')
    axes[1, ax_col].set_xlabel('Frame number')
    axes[1, ax_col].set_ylabel('Position (px)')
    axes[1, ax_col].legend(fontsize=9)
    axes[1, ax_col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part3_reconstruction.png', dpi=200, bbox_inches='tight')
plt.close()

print('Part 3 done.')
