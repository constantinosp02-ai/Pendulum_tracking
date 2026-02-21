"""
MECH0107 CW1 — Part 3: Dimensionality Reduction
SVD and PCA (Lecture 2, Sec. 2.1-2.3)
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# Load tracked position data from all three cameras
# =========================================================================
print('Loading tracking data...')
data = np.load('tracking_results.npz')

# each camera gives x(t) and y(t) — 2 measurements per camera
cam1_x = data['cam1_x']
cam1_y = data['cam1_y']
cam2_x = data['cam2_x']
cam2_y = data['cam2_y']
cam3_x = data['cam3_x']
cam3_y = data['cam3_y']

# Truncate all signals to the shortest camera length
n = min(len(cam1_x), len(cam2_x), len(cam3_x))
cam1_x, cam1_y = cam1_x[:n], cam1_y[:n]
cam2_x, cam2_y = cam2_x[:n], cam2_y[:n]
cam3_x, cam3_y = cam3_x[:n], cam3_y[:n]
t = np.arange(n)  # frame numbers

# =========================================================================
# Step 1: Build the data matrix X (Lecture 2, Sec. 2.2, Eq. 45)
#
# Rows = measurements (x,y from each camera)
# Columns = time frames
#
#     X = [ cam1_x ]    shape: 6 x n
#         [ cam1_y ]
#         [ cam2_x ]
#         [ cam2_y ]
#         [ cam3_x ]
#         [ cam3_y ]
# =========================================================================
X = np.vstack([cam1_x, cam1_y,
               cam2_x, cam2_y,
               cam3_x, cam3_y])

m = X.shape[0]  # number of measurements (6)
print(f'Data matrix X: {m} measurements x {n} frames')

# =========================================================================
# Step 2: Compute the mean of each row (Lecture 2, Sec. 2.2, Eq. 46)
# =========================================================================
mn = np.mean(X, axis=1, keepdims=True)

# =========================================================================
# Step 3: Subtract mean — centre the data (Lecture 2, Sec. 2.2, Eq. 47)
# =========================================================================
B = X - mn

# =========================================================================
# Method 1: PCA via covariance matrix (Lecture 2, Sec. 2.2, Steps 4-7)
# =========================================================================
print('\n--- PCA ---')

# Step 4: Covariance matrix (Eq. 48)
C = (1 / (n - 1)) * B @ B.T  # m x m matrix

# Step 5: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(C)

# Step 6: Sort eigenvalues in decreasing order
sort_idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[sort_idx].real
eigenvectors = eigenvectors[:, sort_idx].real

# variance explained by each component
variance_pca = eigenvalues / np.sum(eigenvalues)
print('PCA variance explained:')
for i in range(m):
    print(f'  Component {i+1}: {variance_pca[i]*100:.2f}%')

# Step 7: Project data onto principal components (Eq. 52)
Y_pca = eigenvectors.T @ B

# =========================================================================
# Method 2: SVD (Lecture 2, Sec. 2.1, matching MATLAB code)
# =========================================================================
print('\n--- SVD ---')

# SVD of the normalised mean-centred matrix (matches MATLAB: svd(X/sqrt(n-1)))
U, S, Vt = np.linalg.svd(B / np.sqrt(n - 1), full_matrices=False)

# singular values squared = eigenvalues
lambda_svd = S**2
variance_svd = lambda_svd / np.sum(lambda_svd)
print('SVD variance explained:')
for i in range(m):
    print(f'  Mode {i+1}: {variance_svd[i]*100:.2f}%')

# SVD modes (projections)
Y_svd = U.T @ B

# =========================================================================
# Figure: PCA results — eigenvalue spectrum + modes
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# eigenvalue spectrum
axes[0].plot(np.arange(1, m+1), variance_pca * 100, 'bo-', linewidth=1.2,
             markersize=8)
axes[0].set_xlabel('PCA Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('PCA Eigenvalue Spectrum')
axes[0].set_xticks(np.arange(1, m+1))
axes[0].grid(True, alpha=0.3)

# first 3 modes
for i, (col, label) in enumerate(zip(['b', 'r', 'g'],
                                      ['Mode 1', 'Mode 2', 'Mode 3'])):
    axes[1].plot(t, Y_pca[i, :], col, linewidth=0.9, label=label)
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('Displacement Position')
axes[1].set_title('PCA Scores (first 3 modes)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part3_pca.png', dpi=200, bbox_inches='tight')
plt.close()
print('\n  Saved fig_part3_pca.png')

# =========================================================================
# Figure: SVD results — singular value spectrum + modes
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# singular value spectrum
axes[0].plot(np.arange(1, m+1), variance_svd * 100, 'bo-', linewidth=1.2,
             markersize=8)
axes[0].set_xlabel('SVD Singular Value')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('SVD Singular Value Spectrum')
axes[0].set_xticks(np.arange(1, m+1))
axes[0].grid(True, alpha=0.3)

# first 3 modes
for i, (col, label) in enumerate(zip(['b', 'r', 'g'],
                                      ['Mode 1', 'Mode 2', 'Mode 3'])):
    axes[1].plot(t, Y_svd[i, :], col, linewidth=0.9, label=label)
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('Displacement Position')
axes[1].set_title('SVD Modes (first 3 modes)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part3_svd.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved fig_part3_svd.png')

# =========================================================================
# Figure: Cumulative variance — how many modes are needed
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 5))
cumvar = np.cumsum(variance_svd) * 100
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
print('  Saved fig_part3_cumulative_variance.png')

# =========================================================================
# Figure: Individual mode interpretation
# =========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

labels = ['Mode 1', 'Mode 2', 'Mode 3']
colors = ['b', 'r', 'g']

for i in range(3):
    axes[i].plot(t, Y_svd[i, :], colors[i], linewidth=0.9)
    axes[i].set_ylabel('Amplitude')
    axes[i].set_title(f'{labels[i]} — {variance_svd[i]*100:.1f}% of variance')
    axes[i].grid(True, alpha=0.3)

axes[2].set_xlabel('Frame number')
plt.tight_layout()
plt.savefig('fig_part3_modes_individual.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved fig_part3_modes_individual.png')

# =========================================================================
# Figure: Spatial structure of modes (what each mode "looks like" per camera)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

row_labels = ['Cam1 x', 'Cam1 y', 'Cam2 x', 'Cam2 y', 'Cam3 x', 'Cam3 y']
x_pos_bar = np.arange(m)
width = 0.25

for i in range(3):
    ax.bar(x_pos_bar + i * width, U[:, i], width, label=f'Mode {i+1}',
           alpha=0.8)

ax.set_xticks(x_pos_bar + width)
ax.set_xticklabels(row_labels)
ax.set_ylabel('Weight (left singular vector)')
ax.set_title('Spatial Structure of SVD Modes')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig_part3_mode_structure.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved fig_part3_mode_structure.png')

# =========================================================================
# Figure: Low-rank reconstruction
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# reconstruct using only the first k modes
for k, ax_col in zip([1, 2], [0, 1]):
    # X_approx = U_k * S_k * Vt_k (truncated SVD) + mean
    X_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :] * np.sqrt(n - 1) + mn

    # show cam1 x (row 0) and cam1 y (row 1) as example
    axes[0, ax_col].plot(t, X[0, :], 'k', linewidth=0.7, label='Original')
    axes[0, ax_col].plot(t, X_approx[0, :], 'b--', linewidth=0.9,
                          label=f'Rank-{k} approx')
    axes[0, ax_col].set_title(f'Cam1 horizontal — rank {k}')
    axes[0, ax_col].set_ylabel('Position (px)')
    axes[0, ax_col].legend(fontsize=9)
    axes[0, ax_col].grid(True, alpha=0.3)

    axes[1, ax_col].plot(t, X[1, :], 'k', linewidth=0.7, label='Original')
    axes[1, ax_col].plot(t, X_approx[1, :], 'r--', linewidth=0.9,
                          label=f'Rank-{k} approx')
    axes[1, ax_col].set_title(f'Cam1 vertical — rank {k}')
    axes[1, ax_col].set_xlabel('Frame number')
    axes[1, ax_col].set_ylabel('Position (px)')
    axes[1, ax_col].legend(fontsize=9)
    axes[1, ax_col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_part3_reconstruction.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved fig_part3_reconstruction.png')

# =========================================================================
# Summary
# =========================================================================
print(f'\n{"="*55}')
print('Part 3 complete.')
print(f'{"="*55}')
print('Figures:')
print('  fig_part3_pca.png                — PCA eigenvalues + scores')
print('  fig_part3_svd.png                — SVD singular values + modes')
print('  fig_part3_cumulative_variance.png — cumulative variance')
print('  fig_part3_modes_individual.png   — first 3 modes separately')
print('  fig_part3_mode_structure.png     — spatial weights of each mode')
print('  fig_part3_reconstruction.png     — low-rank signal reconstruction')