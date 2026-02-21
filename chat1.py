"""
=============================================================================
MECH0107 Coursework Part 1 — Motion Tracking
=============================================================================
This script extracts the position of an oscillating mass (paint can) from
video frames captured by three cameras.

Every image-processing step follows techniques taught in MECH0107 Lecture 4
(Image Processing & Analysis):

    1. RGB channel combination  (Lecture 4, Sec. 1.1.1)
       Camera 1: "pink channel" — R + B − 2G isolates the pink marker
           on the paint can.  Pink has high Red, high Blue, low Green.
       Camera 2: custom "yellow channel" to isolate the yellow marker
           Y_ch = (R + G)/2 − B
       Camera 3: standard grayscale via luminosity weights
           Gray = 0.2989·R + 0.5870·G + 0.1140·B
       All three are arithmetic on the (R, G, B) pixel triplet
       described in Lecture 4, Sec. 1.1.1.

    2. Gaussian filtering in the FREQUENCY DOMAIN  (Lecture 4, Sec. 1.2.1.1)
       Procedure: FFT → fftshift → multiply by Gaussian kernel → ifftshift → IFFT
       Kernel:  G(kx,ky) = exp( -σx·(kx-a)² - σy·(ky-b)² )
       This matches ImageProcessing_Filtering.m from the lectures exactly.
       Chosen over the Shannon filter because its smooth roll-off avoids
       Gibbs ringing artefacts caused by the Shannon filter's sharp rectangular
       cutoff (Lecture 4, Sec. 1.2.1.2).

    3. Binary thresholding  (Lecture 4, Sec. 1.1.2 & 1.2.2.2)
       Threshold value informed by the intensity histogram of each camera.
       Analogous to the FFT magnitude thresholding used for image compression
       where a binary mask isolates significant components.

    4. Morphological cleanup  (Lecture 4, Sec. 1.2)
       Connected component labelling breaks the binary image into 2D subsets
       based on shape and size, filtering out noise blobs.

    5. Centroid computation
       The mean (x, y) position of all object pixels gives a single position
       coordinate per frame for the tracked mass.

The same five-step pipeline is applied to all three cameras. The only
per-camera differences are: (i) the channel combination used in Step 1
(pink for Camera 1, yellow for Camera 2, grayscale for Camera 3),
and (ii) the brightness threshold, tuned via each camera's histogram.

Input:   cam1.mat, cam2.mat, cam3.mat
Output:  tracking_results.npz, diagnostic and trajectory figures
=============================================================================
"""

# =========================================================================
# Import Libraries
# =========================================================================
import numpy as np                        # numerical array operations
import scipy.io as sio                    # reading MATLAB .mat files
from scipy.ndimage import label, rotate   # connected component labelling + rotation
import matplotlib.pyplot as plt           # plotting and visualisation
import os                                 # file path handling

# =========================================================================
# Configuration — Per-Camera Parameters
# =========================================================================
# Variable names inside each .mat file (set by whoever recorded the data)
VID_KEYS = {
    1: 'vidFrames1_4',
    2: 'vidFrames2_4',
    3: 'vidFrames3_4',
}

# File path template
DATA_PATH = 'cam{}.mat'

def rotate_cam3_video(vid):
    """
    Rotate Camera 3 video: 90° clockwise then an additional 24° clockwise.

    The raw Camera 3 frames are rotated in the .mat file.  This function
    first applies a 90° clockwise rotation (exact, no interpolation) and
    then an additional 24° clockwise rotation using scipy.ndimage.rotate
    with reshape=False to keep the frame dimensions constant.

    Parameters
    ----------
    vid : ndarray, shape (H, W, 3, N)
        Raw video array.

    Returns
    -------
    rotated : ndarray, same dtype as input
        Rotated video array.
    """
    # Step 1: exact 90° CW rotation
    vid = np.rot90(vid, k=-1, axes=(0, 1))      # (480,640,3,N) → (640,480,3,N)
    # Step 2: additional 24° CW  (scipy uses CCW positive, so angle = -24)
    nframes = vid.shape[3]
    for i in range(nframes):
        vid[:, :, :, i] = rotate(vid[:, :, :, i], angle=-24,
                                  reshape=False, order=1)
    return vid


# Region of Interest for each camera: (y_min, y_max, x_min, x_max)
# Cropping to a region around the mass avoids bright distractors elsewhere
# in the frame (ceiling lights, reflections, etc.)
ROIS = {
    1: (150, 430, 280, 500),
    2: (80, 410, 180, 480),
    3: (200, 520, 129, 349),   # after 90° CW rotation of Camera 3
}

# Channel mode for each camera:
#   'grayscale' — standard luminosity conversion (Lecture 4, Sec. 1.1.1)
#   'yellow'    — custom (R+G)/2 - B channel to isolate the yellow marker
#                 on the paint can (also based on RGB pixel model, Sec. 1.1.1)
#   'pink'      — R + B - 2G to isolate the pink marker on the paint can
CHANNEL_MODE = {
    1: 'pink',
    2: 'yellow',
    3: 'grayscale',
}

# Binary threshold for the binary mask
# Camera 1: pink channel (R+B−2G, typical range 0–80)
# Camera 2: yellow channel response (different scale, typically 0–120)
# Camera 3: grayscale intensity (0–255 range)
# Chosen by examining the intensity histogram of each camera's frames.
THRESHOLDS = {
    1: 20,
    2: 60,
    3: 215,
}

# Gaussian filter width σ for the frequency-domain filter (Lecture 4, Eq. 1)
# Smaller σ → stronger low-pass (more denoising, more blur)
# Larger  σ → weaker low-pass  (less denoising, preserves detail)
GAUSS_SIGMA = {
    1: 0.0008,
    2: 0.0008,
    3: 0.0008,
}

# Minimum blob area (pixels) — blobs smaller than this are noise
MIN_BLOB_SIZE = 20

# Maximum blob area (pixels) — blobs larger than this are background
MAX_BLOB_SIZE = 8000


# =========================================================================
# STEP 1: RGB to Grayscale Conversion
# =========================================================================
def rgb_to_grayscale(rgb_image):
    """
    Convert an RGB image to grayscale using the luminosity method.

    The standard perceptual weights are:
        Gray = 0.2989 × R  +  0.5870 × G  +  0.1140 × B

    These weights account for the human eye's different sensitivity to
    each colour channel. This is the same conversion used by MATLAB's
    rgb2gray() and is described in Lecture 4, Section 1.1.1.

    Parameters
    ----------
    rgb_image : ndarray, shape (H, W, 3)
        Input image in RGB format (values 0–255).

    Returns
    -------
    gray : ndarray, shape (H, W)
        Grayscale image as float64 (values 0–255).
    """
    # extract individual colour channels
    R = rgb_image[:, :, 0].astype(np.float64)   # red channel
    G = rgb_image[:, :, 1].astype(np.float64)   # green channel
    B = rgb_image[:, :, 2].astype(np.float64)   # blue channel

    # apply the luminosity weights to produce the grayscale image
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return gray


def rgb_to_yellow_channel(rgb_image):
    """
    Compute a "yellow channel" image from RGB pixel values.

    This is a custom linear combination of the RGB channels designed to
    isolate yellow-coloured regions.  Yellow in RGB has high Red, high Green,
    and low Blue, so:

        Y_ch = (R + G) / 2  −  B

    Colour responses:
        Yellow marker:  (high + high)/2 − low  =  large positive value
        White wall:     (high + high)/2 − high ≈  0 (neutral)
        Dark jeans:     (low  + low )/2 − low  ≈  0 (neutral)
        Grey floor:     (mid  + mid )/2 − mid  ≈  0 (neutral)

    This approach builds directly on the RGB pixel model from Lecture 4,
    Section 1.1.1: each pixel stores three channel values (R, G, B) in the
    range 0–255, and any arithmetic combination of these channels produces
    a new single-channel image.  The standard grayscale conversion is itself
    just one such combination (with luminosity weights).  Here, the weights
    are chosen to maximise contrast for the yellow marker specifically.

    Negative values are clipped to zero since they correspond to blue-ish
    pixels (the opposite of what we are looking for).

    Parameters
    ----------
    rgb_image : ndarray, shape (H, W, 3)
        Input image in RGB format (values 0–255).

    Returns
    -------
    yellow : ndarray, shape (H, W)
        Yellow-channel image as float64.  High values = likely yellow.
    """
    # extract individual colour channels (Lecture 4, Sec. 1.1.1)
    R = rgb_image[:, :, 0].astype(np.float64)   # red channel
    G = rgb_image[:, :, 1].astype(np.float64)   # green channel
    B = rgb_image[:, :, 2].astype(np.float64)   # blue channel

    # compute the yellow channel: average of R and G minus B
    yellow = (R + G) / 2.0 - B

    # clip negative values to zero (non-yellow pixels)
    yellow = np.clip(yellow, 0, None)

    return yellow


def rgb_to_pink_channel(rgb_image):
    """
    Compute a "pink channel" image from RGB pixel values.

    This is a custom linear combination designed to isolate pink/magenta
    regions.  Pink in RGB has high Red, high Blue, and low Green, so:

        P = R + B − 2·G

    Colour responses:
        Pink marker:   (high + high) − 2·low   =  large positive value
        White wall:    (high + high) − 2·high   ≈  0 (neutral)
        Blue jeans:    (low  + high) − 2·low    ≈  moderate (but filtered by ROI)
        Black can:     (low  + low)  − 2·low    ≈  0 (neutral)
        Skin:          (high + low)  − 2·mid    ≈  0 (neutral)

    Like the yellow channel and standard grayscale, this is an arithmetic
    combination of the (R, G, B) pixel values (Lecture 4, Sec. 1.1.1).

    Negative values are clipped to zero since they correspond to green-ish
    pixels (the opposite of what we are looking for).

    Parameters
    ----------
    rgb_image : ndarray, shape (H, W, 3)
        Input image in RGB format (values 0–255).

    Returns
    -------
    pink : ndarray, shape (H, W)
        Pink-channel image as float64. High values = likely pink.
    """
    R = rgb_image[:, :, 0].astype(np.float64)
    G = rgb_image[:, :, 1].astype(np.float64)
    B = rgb_image[:, :, 2].astype(np.float64)

    pink = R + B - 2.0 * G
    pink = np.clip(pink, 0, None)

    return pink


def rgb_to_dark_yellowgreen_channel(rgb_image):
    """
    Compute a "dark yellow-green channel" from RGB pixel values.

    This channel isolates dark yellow-green regions by combining a
    colour signal (G − B) with a darkness weighting factor.  The idea
    is that the dark yellow-green stripe on the paint can has high Green
    relative to Blue, but low overall brightness compared to the white
    wall or bright yellow areas:

        colour  = max(G − B, 0)
        darkness = clip(150 − brightness, 0, 150) / 150
        output  = colour × darkness

    Colour responses:
        Dark yellow-green stripe: high G−B, low brightness → large value
        Bright wall / paper:      moderate G−B, high brightness → suppressed
        Black can body:           low G−B, low brightness → near zero
        Background:               variable, but mostly bright → suppressed

    Like the other custom channels, this is an arithmetic combination of
    the (R, G, B) pixel values (Lecture 4, Sec. 1.1.1), extended with a
    brightness-based weighting.

    Parameters
    ----------
    rgb_image : ndarray, shape (H, W, 3)
        Input image in RGB format (values 0–255).

    Returns
    -------
    dark_yg : ndarray, shape (H, W)
        Dark yellow-green channel as float64.  High values = likely the
        dark yellow-green stripe.
    """
    R = rgb_image[:, :, 0].astype(np.float64)
    G = rgb_image[:, :, 1].astype(np.float64)
    B = rgb_image[:, :, 2].astype(np.float64)

    brightness = (R + G + B) / 3.0
    colour_signal = np.clip(G - B, 0, None)
    darkness = np.clip(150.0 - brightness, 0, 150.0) / 150.0
    dark_yg = colour_signal * darkness

    return dark_yg


# =========================================================================
# STEP 2: Gaussian Filter in the Frequency Domain
# =========================================================================
def gaussian_filter_freq(image, sigma):
    """
    Apply a Gaussian low-pass filter in the frequency domain.

    This is a direct Python translation of the MATLAB code from
    ImageProcessing_Filtering.m taught in Lecture 4, Section 1.2.1.1.

    The procedure follows these steps:
        1. Compute the 2D FFT of the image              — fft2()
        2. Shift zero-frequency to the centre            — fftshift()
        3. Build the Gaussian kernel G(kx,ky)            — Equation (1)
        4. Multiply the shifted spectrum by the kernel   — element-wise ×
        5. Shift back                                    — ifftshift()
        6. Inverse FFT to return to the spatial domain   — ifft2()

    The Gaussian kernel is defined as (Lecture 4, Eq. 1):
        G(kx, ky) = exp( -σx·(kx - a)²  -  σy·(ky - b)² )
    where:
        (a, b) = centre of the frequency domain
        σx, σy = filter widths (here set equal: σx = σy = sigma)
        (kx, ky) = frequency coordinates

    The Gaussian filter is preferred over the Shannon filter because:
      - Its smooth frequency response avoids ringing artefacts
      - The inverse FT of a Gaussian is always a Gaussian (smooth in
        both domains), whereas the inverse FT of a rectangular window
        produces sinc-like ringing (Lecture 4, Sec. 1.2.1.2)

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image (float64).
    sigma : float
        Filter width parameter. Smaller = more denoising.

    Returns
    -------
    filtered : ndarray, shape (H, W)
        Denoised image in the spatial domain.
    """
    # get image dimensions (rows = height, cols = width)
    rows, cols = image.shape

    # --- Step 1: 2D Fast Fourier Transform ---
    # Transforms the image from the spatial domain to the frequency domain
    image_fft = np.fft.fft2(image)

    # --- Step 2: Shift the zero-frequency component to the centre ---
    # After fft2, the DC component is at (0,0). fftshift moves it to
    # the centre of the array so that the Gaussian kernel is centred.
    # This matches MATLAB: Bts = fftshift(Bt)
    image_fft_shifted = np.fft.fftshift(image_fft)

    # --- Step 3: Construct the 2D Gaussian filter kernel ---
    # Define the frequency coordinate arrays (1-indexed like MATLAB)
    # MATLAB:  kx = 1:size(Abw,2);  ky = 1:size(Abw,1);
    kx = np.arange(1, cols + 1)     # frequency coordinates along x
    ky = np.arange(1, rows + 1)     # frequency coordinates along y

    # Create a 2D meshgrid of frequency coordinates
    # MATLAB:  [KX, KY] = meshgrid(kx, ky);
    KX, KY = np.meshgrid(kx, ky)

    # Centre of the frequency domain
    # MATLAB:  mid_width  = size(Abw,2)/2;
    #          mid_height = size(Abw,1)/2;
    a = cols / 2                     # centre-frequency in x (width)
    b = rows / 2                     # centre-frequency in y (height)

    # Build the Gaussian kernel using Lecture 4 Equation (1):
    # G(kx,ky) = exp( -σ·(kx - a)² - σ·(ky - b)² )
    # MATLAB:  F = exp(-width*(KX-mid_width).^2 - width*(KY-mid_height).^2);
    G = np.exp(-sigma * (KX - a)**2 - sigma * (KY - b)**2)

    # --- Step 4: Apply the filter by element-wise multiplication ---
    # In the frequency domain, convolution becomes multiplication.
    # MATLAB:  Btsf = Bts .* F;
    filtered_fft_shifted = image_fft_shifted * G

    # --- Step 5: Shift back to the original FFT layout ---
    # MATLAB:  Btf = ifftshift(Btsf);
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)

    # --- Step 6: Inverse FFT to return to the spatial domain ---
    # MATLAB:  Bf = ifft2(Btf);
    filtered = np.fft.ifft2(filtered_fft)

    # Take the real part (imaginary component is negligible rounding error)
    return np.real(filtered)


# =========================================================================
# STEP 3: Binary Thresholding
# =========================================================================
def apply_threshold(image, threshold):
    """
    Create a binary image by thresholding on pixel intensity.

    Pixels with intensity > threshold are set to 1 (object).
    Pixels with intensity ≤ threshold are set to 0 (background).

    The threshold value is selected by examining the image histogram
    (Lecture 4, Sec. 1.1.2): the histogram shows the distribution of
    pixel intensities, and we choose a value that separates the bright
    mass (paint can with yellow label) from the darker background.

    This concept is analogous to the FFT magnitude thresholding used in
    image compression (Lecture 4, Sec. 1.2.2.2), where a binary mask
    is constructed: freq = |Xf| > threshold.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale (possibly filtered) image.
    threshold : float
        Intensity cutoff (0–255 range).

    Returns
    -------
    binary : ndarray, shape (H, W)
        Binary image (0s and 1s).
    """
    # create the binary mask
    binary = (image > threshold).astype(np.float64)
    return binary


# =========================================================================
# STEP 4: Morphological Cleanup (Connected Component Analysis)
# =========================================================================
def morphological_cleanup(binary_image, min_size, max_size):
    """
    Clean a binary image by removing blobs that are too small or too large.

    Connected component labelling assigns a unique integer label to each
    contiguous group of white pixels (blob). We then filter by area:
      - Blobs smaller than min_size are noise artefacts → remove
      - Blobs larger than max_size are background regions → remove

    This is the morphological approach described in Lecture 4, Section 1.2:
    "breaks an image, which is a 2D plane, into 2D subsets or domains
    based on their shapes."

    Parameters
    ----------
    binary_image : ndarray, shape (H, W)
        Binary image (0s and 1s).
    min_size : int
        Minimum blob area in pixels.
    max_size : int
        Maximum blob area in pixels.

    Returns
    -------
    cleaned : ndarray, shape (H, W)
        Cleaned binary image.
    blob_info : list of tuples
        List of (centroid_x, centroid_y, area) for each valid blob.
    """
    # label each connected component with a unique integer
    labelled, num_features = label(binary_image)

    # prepare output
    cleaned = np.zeros_like(binary_image)
    blob_info = []

    # build coordinate grids for centroid calculation
    rows, cols = binary_image.shape
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # examine each blob
    for k in range(1, num_features + 1):
        # create a mask for this blob only
        mask = (labelled == k)

        # compute the blob area (number of pixels)
        area = int(np.sum(mask))

        # reject blobs outside the acceptable size range
        if area < min_size or area > max_size:
            continue

        # compute the centroid (mean position of all blob pixels)
        cx = float(np.sum(xx * mask)) / area   # horizontal centre (column)
        cy = float(np.sum(yy * mask)) / area   # vertical centre (row)

        # keep this blob
        cleaned[mask] = 1
        blob_info.append((cx, cy, area))

    return cleaned, blob_info


# =========================================================================
# STEP 5: Centroid Selection
# =========================================================================
def select_best_blob(blob_info, prev_cx, prev_cy):
    """
    Select the most likely blob for the tracked object.

    If only one valid blob exists, use it.  If multiple blobs exist and
    we have a previous position, pick the one closest to the last known
    position (spatial continuity).  If this is the first frame, pick the
    largest blob.

    Parameters
    ----------
    blob_info : list of (cx, cy, area) tuples
    prev_cx, prev_cy : float or None
        Previous frame's centroid position.

    Returns
    -------
    cx, cy : float
        Selected centroid position.
    """
    if len(blob_info) == 0:
        # no valid blobs found — return previous position or centre of ROI
        return prev_cx, prev_cy

    if len(blob_info) == 1:
        # single blob — use it
        return blob_info[0][0], blob_info[0][1]

    if prev_cx is not None:
        # multiple blobs — pick the closest to the previous frame's centroid
        best = min(blob_info,
                   key=lambda b: (b[0] - prev_cx)**2 + (b[1] - prev_cy)**2)
        return best[0], best[1]
    else:
        # first frame — pick the largest blob
        best = max(blob_info, key=lambda b: b[2])
        return best[0], best[1]


# =========================================================================
# Full Tracking Pipeline for One Camera
# =========================================================================
def track_camera(cam_id):
    """
    Run the complete tracking pipeline for a single camera.

    For every frame:
        1. Crop to the region of interest
        2. Convert RGB → single channel  (Lecture 4, Sec. 1.1.1)
           - Camera 1: pink channel R + B − 2G
           - Camera 2: yellow channel (R+G)/2 − B
           - Camera 3: standard grayscale (luminosity weights)
        3. Gaussian filter in frequency domain  (Lecture 4, Sec. 1.2.1.1)
        4. Binary thresholding  (Lecture 4, Sec. 1.1.2)
        5. Morphological cleanup  (Lecture 4, Sec. 1.2)
        6. Centroid extraction

    Parameters
    ----------
    cam_id : int
        Camera number (1, 2, or 3).

    Returns
    -------
    x_pos, y_pos : ndarray
        Centroid positions (in full-frame pixel coordinates) for each frame.
    """
    # -- Load video data from .mat file ------------------------------------
    filepath = DATA_PATH.format(cam_id)
    print(f'  Loading {filepath}...')
    data = sio.loadmat(filepath)
    vid = data[VID_KEYS[cam_id]]            # shape: (480, 640, 3, nframes)

    # Camera 3 frames are rotated in the raw data.
    # Rotate 90° + 24° clockwise to get the correct orientation.
    if cam_id == 3:
        vid = rotate_cam3_video(vid)
        print(f'  Rotated Camera 3 frames 90°+24° clockwise')

    nframes = vid.shape[3]
    print(f'  Video shape: {vid.shape}  →  {nframes} frames')

    # -- Unpack per-camera parameters --------------------------------------
    y1, y2, x1, x2 = ROIS[cam_id]
    threshold = THRESHOLDS[cam_id]
    sigma = GAUSS_SIGMA[cam_id]
    mode = CHANNEL_MODE[cam_id]
    print(f'  Channel mode: {mode}  |  Threshold: {threshold}  |  σ: {sigma}')

    # -- Pre-compute the Gaussian filter kernel ----------------------------
    # The ROI size is the same for every frame, so we build the kernel once.
    roi_h = y2 - y1                          # ROI height in pixels
    roi_w = x2 - x1                          # ROI width in pixels

    # frequency coordinate arrays (1-indexed to match MATLAB convention)
    kx = np.arange(1, roi_w + 1)
    ky = np.arange(1, roi_h + 1)
    KX, KY = np.meshgrid(kx, ky)

    # centre of the frequency domain
    a = roi_w / 2
    b = roi_h / 2

    # Gaussian kernel — Lecture 4, Equation (1)
    G = np.exp(-sigma * (KX - a)**2 - sigma * (KY - b)**2)

    # -- Preallocate output arrays -----------------------------------------
    x_pos = np.full(nframes, np.nan)
    y_pos = np.full(nframes, np.nan)

    # previous centroid for spatial continuity
    prev_cx, prev_cy = None, None

    # -- Frame-by-frame processing -----------------------------------------
    for i in range(nframes):
        # extract this frame as float64
        rgb_frame = vid[:, :, :, i].astype(np.float64)

        # STEP 1: crop to the region of interest
        roi_rgb = rgb_frame[y1:y2, x1:x2, :]

        # STEP 2: convert RGB → single-channel image
        # All are linear combinations of RGB channels (Lecture 4, Sec. 1.1.1)
        if mode == 'yellow':
            gray = rgb_to_yellow_channel(roi_rgb)
        elif mode == 'pink':
            gray = rgb_to_pink_channel(roi_rgb)
        elif mode == 'dark_yellowgreen':
            gray = rgb_to_dark_yellowgreen_channel(roi_rgb)
        else:
            gray = rgb_to_grayscale(roi_rgb)

        # STEP 3: Gaussian filter in the frequency domain
        # FFT → fftshift → multiply by G → ifftshift → IFFT
        fft_img = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_img)
        fft_filtered = fft_shifted * G          # apply the Gaussian kernel
        fft_unshifted = np.fft.ifftshift(fft_filtered)
        filtered = np.real(np.fft.ifft2(fft_unshifted))

        # STEP 4: binary thresholding
        binary = apply_threshold(filtered, threshold)

        # STEP 5: morphological cleanup
        cleaned, blob_info = morphological_cleanup(
            binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE
        )

        # STEP 6: centroid selection
        cx, cy = select_best_blob(blob_info, prev_cx, prev_cy)

        # Jump guard (pink channel only): reject centroids that move > 25 px
        # in one frame.  The pink marker can disappear when the can rotates,
        # causing the tracker to briefly lock onto a background blob.
        if mode == 'pink' and cx is not None and prev_cx is not None:
            jump = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if jump > 25:
                cx, cy = None, None   # treat as lost — will be interpolated

        # store result (convert ROI-local coords → full-frame coords)
        if cx is not None:
            x_pos[i] = cx + x1
            y_pos[i] = cy + y1
            prev_cx, prev_cy = cx, cy

        # progress reporting
        if (i + 1) % 100 == 0 or i == nframes - 1:
            print(f'    Frame {i+1:>4d}/{nframes}', end='')
            if cx is not None:
                print(f'  centroid = ({cx + x1:.1f}, {cy + y1:.1f})')
            else:
                print('  centroid = NOT FOUND')

    # -- Interpolate through any NaN gaps ---------------------------------
    # When the pink marker briefly disappears (can rotates), we linearly
    # interpolate through the gap using the surrounding valid positions.
    valid = ~np.isnan(x_pos)
    n_gaps = int(np.sum(~valid))
    if n_gaps > 0 and np.sum(valid) >= 2:
        indices = np.arange(nframes)
        x_pos = np.interp(indices, indices[valid], x_pos[valid])
        y_pos = np.interp(indices, indices[valid], y_pos[valid])
        print(f'  Interpolated {n_gaps} gap frames')

    # -- Report tracking success -------------------------------------------
    tracked = np.sum(~np.isnan(x_pos))
    print(f'  Tracking success: {tracked}/{nframes} frames '
          f'({100 * tracked / nframes:.1f}%)')

    return x_pos, y_pos


# =========================================================================
# Diagnostic Figure: Processing Pipeline for One Frame
# =========================================================================
def plot_pipeline_diagnostic(cam_id, frame_idx=0):
    """
    Generate a 6-panel diagnostic figure showing every processing step
    for a single frame from one camera.

    Panels:
        (a) Original RGB frame with ROI boundary
        (b) Cropped ROI in grayscale
        (c) Intensity histogram with threshold line
        (d) Gaussian-filtered image (frequency domain)
        (e) Binary thresholded image
        (f) Cleaned binary image with centroid marked

    Parameters
    ----------
    cam_id : int
        Camera number.
    frame_idx : int
        Frame index to visualise.
    """
    # load data
    data = sio.loadmat(DATA_PATH.format(cam_id))
    vid = data[VID_KEYS[cam_id]]
    if cam_id == 3:
        vid = rotate_cam3_video(vid)
    rgb_frame = vid[:, :, :, frame_idx]

    # unpack parameters
    y1, y2, x1, x2 = ROIS[cam_id]
    threshold = THRESHOLDS[cam_id]
    sigma = GAUSS_SIGMA[cam_id]
    mode = CHANNEL_MODE[cam_id]

    # process the frame step by step
    roi_rgb = rgb_frame[y1:y2, x1:x2, :]
    if mode == 'yellow':
        gray = rgb_to_yellow_channel(roi_rgb.astype(np.float64))
        channel_label = 'Yellow Channel\n(R+G)/2 − B'
    elif mode == 'pink':
        gray = rgb_to_pink_channel(roi_rgb.astype(np.float64))
        channel_label = 'Pink Channel\nR + B − 2G'
    else:
        gray = rgb_to_grayscale(roi_rgb.astype(np.float64))
        channel_label = 'Grayscale ROI\n(Luminosity Method)'
    filtered = gaussian_filter_freq(gray, sigma)
    binary = apply_threshold(filtered, threshold)
    cleaned, blob_info = morphological_cleanup(
        binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE
    )

    # find centroid
    cx, cy = select_best_blob(blob_info, None, None)

    # -- Create figure -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'Camera {cam_id} — Frame {frame_idx} — Image Processing Pipeline\n'
        f'(All steps from Lecture 4: Image Processing & Analysis)',
        fontsize=13, fontweight='bold'
    )

    # (a) Original frame with ROI box
    axes[0, 0].imshow(rgb_frame)
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor='lime', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('(a) Original RGB + ROI', fontsize=11)
    axes[0, 0].set_xlabel('x (pixels)')
    axes[0, 0].set_ylabel('y (pixels)')

    # (b) Single-channel ROI (grayscale or yellow channel)
    img_max = max(np.max(gray), 1)
    axes[0, 1].imshow(gray, cmap='hot' if mode in ('yellow', 'pink') else 'gray',
                       vmin=0, vmax=img_max)
    axes[0, 1].set_title(f'(b) {channel_label}', fontsize=11)
    axes[0, 1].set_xlabel('x (pixels)')
    axes[0, 1].set_ylabel('y (pixels)')

    # (c) Histogram of pixel intensities (Lecture 4, Sec. 1.1.2)
    axes[0, 2].hist(gray.ravel(), bins=256,
                     range=(0, max(img_max, 1)),
                     color='black', alpha=0.7)
    axes[0, 2].axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold = {threshold}')
    hist_title = ('Yellow Channel Histogram' if mode == 'yellow'
                   else 'Pink Channel Histogram' if mode == 'pink'
                   else 'Intensity Histogram')
    axes[0, 2].set_title(f'(c) {hist_title}\n(Sec. 1.1.2)', fontsize=11)
    axes[0, 2].set_xlabel('Pixel Intensity')
    axes[0, 2].set_ylabel('Frequency (count)')
    axes[0, 2].legend(fontsize=9)

    # (d) Gaussian-filtered image
    axes[1, 0].imshow(filtered, cmap='hot' if mode in ('yellow', 'pink') else 'gray',
                       vmin=0, vmax=img_max)
    axes[1, 0].set_title(
        f'(d) Gaussian Filtered\n(Freq. Domain, σ={sigma})', fontsize=11
    )
    axes[1, 0].set_xlabel('x (pixels)')
    axes[1, 0].set_ylabel('y (pixels)')

    # (e) Binary thresholded
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f'(e) Binary Threshold\n(T = {threshold})', fontsize=11)
    axes[1, 1].set_xlabel('x (pixels)')
    axes[1, 1].set_ylabel('y (pixels)')

    # (f) Cleaned + centroid
    axes[1, 2].imshow(cleaned, cmap='gray')
    if cx is not None:
        axes[1, 2].plot(cx, cy, 'r+', markersize=18, markeredgewidth=3,
                         label='Centroid')
        circle = plt.Circle((cx, cy), 10, color='red', fill=False, linewidth=2)
        axes[1, 2].add_patch(circle)
        axes[1, 2].legend(fontsize=9)
    axes[1, 2].set_title('(f) Cleaned + Centroid', fontsize=11)
    axes[1, 2].set_xlabel('x (pixels)')
    axes[1, 2].set_ylabel('y (pixels)')

    plt.tight_layout()
    fname = f'fig_diagnostic_cam{cam_id}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('MECH0107 CW1 — Part 1: Motion Tracking')
    print('Pipeline: Lecture 4 (Image Processing & Analysis)')
    print('=' * 60)

    # ---- Generate diagnostic plots for each camera -----------------------
    print('\n--- Diagnostic Plots (one sample frame per camera) ---')
    for cam in [1, 2, 3]:
        if os.path.exists(DATA_PATH.format(cam)):
            plot_pipeline_diagnostic(cam, frame_idx=0)
        else:
            print(f'  WARNING: {DATA_PATH.format(cam)} not found.')

    # ---- Run tracking on all three cameras -------------------------------
    results = {}
    for cam in [1, 2, 3]:
        print(f'\n{"="*60}')
        print(f'Camera {cam}')
        print(f'{"="*60}')
        filepath = DATA_PATH.format(cam)
        if not os.path.exists(filepath):
            print(f'  WARNING: {filepath} not found. Skipping.')
            continue

        x, y = track_camera(cam)
        # Mean-subtract to get displacement from equilibrium
        # Negate y because image y-axis points downward
        results[cam] = {
            'x_raw': x,
            'y_raw': y,
            'dx': x - np.nanmean(x),
            'dy': -(y - np.nanmean(y)),
        }
        nf = len(x)
        print(f'  Horizontal range: {results[cam]["dx"][~np.isnan(results[cam]["dx"])].min():.1f}'
              f' to {results[cam]["dx"][~np.isnan(results[cam]["dx"])].max():.1f} px')
        print(f'  Vertical   range: {results[cam]["dy"][~np.isnan(results[cam]["dy"])].min():.1f}'
              f' to {results[cam]["dy"][~np.isnan(results[cam]["dy"])].max():.1f} px')

    # ---- Quality check: flag suspicious jumps ----------------------------
    print(f'\n{"="*60}')
    print('Tracking Quality Check')
    print(f'{"="*60}')
    for cam in results:
        jx = np.diff(np.nan_to_num(results[cam]['dx']))
        jy = np.diff(np.nan_to_num(results[cam]['dy']))
        jumps = np.sqrt(jx**2 + jy**2)
        bad = np.where(jumps > 15)[0]
        print(f'  Camera {cam}: {len(bad)} suspicious jumps (>15 px/frame)')
        if len(bad) > 0:
            print(f'    At frames: {bad[:20]}')

    # ---- Save tracking results -------------------------------------------
    if results:
        np.savez('tracking_results.npz',
                 **{f'cam{c}_{ax}': results[c][f'd{ax}']
                    for c in results for ax in ['x', 'y']})
        print('\nSaved: tracking_results.npz')

    # =====================================================================
    # FIGURE 1: Processing Pipeline Illustration
    # =====================================================================
    if 1 in results:
        print('\nGenerating Figure 1: Processing pipeline...')
        data = sio.loadmat(DATA_PATH.format(1))
        vid = data[VID_KEYS[1]]
        fidx = min(200, vid.shape[3] - 1)
        rgb_frame = vid[:, :, :, fidx]
        y1, y2, x1, x2 = ROIS[1]
        sigma = GAUSS_SIGMA[1]
        roi_rgb = rgb_frame[y1:y2, x1:x2, :]
        mode1 = CHANNEL_MODE[1]
        if mode1 == 'yellow':
            gray = rgb_to_yellow_channel(roi_rgb.astype(np.float64))
            ch_lbl = 'Yellow Channel'
        elif mode1 == 'pink':
            gray = rgb_to_pink_channel(roi_rgb.astype(np.float64))
            ch_lbl = 'Pink Channel'
        else:
            gray = rgb_to_grayscale(roi_rgb.astype(np.float64))
            ch_lbl = 'Grayscale ROI'
        filtered = gaussian_filter_freq(gray, sigma)
        binary = apply_threshold(filtered, THRESHOLDS[1])
        cleaned, blob_info = morphological_cleanup(
            binary, MIN_BLOB_SIZE, MAX_BLOB_SIZE
        )
        cx, cy = select_best_blob(blob_info, None, None)

        cmap1 = 'hot' if mode1 in ('yellow', 'pink') else 'gray'
        vmax1 = max(np.max(gray), 1)
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
        # (a) Original with ROI
        axes[0].imshow(rgb_frame)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('(a) Original Frame\nwith ROI', fontsize=11)
        axes[0].axis('off')
        # (b) Channel ROI
        axes[1].imshow(gray, cmap=cmap1, vmin=0, vmax=vmax1)
        axes[1].set_title(f'(b) {ch_lbl}', fontsize=11)
        axes[1].axis('off')
        # (c) Gaussian filtered (frequency domain)
        axes[2].imshow(filtered, cmap=cmap1, vmin=0, vmax=vmax1)
        axes[2].set_title(f'(c) Gaussian Filtered\n(Freq. Domain, σ={sigma})',
                          fontsize=11)
        axes[2].axis('off')
        # (d) Binary threshold
        axes[3].imshow(binary, cmap='gray')
        axes[3].set_title(f'(d) Thresholded\n(T = {THRESHOLDS[1]})', fontsize=11)
        axes[3].axis('off')
        # (e) Centroid
        axes[4].imshow(roi_rgb)
        if cx is not None:
            axes[4].plot(cx, cy, 'r+', markersize=22, markeredgewidth=3)
            circle = plt.Circle((cx, cy), 12, color='red',
                                fill=False, linewidth=2)
            axes[4].add_patch(circle)
        axes[4].set_title('(e) Detected Centroid', fontsize=11)
        axes[4].axis('off')

        plt.suptitle('Figure 1: Image Processing Pipeline (Lecture 4 Methods)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('fig1_pipeline.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  Saved fig1_pipeline.png')

    # =====================================================================
    # FIGURE 2: Tracking Overlay on Sample Frames
    # =====================================================================
    print('Generating Figure 2: Tracking verification...')
    sample_frames = list(range(0, 400, 50))
    ncols = len(sample_frames)
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 3.2, 10))

    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        data = sio.loadmat(DATA_PATH.format(cam))
        vid = data[VID_KEYS[cam]]
        if cam == 3:
            vid = rotate_cam3_video(vid)
        y1, y2, x1, x2 = ROIS[cam]
        for j, fidx in enumerate(sample_frames):
            if fidx < vid.shape[3]:
                axes[i, j].imshow(vid[:, :, :, fidx])
                if not np.isnan(results[cam]['x_raw'][fidx]):
                    axes[i, j].plot(results[cam]['x_raw'][fidx],
                                    results[cam]['y_raw'][fidx],
                                    'r+', markersize=14, markeredgewidth=2)
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1.2, edgecolor='lime',
                                     facecolor='none', linestyle='--')
                axes[i, j].add_patch(rect)
                axes[i, j].set_title(f'Cam {cam}, F{fidx}', fontsize=8)
            axes[i, j].axis('off')

    plt.suptitle('Figure 2: Tracking Verification — Position (red +) and ROI (green)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig2_tracking_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig2_tracking_overlay.png')

    # =====================================================================
    # FIGURE 3: Displacement Time Series
    # =====================================================================
    print('Generating Figure 3: Displacement time series...')
    cam_labels = {
        1: 'Camera 1',
        2: 'Camera 2',
        3: 'Camera 3',
    }
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))

    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        frames = np.arange(len(results[cam]['dx']))

        # horizontal displacement
        axes[i, 0].plot(frames, results[cam]['dx'],
                        color='#2166ac', linewidth=0.9)
        axes[i, 0].set_ylabel('Displacement (px)')
        axes[i, 0].set_title(f'{cam_labels[cam]} — Horizontal')
        axes[i, 0].set_xlabel('Frame Number')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(y=0, color='k', linewidth=0.5, linestyle='--')

        # vertical displacement
        axes[i, 1].plot(frames, results[cam]['dy'],
                        color='#b2182b', linewidth=0.9)
        axes[i, 1].set_ylabel('Displacement (px)')
        axes[i, 1].set_title(f'{cam_labels[cam]} — Vertical')
        axes[i, 1].set_xlabel('Frame Number')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color='k', linewidth=0.5, linestyle='--')

    plt.suptitle('Figure 3: Tracked Displacement of the Oscillating Mass',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig3_displacement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig3_displacement.png')

    # =====================================================================
    # FIGURE 4: 2D Trajectories
    # =====================================================================
    print('Generating Figure 4: 2D trajectories...')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, cam in enumerate([1, 2, 3]):
        if cam not in results:
            continue
        dx = results[cam]['dx']
        dy = results[cam]['dy']
        valid = ~(np.isnan(dx) | np.isnan(dy))
        scatter = axes[i].scatter(dx[valid], dy[valid],
                                  c=np.arange(np.sum(valid)),
                                  cmap='viridis', s=4, alpha=0.7)
        axes[i].set_xlabel('Horizontal Displacement (px)')
        axes[i].set_ylabel('Vertical Displacement (px)')
        axes[i].set_title(cam_labels[cam])
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[i], label='Frame', shrink=0.8)

    plt.suptitle('Figure 4: 2D Trajectory (colour = time)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig4_2d_trajectories.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig4_2d_trajectories.png')

    # -- Done --------------------------------------------------------------
    print(f'\n{"="*60}')
    print('Part 1 complete. Generated figures:')
    print('  fig_diagnostic_cam[1-3].png  — Pipeline diagnostic per camera')
    print('  fig1_pipeline.png            — Processing pipeline illustration')
    print('  fig2_tracking_overlay.png    — Tracking verification on frames')
    print('  fig3_displacement.png        — Displacement time series')
    print('  fig4_2d_trajectories.png     — 2D trajectory plots')
    print('  tracking_results.npz         — Saved tracking data')
    print(f'{"="*60}')
    print('\nTUNING TIPS:')
    print('  If tracking is poor on a camera, check fig_diagnostic_cam[N].png:')
    print('    - Panel (c) histogram: move the threshold to separate the')
    print('      bright peak (mass) from the dark peak (background)')
    print('    - Panel (e) binary: should show ONLY the mass, nothing else')
    print('    - Adjust THRESHOLDS[cam] and ROIS[cam] as needed')