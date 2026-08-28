import cv2
import numpy as np
from astropy.io import fits


def load_and_clean_fits(fits_path):
    """Loads a SUIT FITS file and removes sky noise/NaN values."""
    with fits.open(fits_path) as hdul:
        # Get raw pixel array
        data = hdul[0].data.astype(np.float32) if hdul[0].data is not None else hdul[1].data.astype(np.float32)

    data = np.nan_to_num(data, nan=0.0)
    data[data < 0] = 0.0
    return data


def preprocess_solar_disk(image, target_size=(512, 512)):
    """Applies logarithmic scaling and min-max normalization."""
    # Log transform to compress dynamic range of bright flare loops
    log_img = np.log1p(image)

    # Min-max normalization
    denom = log_img.max() - log_img.min() + 1e-8
    norm_img = (log_img - log_img.min()) / denom

    # Resize to target resolution
    return cv2.resize(norm_img, target_size, interpolation=cv2.INTER_AREA)


def extract_active_region(image, bbox=None, patch_size=(256, 256)):
    """
    Crops the brightest Active Region dynamically if no bounding box is given.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
    else:
        # Fallback: Find region with highest intensity
        h, w = image.shape
        cy, cx = np.unravel_index(np.argmax(image), image.shape)

        # Center box around peak intensity point
        half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
        y1, y2 = max(0, cy - half_h), min(h, cy + half_h)
        x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
        crop = image[y1:y2, x1:x2]

    return cv2.resize(crop, patch_size)