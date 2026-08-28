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


def apply_spectral_colormap(gray_image, colormap_name="SUIT_UV_279"):
    """
    Renders false-color multi-spectral representations commonly used in solar physics:
      - SUIT_UV_279: High-contrast cyan/ultraviolet (Aditya-L1 SUIT Mg II k filter)
      - AIA_171_GOLD: SDO/AIA 171 Å Quiet Corona & Loop Golden palette
      - AIA_193_BRONZE: SDO/AIA 193 Å Active Region Bronze/Rust palette
      - MAGNETOGRAM: SOHO/MDI line-of-sight dipole contrast (Blue/Red magnetic polarity)
      - PLASMA_INFERNO: High-energy thermal gradient colormap
    """
    # Ensure uint8 in [0, 255]
    img_uint8 = np.clip(gray_image * 255.0, 0, 255).astype(np.uint8)

    if colormap_name == "SUIT_UV_279":
        # Custom Aditya-L1 UV LUT: deep purple-blue to cyan-white
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip((i - 100) * 1.6, 0, 255))
            g = int(np.clip(i * 1.1, 0, 255))
            b = int(np.clip(120 + i * 0.6, 0, 255))
            lut[i, 0] = [b, g, r]  # BGR order for OpenCV
        colored = cv2.LUT(cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR), lut)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    elif colormap_name == "AIA_171_GOLD":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_AUTUMN)

    elif colormap_name == "AIA_193_BRONZE":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_COPPER)

    elif colormap_name == "MAGNETOGRAM":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_TWILIGHT_SHIFTED)

    elif colormap_name == "PLASMA_INFERNO":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)

    else:
        # Default grayscale to RGB
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)


def compute_magnetic_flux_gradient(gray_image):
    """
    Computes Sobel spatial flux gradients (nabla I) representing magnetic field shear lines.
    Returns: gradient magnitude map and active neutral line contours.
    """
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Normalize to [0, 1]
    denom = grad_mag.max() - grad_mag.min() + 1e-8
    grad_norm = (grad_mag - grad_mag.min()) / denom

    # Extract contours of strong flux gradient (> 60th percentile)
    threshold_val = np.percentile(grad_norm, 75)
    binary_mask = (grad_norm > threshold_val).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return grad_norm, contours


def compute_solar_physical_metrics(patch):
    """
    Computes quantitative physics indicators from the active region patch:
      - Total Unsigned Magnetic Flux proxy (Phi)
      - Peak Flux Gradient (Shear Intensity)
      - Magnetic Neutral Line Complexity Index
    """
    phi_proxy = float(np.sum(patch) / 1000.0)
    grad_norm, contours = compute_magnetic_flux_gradient(patch)
    max_gradient = float(np.max(grad_norm))
    shear_complexity = float(len(contours) * 1.5 + (np.mean(grad_norm) * 100.0))

    return {
        "unsigned_flux_proxy": phi_proxy,
        "max_flux_gradient": max_gradient,
        "shear_complexity_index": min(100.0, shear_complexity),
        "total_contour_loops": len(contours)
    }