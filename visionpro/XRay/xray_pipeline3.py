# ============================================================
#  CELL 6 — Pipeline C: High-Boost + Adaptive Gamma + Morphology
# ============================================================
import numpy as np
import cv2
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask

def adaptive_gamma(img, gamma_dark=0.6, gamma_bright=1.2):
    img_f = img.astype(np.float64) / 255.0
    # Local mean using Gaussian blur
    local_mean = cv2.GaussianBlur(img_f, (31, 31), 0)
    # Interpolate gamma based on local brightness
    gamma_map = gamma_dark + (gamma_bright - gamma_dark) * local_mean
    # Apply pixel-wise adaptive gamma
    result = np.power(img_f, gamma_map)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def pipeline3(img, boost_factor=2.5):
    img_f = img.astype(np.float64)

    # Step 1: High-Boost Filtering
    # Blurred version (low-pass)
    blurred = cv2.GaussianBlur(img_f, (9, 9), 0)
    # High-boost = A * original - blur  (A > 1 for high-boost)
    high_boost = boost_factor * img_f - blurred
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)

    # Step 2: Adaptive Gamma
    gamma_out = adaptive_gamma(high_boost)

    # Step 3: Morphological White Top-Hat
    # Highlights bright structures smaller than the structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat = cv2.morphologyEx(gamma_out, cv2.MORPH_TOPHAT, se)
    # Add top-hat details back to the image
    result = cv2.add(gamma_out, top_hat)
    return result
    """
    Pipeline C: High-Boost Filtering → Adaptive Gamma → Morphological Top-Hat

    Steps:
      1. High-Boost Filtering   – amplified unsharp mask variant
      2. Adaptive Gamma         – spatially varying brightness correction
      3. Morphological Top-Hat  – reveals fine details & textures

    Parameters:
      boost_factor – amplification factor (A in high-boost = A*original - blur)
    """