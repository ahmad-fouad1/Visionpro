#  CELL 5 — Pipeline B: Laplacian + HE + Bilateral Filter
import numpy as np
import cv2
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask
def pipeline2(img, lap_weight=0.7, sigma_color=75, sigma_space=75):
    
    img_f = img.astype(np.float64)

    # Step 1: Laplacian Sharpening
    # Compute Laplacian (second derivative)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float64)
    laplacian = cv2.filter2D(img_f, -1, kernel)
    sharpened = img_f + lap_weight * laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Step 2: Histogram Equalization
    he_out = cv2.equalizeHist(sharpened)

    # Step 3: Bilateral Filter (edge-preserving smoothing)
    result = cv2.bilateralFilter(he_out, d=9,
                                  sigmaColor=sigma_color,
                                  sigmaSpace=sigma_space)
    return result
    """
    Pipeline B: Laplacian Sharpening → Histogram Equalization → Bilateral Filter

    Steps:
      1. Laplacian Sharpening – second-derivative edge enhancement
      2. Histogram Equalization – global contrast redistribution
      3. Bilateral Filter – noise removal while preserving edges

    Parameters:
      lap_weight   – blending weight for Laplacian
      sigma_color  – bilateral filter color sigma
      sigma_space  – bilateral filter spatial sigma
    """