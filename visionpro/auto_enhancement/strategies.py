# ============================================================
#  CELL 5 — Enhancement Strategies
# ============================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import unsharp_mask

class EnhancementStrategies:
    """
    Four enhancement strategies:
      enhance_dark        – for under-exposed / dark images
      enhance_bright      – for over-exposed / bright images
      enhance_low_contrast – for flat, low-contrast images
      enhance_normal      – standard sharpening for normal images
    """

    @staticmethod
    def enhance_dark(img):
        """
        Strategy for DARK images.
        Pipeline: Gamma brightening → CLAHE → Unsharp Masking

        Gamma < 1 brightens dark pixels aggressively.
        CLAHE then restores local contrast.
        Unsharp mask recovers edges lost in brightening.
        """
        # Auto-tune gamma based on mean brightness
        mean_b = np.mean(img)
        gamma  = max(0.3, min(0.7, mean_b / 255.0 * 0.8))

        # Step 1: Gamma brightening
        img_f   = img.astype(np.float64) / 255.0
        gamma_out = np.power(img_f, gamma)
        gamma_u8  = (np.clip(gamma_out, 0, 1) * 255).astype(np.uint8)

        # Step 2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_out = clahe.apply(gamma_u8)

        # Step 3: Unsharp masking
        blurred   = cv2.GaussianBlur(clahe_out, (5, 5), 0)
        sharpened = cv2.addWeighted(clahe_out, 1.5, blurred, -0.5, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def enhance_bright(img):
        """
        Strategy for BRIGHT (over-exposed) images.
        Pipeline: Gamma darkening → Contrast Stretching → Bilateral Filter

        Gamma > 1 compresses over-bright pixels.
        Contrast stretching redistributes intensities.
        Bilateral filter preserves edges while removing noise.
        """
        mean_b = np.mean(img)
        gamma  = min(2.5, max(1.2, (255 - mean_b) / 255.0 * 3.0 + 1.0))

        # Step 1: Gamma darkening
        img_f     = img.astype(np.float64) / 255.0
        gamma_out = np.power(img_f, gamma)
        gamma_u8  = (np.clip(gamma_out, 0, 1) * 255).astype(np.uint8)

        # Step 2: Contrast stretching (percentile-based)
        p2, p98   = np.percentile(gamma_u8, 2), np.percentile(gamma_u8, 98)
        stretched = np.clip((gamma_u8.astype(np.float64) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)

        # Step 3: Edge-preserving bilateral filter
        result = cv2.bilateralFilter(stretched, d=9, sigmaColor=75, sigmaSpace=75)

        return result

    @staticmethod
    def enhance_low_contrast(img):
        """
        Strategy for LOW CONTRAST images.
        Pipeline: CLAHE → Laplacian Sharpening → Morphological Top-Hat

        CLAHE maximally redistributes low-contrast histogram.
        Laplacian sharpening recovers subtle edges.
        Top-hat adds fine detail highlights.
        """
        # Step 1: CLAHE with aggressive settings
        clahe     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_out = clahe.apply(img)

        # Step 2: Laplacian Sharpening
        laplacian = cv2.Laplacian(clahe_out, cv2.CV_64F)
        sharpened = clahe_out.astype(np.float64) + 0.5 * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Step 3: Top-Hat morphological transform
        se      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        top_hat = cv2.morphologyEx(sharpened, cv2.MORPH_TOPHAT, se)
        result  = cv2.add(sharpened, top_hat)

        return result

    @staticmethod
    def enhance_normal(img):
        """
        Strategy for NORMAL images.
        Pipeline: Unsharp Masking → CLAHE → Gamma (mild)

        Conservative enhancement — boost edges without altering
        the overall exposure significantly.
        """
        # Step 1: Unsharp Masking
        img_f     = img_as_float(img)
        sharpened = unsharp_mask(img_f, radius=2.0, amount=1.2)
        sharpened = (np.clip(sharpened, 0, 1) * 255).astype(np.uint8)

        # Step 2: Mild CLAHE
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_out = clahe.apply(sharpened)

        # Step 3: Mild gamma (near neutral)
        g_f    = clahe_out.astype(np.float64) / 255.0
        result = (np.power(g_f, 0.95) * 255).astype(np.uint8)

        return result


print('EnhancementStrategies defined ✓')