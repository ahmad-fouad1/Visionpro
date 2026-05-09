# ============================================================
#  CELL 4 — Image Analyzer
# ============================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import unsharp_mask

class ImageAnalyzer:
    """
    Analyzes image characteristics to determine enhancement needs.

    Metrics computed:
      - mean_brightness : average pixel intensity (0-255)
      - std_brightness  : pixel intensity std dev (measure of contrast)
      - entropy         : Shannon entropy (information content)
      - sharpness       : Laplacian variance (edge strength)
      - histogram_spread: difference between 95th and 5th percentile
      - dark_ratio      : fraction of pixels below threshold 60
      - bright_ratio    : fraction of pixels above threshold 200

    Classification thresholds (tunable):
      DARK_THRESH        = 80    (mean brightness)
      BRIGHT_THRESH      = 180   (mean brightness)
      LOW_CONTRAST_THRESH = 30   (std brightness)
    """

    # ── Tunable classification thresholds ──────────────────
    DARK_THRESH         = 80
    BRIGHT_THRESH       = 180
    LOW_CONTRAST_THRESH = 30
    LOW_SHARPNESS_THRESH = 100

    @staticmethod
    def extract_features(img):
        """Compute all diagnostic metrics from a grayscale image."""
        f = img.astype(np.float64)
        mean_b  = float(np.mean(f))
        std_b   = float(np.std(f))

        # Shannon entropy
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0,256), density=True)
        hist    = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        # Sharpness (Laplacian variance)
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())

        # Histogram spread
        p5, p95   = np.percentile(img, 5), np.percentile(img, 95)
        hist_spread = float(p95 - p5)

        # Pixel ratio extremes
        dark_ratio   = float(np.mean(img < 60))
        bright_ratio = float(np.mean(img > 200))

        return {
            'mean_brightness': mean_b,
            'std_brightness' : std_b,
            'entropy'        : entropy,
            'sharpness'      : sharpness,
            'histogram_spread': hist_spread,
            'dark_ratio'     : dark_ratio,
            'bright_ratio'   : bright_ratio,
        }

    @classmethod
    def classify(cls, features):
        """
        Rule-based classifier that maps features → image condition.
        Returns: (condition_str, description_str)
        """
        mean_b = features['mean_brightness']
        std_b  = features['std_brightness']

        if mean_b < cls.DARK_THRESH:
            return 'DARK', f'Dark image (mean={mean_b:.1f} < {cls.DARK_THRESH})'
        elif mean_b > cls.BRIGHT_THRESH:
            return 'BRIGHT', f'Over-bright image (mean={mean_b:.1f} > {cls.BRIGHT_THRESH})'
        elif std_b < cls.LOW_CONTRAST_THRESH:
            return 'LOW_CONTRAST', f'Low contrast (std={std_b:.1f} < {cls.LOW_CONTRAST_THRESH})'
        else:
            return 'NORMAL', f'Normal image (mean={mean_b:.1f}, std={std_b:.1f}) — applying sharpening'

    @classmethod
    def analyze(cls, img):
        """Full analysis: extract features + classify."""
        features  = cls.extract_features(img)
        condition, description = cls.classify(features)
        return features, condition, description


