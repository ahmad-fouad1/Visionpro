import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from visionpro.auto_enhancement.analyzer import ImageAnalyzer
from visionpro.auto_enhancement.strategies import EnhancementStrategies


class AutoEnhancer:

    STRATEGY_MAP = {
        'DARK': EnhancementStrategies.enhance_dark,
        'BRIGHT': EnhancementStrategies.enhance_bright,
        'LOW_CONTRAST': EnhancementStrategies.enhance_low_contrast,
        'NORMAL': EnhancementStrategies.enhance_normal,
    }

    STRATEGY_LABELS = {
        'DARK': 'Dark Enhancement',
        'BRIGHT': 'Bright Correction',
        'LOW_CONTRAST': 'Low Contrast Boost',
        'NORMAL': 'Normal Enhancement',
    }

    CONDITION_COLORS = {
        'DARK': '#1565C0',
        'BRIGHT': '#F57F17',
        'LOW_CONTRAST': '#6A1B9A',
        'NORMAL': '#1B5E20',
    }

    @classmethod
    def enhance(cls, img):

        features, condition, description = ImageAnalyzer.analyze(img)

        enhanced = cls.STRATEGY_MAP[condition](img)

        psnr_val = psnr(img, enhanced, data_range=255)
        ssim_val = ssim(img, enhanced, data_range=255)

        return {
            "original": img,
            "enhanced": enhanced,
            "condition": condition,
            "description": description,
            "strategy": cls.STRATEGY_LABELS[condition],
            "color": cls.CONDITION_COLORS[condition],
            "features": features,
            "psnr": psnr_val,
            "ssim": ssim_val,
        }