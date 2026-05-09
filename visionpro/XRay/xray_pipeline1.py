#  CELL 4 — Pipeline A: Unsharp Masking + CLAHE + Gamma
import numpy as np
import cv2
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask

def pipeline1(img, radius=2.0, amount=1.5, clip_limit=2.0, gamma=0.85):
    # Normalize image to float [0,1]
    img_f = img_as_float(img)

    # 1) Sharpen (Unsharp Mask)
    sharpened = unsharp_mask(img_f, radius=radius, amount=amount)
    sharpened = np.clip(sharpened, 0, 1)

    # 2) CLAHE (contrast enhancement)
    sharpened_u8 = img_as_ubyte(sharpened)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_out = clahe.apply(sharpened_u8)

    # 3) Gamma correction
    clahe_f = clahe_out / 255.0
    gamma_out = np.power(clahe_f, gamma)

    return (np.clip(gamma_out, 0, 1) * 255).astype(np.uint8)