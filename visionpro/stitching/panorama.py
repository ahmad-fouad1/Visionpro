import cv2
import numpy as np

from .feature_match import detect_and_match
from .homography import compute_homography
from .blend import simple_blend


def stitch_images(img1, img2):

    kp1, kp2, matches = detect_and_match(img1, img2)

    if matches is None:
        raise ValueError("Not enough features detected")

    H = compute_homography(kp1, kp2, matches)

    if H is None:
        raise ValueError("Homography could not be computed")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    result = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))

    result = simple_blend(img1, result)

    return result