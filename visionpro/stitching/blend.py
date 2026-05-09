import numpy as np
import cv2

def simple_blend(img1, img2_warped):
    h, w = img1.shape[:2]

    result = img2_warped.copy()
    result[0:h, 0:w] = img1

    return result