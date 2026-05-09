import cv2
import numpy as np
def get_full_red_mask(hsv_img):
    """
    Creates a combined mask for red, handling the hue wrap-around 
    at 0 and 180 degrees.
    """
    # Range 1: 0-10 (Standard Red)
    lower1 = np.array([0, 120, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    
    # Range 2: 170-180 (Deep/Pink Red)
    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)
    
    # Combine both ranges
    full_mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological opening to clean up small noise dots
    kernel = np.ones((5, 5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    
    return full_mask