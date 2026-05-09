import cv2
import numpy as np

def detect_and_match(img1, img2, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches