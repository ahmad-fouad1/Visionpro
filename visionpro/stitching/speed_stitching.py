import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)

    result = cv2.warpPerspective(img2, H, 
               (img1.shape[1] + img2.shape[1], img1.shape[0]))

    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    return result