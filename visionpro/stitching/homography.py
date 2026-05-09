import cv2
import numpy as np

def compute_homography(kp1, kp2, matches, max_matches=50):
    if len(matches) < 4:
        return None

    matches = matches[:max_matches]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    return H