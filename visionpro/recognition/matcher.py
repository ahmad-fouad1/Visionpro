import cv2
def matcher(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    return matches