import cv2

def keypoints(image):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(image, None)
    return keypoints