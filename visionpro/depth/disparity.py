import cv2
import numpy as np


class DisparityEstimator:
    """
    Computes disparity map from stereo image pair.
    """

    def __init__(self, num_disparities=16*6, block_size=15):
        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )

    def compute(self, left_img, right_img):
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray)

        # normalize for visualization
        disparity = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX
        )

        return np.uint8(disparity)