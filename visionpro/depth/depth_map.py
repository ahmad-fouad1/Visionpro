import cv2
import numpy as np


class DepthMapGenerator:
    """
    Converts disparity map into a visual depth map.
    """

    def to_depth_map(self, disparity):
        # closer = brighter (Jet colormap)
        depth_map = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
        return depth_map

    def inverse_depth(self, disparity):
        # optional: mathematical inverse depth
        disparity = disparity.astype(np.float32)
        depth = 1.0 / (disparity + 1e-6)
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth.astype(np.uint8)