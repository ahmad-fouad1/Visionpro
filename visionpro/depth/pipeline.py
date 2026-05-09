from visionpro.depth.disparity import DisparityEstimator
from visionpro.depth.depth_map import DepthMapGenerator


class DepthPipeline:
    """
    Full stereo depth estimation pipeline:
    image pair → disparity → depth map
    """

    def __init__(self):
        self.disparity_model = DisparityEstimator()
        self.depth_model = DepthMapGenerator()

    def run(self, left_img, right_img):
        disparity = self.disparity_model.compute(left_img, right_img)
        depth_map = self.depth_model.to_depth_map(disparity)

        return disparity, depth_map