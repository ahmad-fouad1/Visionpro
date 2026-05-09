import cv2
import numpy as np

class HDR:
    """
    Architectural HDR Pipeline:
    - Alignment (MTB)
    - Weighted Fusion
    - Tone Mapping
    - CLAHE Enhancement
    """

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = []

    # -----------------------------
    # 1. Load Images
    # -----------------------------
    def load_images(self):
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is not None:
                self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if len(self.images) != 4:
            raise ValueError(f"Expected 4 images, got {len(self.images)}")

    # -----------------------------
    # 2. Align Images (MTB)
    # -----------------------------
    def align_images(self):
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(self.images, self.images)

    # -----------------------------
    # 3. Weight Calculation
    # -----------------------------
    def calculate_weights(self, img):
        sigma = 0.4
        dist = (img - 0.5) ** 2
        return np.exp(-dist / (2 * sigma ** 2)).prod(axis=2)

    # -----------------------------
    # 4. HDR Fusion
    # -----------------------------
    def fuse(self):
        images_f = [img.astype(np.float32) / 255.0 for img in self.images]

        h, w, c = images_f[0].shape
        hdr_num = np.zeros((h, w, c), dtype=np.float32)
        hdr_den = np.zeros((h, w), dtype=np.float32)

        for img in images_f:
            w_map = self.calculate_weights(img)

            for i in range(3):
                hdr_num[:, :, i] += img[:, :, i] * w_map

            hdr_den += w_map

        hdr_den[hdr_den == 0] = 1.0
        hdr = hdr_num / hdr_den[:, :, np.newaxis]

        return hdr

    # -----------------------------
    # 5. Tone Mapping + CLAHE
    # -----------------------------
    def tone_map(self, hdr):
        gamma = 2.2
        hdr_gamma = np.power(hdr, 1.0 / gamma)
        hdr_uint8 = (np.clip(hdr_gamma * 255, 0, 255)).astype(np.uint8)

        lab = cv2.cvtColor(hdr_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        enhanced = cv2.merge((cl, a, b))
        final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return final

    # -----------------------------
    # 6. Full Pipeline
    # -----------------------------
    def hdr(self):
        self.load_images()
        self.align_images()

        hdr = self.fuse()
        result = self.tone_map(hdr)

        return result
