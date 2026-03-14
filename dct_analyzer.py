import cv2
import numpy as np


class DCTAnalyzer:

    def analyze(self, img_gray):

        img = np.float32(img_gray) / 255.0

        dct = cv2.dct(img)

        dct_abs = np.abs(dct)

        dct_map = cv2.log(dct_abs + 1)

        dct_map = cv2.normalize(
            dct_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return dct_map.astype("uint8")

    def score(self, dct_map):

        score = np.percentile(dct_map, 95) / 255 * 100

        return min(score, 100)