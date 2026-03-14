import cv2
import numpy as np


class JPEGGhostDetector:

    def recompress(self, img, quality):

        encode = [cv2.IMWRITE_JPEG_QUALITY, quality]

        _, enc = cv2.imencode(".jpg", img, encode)

        dec = cv2.imdecode(enc, 1)

        return dec

    def analyze(self, img):

        qualities = [60, 70, 80, 90]

        maps = []

        for q in qualities:

            recompressed = self.recompress(img, q)

            diff = cv2.absdiff(img, recompressed)

            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            maps.append(gray.astype(float))

        ghost = np.mean(maps, axis=0)

        ghost = cv2.normalize(ghost, None, 0, 255, cv2.NORM_MINMAX)

        return ghost.astype("uint8")

    def score(self, ghost_map):

        score = np.percentile(ghost_map, 95) / 255 * 100

        return min(score, 100)