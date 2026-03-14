import cv2
import numpy as np


class CopyMoveDetector:

    def detect(self, img_gray):

        orb = cv2.ORB_create(2000)

        kp, des = orb.detectAndCompute(img_gray, None)

        if des is None or len(kp) < 10:
            return np.zeros_like(img_gray), 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des, des)

        suspicious = np.zeros_like(img_gray)

        count = 0

        for m in matches:

            if m.distance < 20 and m.queryIdx != m.trainIdx:

                pt = kp[m.queryIdx].pt

                x, y = int(pt[0]), int(pt[1])

                suspicious[y-3:y+3, x-3:x+3] = 255

                count += 1

        score = min((count / 200) * 100, 100)

        return suspicious, score