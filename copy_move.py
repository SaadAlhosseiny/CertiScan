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
            # رفعنا الـ threshold من 20 لـ 40 عشان يشتغل على صور مزورة فعلاً
            if m.distance < 40 and m.queryIdx != m.trainIdx:
                pt1 = kp[m.queryIdx].pt
                pt2 = kp[m.trainIdx].pt
                # نتاكد ان المسافة بين النقطتين كبيرة (مش نفس المنطقة)
                dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                if dist > 20:
                    x, y = int(pt1[0]), int(pt1[1])
                    suspicious[max(0,y-3):y+3, max(0,x-3):x+3] = 255
                    count += 1

        score = min((count / 100) * 100, 100)
        return suspicious, score