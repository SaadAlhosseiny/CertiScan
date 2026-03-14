import cv2
import numpy as np


class AIDetector:

    def analyze_frequency(self, img_gray):

        f = np.fft.fft2(img_gray)

        fshift = np.fft.fftshift(f)

        magnitude = np.log(np.abs(fshift) + 1)

        mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return mag.astype("uint8")

    def grid_artifacts(self, fft_map):

        horizontal = np.mean(fft_map, axis=1)

        vertical = np.mean(fft_map, axis=0)

        score = (np.std(horizontal) + np.std(vertical)) / 2

        score = (score / np.max(fft_map)) * 100

        return min(score, 100)

    def noise_pattern(self, img_gray):

        blur = cv2.medianBlur(img_gray, 5)

        noise = img_gray.astype(float) - blur.astype(float)

        score = np.std(noise)

        score = (score / 50) * 100

        return min(score, 100)

    def score(self, img_gray):

        fft_map = self.analyze_frequency(img_gray)

        freq = self.grid_artifacts(fft_map)

        noise = self.noise_pattern(img_gray)

        return 0.6 * freq + 0.4 * noise