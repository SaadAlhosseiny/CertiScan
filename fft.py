import cv2
import numpy as np

class FFTAnalyzer:
    
    def __init__(self):
        print(f"✅ FFTAnalyzer initialized")
    
    def analyze_fft(self, img_gray):
        fft = np.fft.fft2(img_gray)
        fft_shift = np.fft.fftshift(fft)
        
        magnitude = np.abs(fft_shift)
        magnitude_log = np.log1p(magnitude)
        
        fft_map = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
        fft_map = fft_map.astype(np.uint8)
        
        print(f"✅ FFT map generated")
        return fft_map
    
    def get_fft_score(self, fft_map):
        score = np.mean(fft_map) / 255 * 100
        score = min(score, 100)
        print(f"✅ FFT Score: {score:.2f}%")
        return score