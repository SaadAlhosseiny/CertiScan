import cv2
import numpy as np

class FFTAnalyzer:
    def __init__(self):
        pass

    def analyze_fft(self, img_gray):
        dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def get_fft_score(self, fft_map):
        rows, cols = fft_map.shape
        crow, ccol = rows//2 , cols//2
        
        # بنعمل "ماسك" يغطي المركز (نطنش الترددات الواطية المستقرة)
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 40, 0, -1) 
        
        high_freq_area = fft_map * mask
        peaks = high_freq_area[high_freq_area > 10] 
        
        if len(peaks) > 0:
            # الحسبة دي بتمسك "التشتت" الناتج عن المسح والكتابة الرقمية
            # التزوير بيخلي الـ Variance عالي جداً مقارنة بالأصلية
            score = (np.std(peaks) / (np.mean(peaks) + 1)) * 140
        else:
            score = 0
            
        return round(float(min(max(score, 0), 100)), 2)