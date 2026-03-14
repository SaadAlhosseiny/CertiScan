import cv2
import numpy as np

class NoiseAnalyzer:
    def __init__(self, block_size=8):
        self.block_size = block_size
        print(f"✅ NoiseAnalyzer initialized with block_size={block_size}")

    def analyze_noise(self, img_gray):
        h, w = img_gray.shape
        bs = self.block_size
        noise_map = np.zeros((h,w),dtype=np.float32)
        for y in range(0,h-bs,bs):
            for x in range(0,w-bs,bs):
                block = img_gray[y:y+bs, x:x+bs]
                variance = np.var(block)
                noise_map[y:y+bs, x:x+bs] = variance
        mean_var = np.mean(noise_map[noise_map>0])
        std_var = np.std(noise_map[noise_map>0])
        threshold = mean_var + 2*std_var
        suspicious_map = (noise_map > threshold).astype(np.uint8)*255
        noise_map = cv2.normalize(noise_map,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        return noise_map, suspicious_map

    def get_noise_score(self, noise_map):
        score = min(float(np.percentile(noise_map,95))/255*100,100)
        return round(score,2)