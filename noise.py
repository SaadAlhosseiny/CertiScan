import cv2
import numpy as np
from pathlib import Path

class NoiseAnalyzer:
    
# ✅ __init__(block_size=8)

    def __init__(self, block_size=8):
        self.block_size = block_size
        print(f"✅ NoiseFftAnalyzer initialized with block_size={block_size}")


# ✅ analyze_noise(img_gray)

    def analyze_noise(self, img_gray):
        h, w = img_gray.shape
        bs = self.block_size
        
        noise_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block = img_gray[y:y+bs, x:x+bs]
                variance = np.var(block)
                noise_map[y:y+bs, x:x+bs] = variance
        
        # مقارنة البلوكات ببعض
        mean_variance = np.mean(noise_map[noise_map > 0])
        std_variance = np.std(noise_map[noise_map > 0])
        threshold = mean_variance + 2 * std_variance
        suspicious_map = (noise_map > threshold).astype(np.uint8) * 255
        
        noise_map = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
        noise_map = noise_map.astype(np.uint8)
        
        print(f"✅ Noise map generated")
        return noise_map, suspicious_map
    
    # ⬜ get_noise_score(noise_map)

    def get_noise_score(self, noise_map):
        score = np.mean(noise_map) / 255 * 100
        score = min(score, 100)
        print(f"✅ Noise Score: {score:.2f}%")
        return score
