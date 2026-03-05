# ela_engine.py
import cv2
import numpy as np
from pathlib import Path

class ELAEngine:
    
    def __init__(self, quality=90, scale_factor=20):
        self.quality = quality
        self.scale_factor = scale_factor
        print(f"✅ ELAEngine initialized with quality={quality}%, scale={scale_factor}")
    
    def save_compressed(self, img_rgb, temp_dir="temp"):
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        temp_file = temp_path / "compressed.jpg"
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(temp_file), img_bgr, [
            cv2.IMWRITE_JPEG_QUALITY, self.quality
        ])
        
        print(f"✅ Saved compressed image at quality {self.quality}%")
        return temp_file
    
    def calculate_difference(self, original_rgb, compressed_path):
        
        compressed_bgr = cv2.imread(str(compressed_path))
        compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)
        
        if original_rgb.shape != compressed_rgb.shape:
            compressed_rgb = cv2.resize(compressed_rgb, 
                                       (original_rgb.shape[1], original_rgb.shape[0]))
        
        diff = cv2.absdiff(original_rgb, compressed_rgb)
        
        print(f"✅ Calculated difference between images")
        return diff
    
    def scale_difference(self, diff):
        scaled = diff * self.scale_factor
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        
        print(f"✅ Scaled difference by factor {self.scale_factor}")
        return scaled
    
    def calculate_ela(self, img_rgb, return_diff_only=False):
        print(f"\n🚀 Starting ELA analysis...")
        compressed_path = self.save_compressed(img_rgb)
        
        diff = self.calculate_difference(img_rgb, compressed_path)
        
        ela_result = self.scale_difference(diff)
        
        if not return_diff_only:
            compressed_bgr = cv2.imread(str(compressed_path))
            compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)
        
        compressed_path.unlink()
        
        print(f"✅ ELA analysis completed")
        
        if return_diff_only:
            return ela_result
        else:
            return ela_result, compressed_rgb
    
    def get_ela_score(self, ela_image):
        # تحويل لـ Grayscale لو الصورة ملونة
        if len(ela_image.shape) == 3:
            gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_image
        score = np.mean(gray)
        score_normalized = min(score, 100)
        return score_normalized

if __name__ == "__main__":
    print("🧪 Testing ELA Engine...")
    
    ela = ELAEngine(quality=90, scale_factor=20)
    from preprocessing import ImagePreprocessor
    pre = ImagePreprocessor()
    test_image = "test_images/original.jpeg"
    
    try:
        img_processed = pre.preprocess(test_image)
        
        ela_result, compressed = ela.calculate_ela(img_processed, return_diff_only=False)
        score = ela.get_ela_score(ela_result)
        
        print(f"\n📊 ELA Results:")
        print(f"   - ELA Score: {score:.2f}%")
        print(f"   - Result shape: {ela_result.shape}")
        
        cv2.imwrite("temp/ela_result.jpg", 
                   cv2.cvtColor(ela_result, cv2.COLOR_RGB2BGR))
        cv2.imwrite("temp/compressed.jpg", 
                   cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))
        print(f"✅ Results saved in temp folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")