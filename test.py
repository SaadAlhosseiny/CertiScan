
import sys
from pathlib import Path
import cv2
import numpy as np

# نضيف المجلد src للمسار عشان نقدر نستورد الملفات
sys.path.append('src')

from preprocessing import ImagePreprocessor
from ela_engine import ELAEngine

print("🧡 Testing Preprocessing + ELA together\n")

# 1. تجهيز المسارات
test_image = "test_image/original.jpeg"  # غير المسار لصورة عندك
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

try:
    # 2. Preprocessing
    print("="*50)
    print("📥 Phase 1: Preprocessing")
    print("="*50)
    
    pre = ImagePreprocessor()
    img_processed = pre.preprocess(test_image)
    
    # 3. ELA
    print("\n" + "="*50)
    print("🔍 Phase 2: ELA Engine")
    print("="*50)
    
    ela = ELAEngine(quality=90, scale_factor=20)
    ela_result, compressed = ela.calculate_ela(img_processed, return_diff_only=False)
    ela_score = ela.get_ela_score(ela_result)
    
    # 4. عرض النتائج
    print("\n" + "="*50)
    print("📊 Final Results")
    print("="*50)
    print(f"ELA Score: {ela_score:.2f}%")
    
    # 5. حفظ كل الصور
    cv2.imwrite("temp/01_original.jpg", 
                cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    cv2.imwrite("temp/02_compressed.jpg", 
                cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))
    cv2.imwrite("temp/03_ela_result.jpg", 
                cv2.cvtColor(ela_result, cv2.COLOR_RGB2BGR))
    
    print(f"\n✅ All results saved in 'temp' folder:")
    print(f"   - 01_original.jpg   (after preprocessing)")
    print(f"   - 02_compressed.jpg  (after compression)")
    print(f"   - 03_ela_result.jpg  (ELA result)")
    
except Exception as e:
    print(f"❌ Error: {e}")