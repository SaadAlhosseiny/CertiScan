import cv2
import os
import subprocess
import platform
from ELA import ELAEngine
from preprocessing import ImagePreprocessor

def get_verdict(score):
    if score < 10:
        return "Safe: Image appears to be authentic."
    elif score < 25:
        return "Caution: Moderate variations detected. Possibly due to compression."
    else:
        return "Warning: High variations! Strong possibility of manipulation."

def open_image(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            print(f"To view the result, check this path: {path}")
    except Exception as e:
        print(f"Could not open image automatically: {e}")

def main():
    OUTPUT_DIR = "temp"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("\n" + "—"*30)
    print("🎯 ELA Forensic Scanner")
    print("—"*30)
    
    # هنا البرنامج هيطلب منك اسم الصورة
    # مثال: test_images/usa-NF-1004-d1.jpg.jpeg
    IMAGE_PATH = input("Enter the image path: ").strip()

    try:
        preprocessor = ImagePreprocessor()
        ela_engine = ELAEngine(quality=90, scale_factor=15)

        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"❌ Error: Could not find image at [{IMAGE_PATH}]")
            print("Make sure the folder and file name are correct.")
            return

        ela_display, compressed, diff_raw = ela_engine.calculate_ela(img)
        score = ela_engine.get_ela_score(diff_raw)

        base_name = os.path.basename(IMAGE_PATH).split('.')[0]
        output_filename = f"{base_name}_ela_result.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print("\n" + "="*45)
        print(f"Analyzing: {os.path.basename(IMAGE_PATH)}")
        print(f"ELA Score: {score:.2f}%")
        print(f"Verdict: {get_verdict(score)}")
        print("="*45)

        cv2.imwrite(output_path, cv2.cvtColor(ela_display, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved as: {output_path}")

        open_image(output_path)
        
    except Exception as e:
        print(f"❌ System Error: {e}")

if __name__ == "__main__":
    main()