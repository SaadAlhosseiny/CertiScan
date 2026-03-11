import cv2
import subprocess
import platform
from pathlib import Path
from ELA import ELAEngine
from preprocessing import ImagePreprocessor
from noise import NoiseAnalyzer
from fft import FFTAnalyzer

def get_verdict(ela_score, noise_score, fft_score):
    # الـ FFT واخد 50% من الوزن لأنه الأدق في كشف الفوتوشوب
    final_score = (0.3 * ela_score) + (0.2 * noise_score) + (0.5 * fft_score)
    
    # لو الـ FFT عدي الـ 30 في المعادلة الجديدة، ده تزوير رقمي واضح
    if fft_score > 30:
        return f"❌ Forged: Digital Manipulation Detected (FFT High: {fft_score:.2f}%)"
    elif final_score > 22:
        return f"⚠️ Suspicious: Potential Alteration (Final: {final_score:.2f}%)"
    elif final_score < 12:
        return f"✅ Excellent: Image is authentic"
    else:
        return f"🟡 Good: Image appears authentic"

def open_image(path):
    try:
        if platform.system() == "Windows":
            import os
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
    except: pass

def main():
    OUTPUT_DIR = Path("temp")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*35)
    print("🚀 CertiScan Pro - Forensic Suite")
    print("="*35)
    
    IMAGE_PATH_STR = input("Enter image path: ").strip()
    IMAGE_PATH = Path(IMAGE_PATH_STR)

    if not IMAGE_PATH.exists():
        print(f"❌ Error: File not found!")
        return

    try:
        preprocessor = ImagePreprocessor()
        ela_engine = ELAEngine(quality=90, scale_factor=15)
        
        img = cv2.imread(str(IMAGE_PATH))
        img_processed = preprocessor.preprocess(str(IMAGE_PATH))
        img_gray = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)

        # 1. تحليل ELA و Noise
        ela_display, _, diff_raw = ela_engine.calculate_ela(img)
        ela_score = ela_engine.get_ela_score(diff_raw)
        
        noise_analyzer = NoiseAnalyzer()
        noise_map, suspicious_map = noise_analyzer.analyze_noise(img_gray)
        noise_score = noise_analyzer.get_noise_score(noise_map)

        # 2. تحليل FFT (المطور)
        fft_analyzer = FFTAnalyzer()
        fft_map = fft_analyzer.analyze_fft(img_gray)
        fft_score = fft_analyzer.get_fft_score(fft_map)

        # 3. عرض التقرير
        print("\n" + "📊 ANALYSIS REPORT")
        print("-" * 25)
        print(f"File: {IMAGE_PATH.name}")
        print(f"FFT Score:   {fft_score:.2f}%")
        print(f"ELA Score:   {ela_score:.2f}%")
        print(f"Noise Score: {noise_score:.2f}%")
        print("-" * 25)
        print(f"Verdict: {get_verdict(ela_score, noise_score, fft_score)}")
        print("-" * 25)

        # 4. حفظ النتائج
        output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_result.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(ela_display, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUTPUT_DIR / "fft_map.jpg"), fft_map)
        
        print(f"✅ Maps saved in 'temp/'.")
        open_image(str(output_path))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()