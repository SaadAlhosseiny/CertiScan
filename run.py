import cv2
import numpy as np
import sys
from pathlib import Path
from preprocessing import ImagePreprocessor
from ELA import ELAEngine
from noise import NoiseAnalyzer
from fft import FFTAnalyzer
from masking import create_adaptive_mask, morphological_refine, apply_red_overlay, combine_maps

def print_header():
    print("\033[H\033[J", end="") 
    print("="*60)
    print("      🔍 CERTISCAN PRO - ADVANCED FORENSIC TERMINAL 🔍      ")
    print("="*60)

def get_verdict(total_score):
    if total_score < 12:
        return "✅ AUTHENTIC (Safe)", "\033[92m"
    elif total_score < 30:
        return "⚠️ SUSPICIOUS (Review Needed)", "\033[93m"
    else:
        return "❌ FORGED (Tampered)", "\033[91m"

def process_analysis(image_path_str):
    img_path = Path(image_path_str.strip().replace("'", "").replace('"', ""))
    
    if not img_path.exists():
        print(f"❌ Error: Path '{img_path}' does not exist!")
        return

    # 1. التجهيز
    pre = ImagePreprocessor(target_size=(512, 512))
    img_orig = pre.read_image(str(img_path))
    img_clean = pre.preprocess(str(img_path))
    img_gray = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)

    # 2. المحركات
    ela_eng = ELAEngine()
    noise_eng = NoiseAnalyzer()
    fft_eng = FFTAnalyzer()

    _, _, diff_raw = ela_eng.calculate_ela(img_clean)
    ela_s = ela_eng.get_ela_score(diff_raw)

    n_map, _ = noise_eng.analyze_noise(img_gray)
    noise_s = noise_eng.get_noise_score(n_map)

    fft_map = fft_eng.analyze_fft(img_gray)
    fft_s = fft_eng.get_fft_score(fft_map)

    # 3. الـ Masking
    combined = combine_maps(diff_raw, fft_map, n_map, weights=[0.3, 0.5, 0.2])
    mask_raw = create_adaptive_mask(combined, method="triangle")
    mask_refined = morphological_refine(mask_raw, erode_ksize=5, erode_iterations=2)
    mask_s = min(float(np.sum(mask_refined > 0) / mask_refined.size * 100) * 5, 100)

    # 4. النتيجة النهائية
    final = (0.4 * ela_s) + (0.3 * fft_s) + (0.2 * noise_s) + (0.1 * mask_s)
    
    print(f"\n📄 File: {img_path.name}")
    print("-" * 30)
    print(f"🔹 ELA Score:   {ela_s:>6.2f}%")
    print(f"🔹 FFT Score:   {fft_s:>6.2f}%")
    print(f"🔹 Noise Score: {noise_s:>6.2f}%")
    print(f"🔹 Mask Score:  {mask_s:>6.2f}%")
    print("-" * 30)
    
    verdict, color_code = get_verdict(final)
    reset_color = "\033[0m"
    print(f"🏆 FINAL SCORE: {final:.2f}%")
    print(f"📢 VERDICT: {color_code}{verdict}{reset_color}")
    print("-" * 60)

    # 5. حفظ المخرجات
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)
    
    out_name = f"result_{img_path.stem}.jpg"
    out_path = output_dir / out_name
    
    # التعديل هنا: نرسل mask_refined مباشرة بدون عمل resize خارجي
    overlay = apply_red_overlay(img_orig, mask_refined)
    
    cv2.imwrite(str(out_path), overlay)
    print(f"💾 Result saved to: {out_path}\n")

if __name__ == "__main__":
    print_header()
    if len(sys.argv) > 1:
        path_input = sys.argv[1]
    else:
        path_input = input("👉 Drag & Drop image here: ")
    process_analysis(path_input)