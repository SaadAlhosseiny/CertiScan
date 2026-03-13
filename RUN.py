import cv2
import numpy as np
import subprocess
import platform
from pathlib import Path
from ELA import ELAEngine
from preprocessing import ImagePreprocessor
from noise import NoiseAnalyzer
from fft import FFTAnalyzer
from masking import (
    create_adaptive_mask,
    morphological_refine,
    apply_red_overlay,
)

# ─────────────────────────────────────────────
#  Verdict logic
# ─────────────────────────────────────────────
def get_verdict(ela_score: float, noise_score: float, fft_score: float, mask_score: float = 0) -> str:
    final_score = (0.40 * ela_score) + (0.30 * fft_score) + (0.20 * noise_score) + (0.10 * mask_score)
    tamper_pct = min(round(final_score, 1), 100)
    if tamper_pct < 10:
        return f"✅ Authentic ({tamper_pct}%)"
    elif tamper_pct < 30:
        return f"⚠️ Suspicious ({tamper_pct}%)"
    else:
        return f"❌ Forged ({tamper_pct}%)"

# ─────────────────────────────────────────────
#  OS helper
# ─────────────────────────────────────────────
def open_image(path: str) -> None:
    try:
        if platform.system() == "Windows":
            import os
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass

# ─────────────────────────────────────────────
#  Combine maps helper
# ─────────────────────────────────────────────
def combine_maps(diff_raw, fft_map, noise_map, weights=[0.3, 0.5, 0.2]):
    target_size = (512, 512)

    diff_resized  = cv2.resize(diff_raw,  target_size)
    fft_resized   = cv2.resize(fft_map,   target_size)
    noise_resized = cv2.resize(noise_map, target_size)

    if len(fft_resized.shape) == 2:
        fft_resized = cv2.cvtColor(fft_resized, cv2.COLOR_GRAY2BGR)
    if len(noise_resized.shape) == 2:
        noise_resized = cv2.cvtColor(noise_resized, cv2.COLOR_GRAY2BGR)

    diff_norm  = diff_resized.astype("float32")  / 255.0
    fft_norm   = fft_resized.astype("float32")   / 255.0
    noise_norm = noise_resized.astype("float32") / 255.0

    combined = (
        weights[0] * diff_norm +
        weights[1] * fft_norm  +
        weights[2] * noise_norm
    )

    return (combined * 255).astype("uint8")

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR = Path("temp")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 35)
    print("🚀 CertiScan Pro - Forensic Suite")
    print("=" * 35)

    IMAGE_PATH = Path(input("Enter image path: ").strip())
    if not IMAGE_PATH.exists():
        print("❌ Error: File not found!")
        return

    try:
        preprocessor = ImagePreprocessor()
        ela_engine   = ELAEngine(quality=90, scale_factor=15)

        img           = cv2.imread(str(IMAGE_PATH))
        img_processed = preprocessor.preprocess(str(IMAGE_PATH))
        img_gray      = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)

        # 1. ELA
        ela_display, _, diff_raw = ela_engine.calculate_ela(img)
        ela_gray  = cv2.cvtColor(diff_raw, cv2.COLOR_RGB2GRAY) if len(diff_raw.shape) == 3 else diff_raw
        ela_score = min(float(np.percentile(ela_gray, 95)) / 255 * 100, 100)

        # 2. Noise
        noise_analyzer            = NoiseAnalyzer()
        noise_map, suspicious_map = noise_analyzer.analyze_noise(img_gray)
        noise_score               = min(float(np.percentile(noise_map, 95)) / 255 * 100, 100)

        # 3. FFT
        fft_analyzer = FFTAnalyzer()
        fft_map      = fft_analyzer.analyze_fft(img_gray)
        fft_score    = fft_analyzer.get_fft_score(fft_map)

        # 4. Masking pipeline
        combined_map  = combine_maps(diff_raw, fft_map, noise_map, weights=[0.30, 0.50, 0.20])
        gray_combined = cv2.cvtColor(combined_map, cv2.COLOR_BGR2GRAY)
        raw_mask      = create_adaptive_mask(gray_combined, method="otsu")
        refined_mask  = morphological_refine(
            raw_mask,
            close_ksize=7,
            erode_ksize=3,
            close_iterations=2,
            erode_iterations=1,
        )
        mask_score = min(float(np.sum(refined_mask > 0) / refined_mask.size * 100) * 3, 100)

        # 5. Report
        final_score = (0.40 * ela_score) + (0.30 * fft_score) + (0.20 * noise_score) + (0.10 * mask_score)
        tamper_pct  = min(round(final_score, 1), 100)

        print("\n📊 ANALYSIS REPORT")
        print("-" * 25)
        print(f"File:        {IMAGE_PATH.name}")
        print(f"ELA Score:   {round(ela_score, 2)}%")
        print(f"Noise Score: {round(noise_score, 2)}%")
        print(f"FFT Score:   {round(fft_score, 2)}%")
        print(f"Mask Score:  {round(mask_score, 2)}%")
        print(f"Final Score: {tamper_pct}%")
        print("-" * 25)
        verdict = get_verdict(ela_score, noise_score, fft_score, mask_score)
        print(f"Verdict: {verdict}")
        print("-" * 25)

        # 6. Save outputs
        stem = IMAGE_PATH.stem

        mask_resized = cv2.resize(refined_mask, (img.shape[1], img.shape[0]))
        overlay_bgr  = apply_red_overlay(img, mask_resized, alpha=0.45)

        ela_path     = OUTPUT_DIR / f"{stem}_ela.jpg"
        fft_path     = OUTPUT_DIR / "fft_map.jpg"
        mask_path    = OUTPUT_DIR / f"{stem}_mask_raw.jpg"
        refined_path = OUTPUT_DIR / f"{stem}_mask_refined.jpg"
        overlay_path = OUTPUT_DIR / f"{stem}_overlay.jpg"

        cv2.imwrite(str(ela_path),     cv2.cvtColor(ela_display, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(fft_path),     fft_map)
        cv2.imwrite(str(mask_path),    raw_mask)
        cv2.imwrite(str(refined_path), mask_resized)
        cv2.imwrite(str(overlay_path), overlay_bgr)

        print(f"\n✅ Maps saved in 'temp/':")
        print(f"   • {ela_path.name}        - ELA heat-map")
        print(f"   • {fft_path.name}        - FFT artifact map")
        print(f"   • {mask_path.name}       - Raw adaptive mask")
        print(f"   • {refined_path.name}    - Morphologically refined mask")
        print(f"   • {overlay_path.name}    - Red overlay (main result)")

        open_image(str(overlay_path))

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()