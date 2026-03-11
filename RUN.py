import cv2
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
def get_verdict(ela_score: float, noise_score: float, fft_score: float) -> str:
    final_score = (0.3 * ela_score) + (0.2 * noise_score) + (0.5 * fft_score)

    if fft_score > 30:
        return f"❌ Forged: Digital Manipulation Detected (FFT High: {fft_score:.2f}%)"
    elif final_score > 22:
        return f"⚠️  Suspicious: Potential Alteration (Final: {final_score:.2f}%)"
    elif final_score < 12:
        return f"✅ Excellent: Image is authentic"
    else:
        return f"🟡 Good: Image appears authentic"

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

    # Resize الخرائط
    diff_resized  = cv2.resize(diff_raw, target_size)
    fft_resized   = cv2.resize(fft_map, target_size)
    noise_resized = cv2.resize(noise_map, target_size)

    # لو grayscale → نحولها لـ 3 قنوات
    if len(fft_resized.shape) == 2:
        fft_resized = cv2.cvtColor(fft_resized, cv2.COLOR_GRAY2BGR)
    if len(noise_resized.shape) == 2:
        noise_resized = cv2.cvtColor(noise_resized, cv2.COLOR_GRAY2BGR)

    # Normalize
    diff_norm  = diff_resized.astype("float32") / 255.0
    fft_norm   = fft_resized.astype("float32") / 255.0
    noise_norm = noise_resized.astype("float32") / 255.0

    # Weighted combination
    combined = (
        weights[0] * diff_norm +
        weights[1] * fft_norm +
        weights[2] * noise_norm
    )

    # رجّع الصورة لـ 0–255
    combined = (combined * 255).astype("uint8")
    return combined

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
        # ── Initialise engines ──────────────────────────────────────────────
        preprocessor = ImagePreprocessor()
        ela_engine   = ELAEngine(quality=90, scale_factor=15)

        img           = cv2.imread(str(IMAGE_PATH))
        img_processed = preprocessor.preprocess(str(IMAGE_PATH))
        img_gray      = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)

        # ── 1. ELA ──────────────────────────────────────────────────────────
        ela_display, _, diff_raw = ela_engine.calculate_ela(img)
        ela_score                = ela_engine.get_ela_score(diff_raw)

        # ── 2. Noise ────────────────────────────────────────────────────────
        noise_analyzer             = NoiseAnalyzer()
        noise_map, suspicious_map  = noise_analyzer.analyze_noise(img_gray)
        noise_score                = noise_analyzer.get_noise_score(noise_map)

        # ── 3. FFT ──────────────────────────────────────────────────────────
        fft_analyzer = FFTAnalyzer()
        fft_map      = fft_analyzer.analyze_fft(img_gray)
        fft_score    = fft_analyzer.get_fft_score(fft_map)

        # ── 4. Report ───────────────────────────────────────────────────────
        print("\n📊 ANALYSIS REPORT")
        print("-" * 25)
        print(f"File:        {IMAGE_PATH.name}")
        print(f"FFT Score:   {fft_score:.2f}%")
        print(f"ELA Score:   {ela_score:.2f}%")
        print(f"Noise Score: {noise_score:.2f}%")
        print("-" * 25)
        verdict = get_verdict(ela_score, noise_score, fft_score)
        print(f"Verdict: {verdict}")
        print("-" * 25)

        # ── 5. Advanced masking pipeline ────────────────────────────────────
        combined_map = combine_maps(
            diff_raw, fft_map, noise_map,
            weights=[0.30, 0.50, 0.20],
        )

        # حول الـ combined_map لصورة رمادية قبل الـ Otsu
        gray_combined = cv2.cvtColor(combined_map, cv2.COLOR_BGR2GRAY)
        raw_mask = create_adaptive_mask(gray_combined, method="otsu")

        refined_mask = morphological_refine(
            raw_mask,
            close_ksize=7,
            erode_ksize=3,
            close_iterations=2,
            erode_iterations=1,
        )

        # Resize الماسك لحجم الصورة الأصلية قبل overlay
        mask_resized = cv2.resize(refined_mask, (img.shape[1], img.shape[0]))
        overlay_bgr = apply_red_overlay(img, mask_resized, alpha=0.45)

        # ── 6. Save outputs ─────────────────────────────────────────────────
        stem = IMAGE_PATH.stem

        ela_path = OUTPUT_DIR / f"{stem}_ela.jpg"
        cv2.imwrite(str(ela_path), cv2.cvtColor(ela_display, cv2.COLOR_RGB2BGR))

        fft_path = OUTPUT_DIR / "fft_map.jpg"
        cv2.imwrite(str(fft_path), fft_map)

        mask_path = OUTPUT_DIR / f"{stem}_mask_raw.jpg"
        cv2.imwrite(str(mask_path), raw_mask)

        refined_path = OUTPUT_DIR / f"{stem}_mask_refined.jpg"
        cv2.imwrite(str(refined_path), mask_resized)

        overlay_path = OUTPUT_DIR / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_bgr)

        print(f"✅ Maps saved in 'temp/':")
        print(f"   • {ela_path.name}         – ELA heat-map")
        print(f"   • {fft_path.name}              – FFT artefact map")
        print(f"   • {mask_path.name}      – Raw adaptive mask")
        print(f"   • {refined_path.name}  – Morphologically refined mask")
        print(f"   • {overlay_path.name}       – ★ Red overlay (main result)")

        open_image(str(overlay_path))

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()