import cv2
import numpy as np
from pathlib import Path

class ELAEngine:
    def __init__(self, quality=90, scale_factor=15):
        self.quality = quality
        self.scale_factor = scale_factor
        print(f"✅ ELAEngine initialized with quality={quality}%, scale={scale_factor}")

    def save_compressed(self, img_rgb, temp_dir="temp"):
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        temp_file = temp_path / "compressed.jpg"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(temp_file), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return temp_file

    def calculate_difference(self, original_rgb, compressed_path):
        compressed_bgr = cv2.imread(str(compressed_path))
        compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)
        if original_rgb.shape != compressed_rgb.shape:
            compressed_rgb = cv2.resize(compressed_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
        diff = cv2.absdiff(original_rgb, compressed_rgb)
        return diff

    def scale_difference(self, diff):
        scaled = diff * self.scale_factor
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        return scaled

    def calculate_ela(self, img_rgb, return_diff_only=False):
        """
        Returns ELA map, compressed image and raw difference.
        If return_diff_only=True, returns (ela_result, diff_raw)
        """
        compressed_path = self.save_compressed(img_rgb)
        diff_raw = self.calculate_difference(img_rgb, compressed_path)
        ela_result = self.scale_difference(diff_raw)

        compressed_path.unlink()  # remove temporary file

        if return_diff_only:
            return ela_result, diff_raw
        else:
            return ela_result, img_rgb, diff_raw  # keep img_rgb for reference

    def get_ela_score(self, diff_raw):
        gray = cv2.cvtColor(diff_raw, cv2.COLOR_RGB2GRAY) if diff_raw.ndim==3 else diff_raw
        score = min(float(np.percentile(gray, 95)) / 255 * 100, 100)
        return min(score, 100)