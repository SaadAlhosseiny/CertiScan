import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        print(f"✅ Preprocessor initialized with target size: {target_size}")

    def read_image(self, image_path):
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"❌ File not found: {image_path}")
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"❌ Cannot read image: {image_path}")
        print(f"✅ Image read: {img_path.name}")
        return img

    def convert_rgb(self, img):
        if len(img.shape) == 2:  # grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✅ Converted to RGB")
        return img_rgb

    def resize_image(self, img):
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        print(f"✅ Resized to {self.target_size}")
        return img_resized

    def remove_metadata(self, img_rgb, temp_dir="temp"):
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        temp_file = temp_path / "no_metadata.jpg"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(temp_file), img_bgr, [
            cv2.IMWRITE_JPEG_QUALITY, 100,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0
        ])
        img_clean_bgr = cv2.imread(str(temp_file))
        img_clean_rgb = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2RGB)
        temp_file.unlink()
        print(f"✅ Metadata removed")
        return img_clean_rgb

    def preprocess(self, image_path):
        print(f"\n🚀 Starting preprocessing for: {image_path}")
        img = self.read_image(image_path)
        img_rgb = self.convert_rgb(img)
        img_resized = self.resize_image(img_rgb)
        img_clean = self.remove_metadata(img_resized)
        print(f"✅ Preprocessing completed!")
        return img_clean