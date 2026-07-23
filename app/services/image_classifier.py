"""
Image classification service.

Classifies images as 'photo' or 'document' using lightweight heuristics.
Photos go to ASIORGA/Fotos/, documents/screenshots get OCR'd and classified.
"""

from pathlib import Path
from typing import Optional

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class ImageClassifier:
    """Classifies images as photos or documents.

    Uses heuristic-based analysis:
    - EXIF data (cameras add EXIF → likely photo)
    - Aspect ratio and resolution
    - Color variance (documents tend to be low-variance)
    - Brightness distribution

    Classification result determines whether the image gets:
    - 'photo': Moved directly to ASIORGA/Fotos/
    - 'document': OCR'd and classified like any other document
    """

    def classify(self, image_path: Path) -> str:
        """Classify an image as 'photo' or 'document'.

        Args:
            image_path: Path to the image file.

        Returns:
            'photo' or 'document'.
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(str(image_path))
            img_array = np.array(img)

            score = 0.0

            # Check EXIF data (cameras add EXIF → likely photo)
            exif = img.getexif() if hasattr(img, "getexif") else {}
            if exif:
                # Camera manufacturer tags
                camera_tags = {271, 272, 306}  # Make, Model, DateTime
                if any(tag in exif for tag in camera_tags):
                    score += 0.4

            # Color variance (photos have more color variety)
            if len(img_array.shape) == 3:
                # Calculate color channel variance
                variance = np.var(img_array, axis=(0, 1))
                avg_variance = np.mean(variance)
                if avg_variance > 1500:  # High variance → likely photo
                    score += 0.3
                elif avg_variance < 500:  # Low variance → likely document
                    score -= 0.3

            # Aspect ratio (standard photo ratios)
            w, h = img.size
            aspect = w / h if h > 0 else 1.0
            photo_ratios = {4 / 3, 3 / 4, 16 / 9, 9 / 16, 1.0}
            if any(abs(aspect - r) < 0.1 for r in photo_ratios):
                score += 0.1

            # Resolution (photos tend to be higher res)
            megapixels = (w * h) / 1_000_000
            if megapixels > 2.0:
                score += 0.1
            elif megapixels < 0.3:
                score -= 0.1

            # Brightness distribution (documents are often mostly white)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array.astype(float)

            white_ratio = np.mean(gray > 200)
            if white_ratio > 0.7:  # Mostly white → likely document
                score -= 0.2

            classification = "photo" if score >= 0 else "document"
            logger.info(
                "Image classification for %s: %s (score=%.2f)",
                image_path.name,
                classification,
                score,
            )
            return classification

        except Exception as e:
            logger.error("Image classification failed for %s: %s", image_path, e)
            # Default to photo to avoid unnecessary OCR
            return "photo"
