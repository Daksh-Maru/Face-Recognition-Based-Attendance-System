# services/inplace_preprocessor.py
import os
import cv2
import numpy as np
import shutil
from datetime import datetime
import logging
from tqdm import tqdm
from preprocessing import AdvancedImagePreprocessor

logger = logging.getLogger(__name__)

class InPlacePreprocessor:
    def __init__(self, create_backup=True):
        self.preprocessor = AdvancedImagePreprocessor()
        self.create_backup = create_backup
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }

    def process_dataset(self, dataset_path, backup_suffix="_backup"):
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Dataset path does not exist: {dataset_path}")
            return False

        # Create backup if needed
        if self.create_backup:
            backup_path = f"{dataset_path}{backup_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copytree(dataset_path, backup_path)
                logger.info(f"📦 Backup created at: {backup_path}")
            except Exception as e:
                logger.error(f"❌ Backup failed: {e}")
                return False

        # Gather all image files recursively
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if self._is_supported_image(file):
                    image_files.append(os.path.join(root, file))
        self.stats['total_images'] = len(image_files)

        # Process each image
        for img_path in tqdm(image_files, desc="In-place preprocessing"):
            try:
                success = self._process_single_image(img_path)
                if success:
                    self.stats['processed'] += 1
                else:
                    self.stats['failed'] += 1
            except Exception as e:
                logger.error(f"❌ Error processing {img_path}: {e}")
                self.stats['failed'] += 1

        # Summary
        total = self.stats['total_images']
        processed = self.stats['processed']
        skipped = self.stats['skipped']
        failed = self.stats['failed']
        success_rate = (processed / total) * 100 if total > 0 else 0
        logger.info(f"\n🎉 In-place preprocessing completed.")
        logger.info(f"📊 Total images: {total}")
        logger.info(f"✅ Processed: {processed}")
        logger.info(f"⏭️ Skipped (good quality): {skipped}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"✅ Success rate: {success_rate:.2f}%")
        return success_rate > 70  # threshold for success

    def _is_supported_image(self, filename):
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)

    def _process_single_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"❌ Cannot load image: {image_path}")
            return False

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assess quality
        quality_metrics = self.preprocessor.assess_image_quality(image_rgb)
        if quality_metrics['overall_quality'] >= 0.75:
            # Skip high-quality images
            self.stats['skipped'] += 1
            return True

        # Enhance image
        enhanced = self.preprocessor.enhance_for_recognition(image_rgb)
        if enhanced is None:
            return False

        # Standardize for recognition
        standardized = self.preprocessor.standardize_for_recognition(enhanced)
        if standardized is None:
            return False

        # Convert back to BGR for saving
        final_bgr = cv2.cvtColor(standardized, cv2.COLOR_RGB2BGR)

        # Save in place
        success = cv2.imwrite(image_path, final_bgr)
        return success

# Usage example
if __name__ == "__main__":
    dataset_dir = r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\dataset"  # Change as needed
    processor = InPlacePreprocessor(create_backup=True)
    processor.process_dataset(dataset_dir)
