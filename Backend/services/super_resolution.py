import cv2
import os
import numpy as np


class SuperResolution:
    def __init__(self, model_name="espcn", scale=3):
        """
        Initialize super-resolution model

        Args:
            model_name: Model architecture ('espcn', 'fsrcnn', or 'lapsrn')
            scale: Upscaling factor (2, 3, 4, or 8 depending on model)
        """
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # Define model paths
        models_dir = os.path.join("..", "assets", "sr_models")
        os.makedirs(models_dir, exist_ok=True)

        # Model file paths based on architecture and scale
        model_files = {
            "espcn": f"ESPCN_x{scale}.pb",
            "fsrcnn": f"FSRCNN_x{scale}.pb",
            "lapsrn": f"LapSRN_x{scale}.pb"
        }

        self.model_path = os.path.join(models_dir, model_files[model_name])

        # Download model if it doesn't exist
        if not os.path.exists(self.model_path):
            self._download_model(model_name, scale)

        # Load the model
        self.sr.readModel(self.model_path)
        self.sr.setModel(model_name, scale)

    def _download_model(self, model_name, scale):
        """Download pre-trained model if not available"""
        import urllib.request

        base_url = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/"
        model_urls = {
            "espcn": f"{base_url}ESPCN_x{scale}.pb",
            "fsrcnn": f"{base_url}FSRCNN_x{scale}.pb",
            "lapsrn": f"{base_url}LapSRN_x{scale}.pb"
        }

        print(f"Downloading {model_name} super-resolution model...")
        urllib.request.urlretrieve(model_urls[model_name], self.model_path)
        print(f"Model downloaded to {self.model_path}")

    def upsample(self, image):
        """
        Apply super-resolution to an image

        Args:
            image: Input image (BGR format)

        Returns:
            Super-resolved image
        """
        return self.sr.upsample(image)
