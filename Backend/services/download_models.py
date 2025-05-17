import os
import urllib.request
import sys


def download_sr_models():
    """Download pre-trained super-resolution models"""
    models_dir = os.path.join("..", "assets", "sr_models")
    os.makedirs(models_dir, exist_ok=True)

    # Define models to download
    models = [
        ("ESPCN_x2.pb", "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb"),
        ("FSRCNN_x2.pb", "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"),
        ("LapSRN_x2.pb", "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb")
    ]

    for model_name, url in models:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"Downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"{model_name} already exists")


if __name__ == "__main__":
    download_sr_models()
