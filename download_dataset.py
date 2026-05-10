#!/usr/bin/env python3

import opendatasets as od
import os
import shutil
from PIL import Image
from pathlib import Path

RAW_DATASET_NAME = "astrophysical-objects-image-dataset"
FINAL_DATASET_NAME = "dataset"

def setup_dataset():
    # download the dataset
    if not os.path.exists(FINAL_DATASET_NAME):
        print("Downloading dataset...")
        dataset_url = f'https://www.kaggle.com/datasets/engeddy/{RAW_DATASET_NAME}'
        od.download(dataset_url)

        if os.path.exists(RAW_DATASET_NAME):
            os.rename(RAW_DATASET_NAME, FINAL_DATASET_NAME)
            print(f"Renamed {RAW_DATASET_NAME} to {FINAL_DATASET_NAME}")

    else:
        print("Dataset folder already exists. Skipping download.")

    print("Starting image conversion to JPG...")
    dataset_path = Path(FINAL_DATASET_NAME)

    # We use rglob to find all files in subdirectories
    for img_path in dataset_path.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.png', '.webp', '.bmp', '.tiff', '.jpeg']:
            try:
                # Open image
                with Image.open(img_path) as img:
                    # Convert to RGB (required for JPG if source is RGBA/PNG)
                    rgb_img = img.convert('RGB')
                
                    # Create new filename
                    new_path = img_path.with_suffix('.jpg')
                
                    # Save as JPG
                    rgb_img.save(new_path, 'JPEG', quality=90)
            
                # Delete original if it wasn't already a .jpg
                if img_path.suffix.lower() != '.jpg':
                    os.remove(img_path)
                
            except Exception as e:
                print(f"Could not convert {img_path}: {e}")

    print("Post-processing complete! All images are now .jpg")

if __name__ == "__main__":
    setup_dataset()
