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

    # delete the readme file
    kaggle_readme = dataset_path / "readme.md"
    if kaggle_readme.exists():
        os.remove(kaggle_readme)
        print("Removed Kaggle-specific README.md from dataset folder.")

    # remove the nested folders
    nested_path = dataset_path / "astro_dataset_maxia" / "astro_dataset_maxia"
    
    if nested_path.exists():
        print("Flattening nested directory structure...")
        # Move all items from the deep folder to the 'dataset/' folder
        for item in nested_path.iterdir():
            dest = dataset_path / item.name
            # If a folder with the same name exists in root, remove it first to avoid errors
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    os.remove(dest)
            shutil.move(str(item), str(dataset_path))
        
        # Remove the now empty 'astro_dataset_maxia' directory tree
        shutil.rmtree(dataset_path / "astro_dataset_maxia")
        print("Flattening complete.")

    # We use rglob to find all files in subdirectories
    for img_path in dataset_path.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.png', '.webp', '.bmp', '.tif', '.tiff', '.jpeg', '.jpg']:
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
