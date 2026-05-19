#!/usr/bin/env python3

import os
from torchvision.datasets import CIFAR10
from PIL import Image

# 1. Download the official dataset via torchvision
train_set = CIFAR10(root='./cifar10_raw', train=True, download=True)
test_set = CIFAR10(root='./cifar10_raw', train=False, download=True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def save_images(dataset, folder_name):
    for i, (img, label_idx) in enumerate(dataset):
        # Organize into class-named folders (e.g., output/train/cat/0001.png)
        class_name = classes[label_idx]
        directory = os.path.join('dataset', folder_name, class_name)
        os.makedirs(directory, exist_ok=True)
        
        # Save as PNG
        img.save(os.path.join(directory, f"{i:05d}.png"))

print("Extracting training images...")
save_images(train_set, 'train')
print("Extracting testing images...")
save_images(test_set, 'test')
print("Done! Check the 'output' directory.")
