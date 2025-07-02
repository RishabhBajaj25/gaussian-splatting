import cv2
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from torch.nn.quantized.functional import threshold
from tqdm import tqdm
import os.path as osp

# Initialize lists to store images
blurry_images = []
not_blurry_images = []

import cv2
import os

def load_images_from_folder(folder):
    images = []
    print("\nExtracting images from directory...")
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

dir_path = "/media/rishabh/SSD_1/Data/UTokyo/RGB_coin_purse/2025-01-16--06-12-50/EXR_RGBD/rgb"

# Run a loop to read images from the directory
laplacian_vars = []

images = load_images_from_folder(dir_path)

print("\nMeasuring blurriness threshold...")
for image in tqdm(images):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    laplacian_vars.append(laplacian_var)

threshold = np.percentile(laplacian_vars,80)

# make histogram of laplacian_vars
# add a vertical line at threshold
plt.figure(figsize=(10, 5))
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold (80th Percentile)')
plt.hist(laplacian_vars, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Laplacian Variance')
plt.xlabel('Laplacian Variance')
plt.ylabel('Frequency')
plt.show()


print("\nFiltering based on threshold...")
for image in tqdm(images):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if laplacian_var > threshold:
        not_blurry_images.append(image)
    else:
        blurry_images.append(image)
    laplacian_vars.append(laplacian_var)


sharp_dir = osp.join(dir_path,"sharp")
blurry_dir = osp.join(dir_path,"blurry")

os.makedirs(sharp_dir, exist_ok=True)
os.makedirs(blurry_dir, exist_ok=True)

print("\nSaving sharp images...")
for idx, img in enumerate(not_blurry_images):
    cv2.imwrite(osp.join(sharp_dir,f"img_{idx:04d}.png"), img)

print("\nSaving blurry images...")
for idx, img in enumerate(blurry_images):
    cv2.imwrite(f"blurry/img_{idx:04d}.png", img)

