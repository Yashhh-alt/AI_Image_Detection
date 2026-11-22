
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset location
dataset_path = "dataset/train"

# Image categories
categories = ["REAL", "FAKE"]

# Create visualization
plt.figure(figsize=(12, 6))

for row, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    # Select 5 random images
    images = random.sample(os.listdir(folder_path), 5)

    for col, image_name in enumerate(images):

        image_path = os.path.join(folder_path, image_name)

        img = Image.open(image_path)

        plt.subplot(2, 5, row * 5 + col + 1)
        plt.imshow(img)
        plt.axis("off")

        if col == 2:
            plt.title(category, fontsize=14)

plt.suptitle("Real vs AI-Generated Images", fontsize=16)
plt.tight_layout()

# Save visualization
os.makedirs("results", exist_ok=True)
plt.savefig("results/sample_images.png")

plt.show()
