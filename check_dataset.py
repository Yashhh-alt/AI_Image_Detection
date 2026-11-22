
import os

dataset_path = "dataset"

folders = [
    "train/REAL",
    "train/FAKE",
    "test/REAL",
    "test/FAKE"
]

for folder in folders:
    path = os.path.join(dataset_path, folder)

    if os.path.exists(path):
        images = [
            f for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"{folder}: {len(images)} images")
    else:
        print(f"{folder}: Folder not found!")
