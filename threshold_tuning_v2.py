
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64

MODEL_PATH = "models/best_model_v2.keras"
MANIFEST_PATH = "results/validation_manifest.csv"

os.makedirs("results", exist_ok=True)

CLASS_NAMES = ["FAKE", "REAL"]

print("CNN V2 Threshold Tuning")
print("=" * 45)


# ==========================================
# 2. LOAD VALIDATION MANIFEST
# ==========================================

if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(
        "Validation manifest not found!"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = ["image_path", "label", "class_name"]

for column in required_columns:
    if column not in manifest.columns:
        raise ValueError(
            f"Missing column: {column}"
        )

print("\nManifest loaded successfully.")

print("Total images:", len(manifest))


# ==========================================
# 3. VALIDATE DATASET
# ==========================================

if len(manifest) != 20000:
    raise ValueError(
        "Expected exactly 20,000 validation images."
    )

y_true = manifest["label"].to_numpy(dtype=int)

paths = manifest["image_path"].astype(str).to_numpy()

if set(np.unique(y_true)) != {0, 1}:
    raise ValueError(
        "Both FAKE (0) and REAL (1) labels are required."
    )

fake_count = np.sum(y_true == 0)
real_count = np.sum(y_true == 1)

if fake_count != 10000 or real_count != 10000:
    raise ValueError(
        "Validation class distribution is incorrect!"
    )

expected_names = np.where(y_true == 0, "FAKE", "REAL")

if not np.array_equal(
    manifest["class_name"].to_numpy(),
    expected_names
):
    raise ValueError(
        "Class names do not match numeric labels!"
    )

print("\nClass distribution verified:")
print("FAKE:", fake_count)
print("REAL:", real_count)


# ==========================================
# 4. VERIFY IMAGE PATHS
# ==========================================

print("\nChecking image paths...")

missing_paths = [
    path for path in paths
    if not os.path.isfile(path)
]

if missing_paths:
    raise FileNotFoundError(
        f"{len(missing_paths)} validation images are missing. "
        f"First missing path: {missing_paths[0]}"
    )

print("All 20,000 image paths exist!")


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "CNN V2 checkpoint not found!"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("\nCNN V2 model loaded successfully!")


# ==========================================
# 6. LOAD VALIDATION IMAGES
# ==========================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    return image


val_data = tf.data.Dataset.from_tensor_slices(paths)

val_data = val_data.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_data = val_data.batch(BATCH_SIZE)

val_data = val_data.prefetch(tf.data.AUTOTUNE)


# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating validation predictions...")

y_prob = model.predict(
    val_data,
    verbose=1
).ravel()

if len(y_prob) != len(y_true):
    raise ValueError(
        "Prediction count does not match validation labels!"
    )

if not np.all(np.isfinite(y_prob)):
    raise ValueError(
        "Invalid prediction values detected!"
    )

if np.any(y_prob < 0) or np.any(y_prob > 1):
    raise ValueError(
        "Predictions must be between 0 and 1."
    )

print("\nPrediction verification passed!")


# ==========================================
# 8. TEST DIFFERENT THRESHOLDS
# ==========================================

thresholds = np.round(
    np.arange(0.02, 0.151, 0.005),
    3
)

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true, y_pred
    )

    precision = precision_score(
        y_true, y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred,
        labels=[0, 1]
    )

    fake_recall = cm[0, 0] / cm[0].sum()

    real_recall = cm[1, 1] / cm[1].sum()

    results.append({

        "Threshold": threshold,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro F1": macro_f1,
        "Precision REAL": precision,
        "Recall REAL": recall,
        "Recall FAKE": fake_recall,
        "F1 REAL": f1

    })


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

df = pd.DataFrame(results)

print("\n========== THRESHOLD COMPARISON ==========\n")

print(
    df.round(4).to_string(index=False)
)


# ==========================================
# 10. SELECT BEST THRESHOLD
# ==========================================

best_index = df["Balanced Accuracy"].idxmax()

best_threshold = float(
    df.loc[best_index, "Threshold"]
)

best_score = float(
    df.loc[best_index, "Balanced Accuracy"]
)

print("\n========== BEST THRESHOLD ==========")

print("Best Threshold:", best_threshold)

print(
    "Balanced Accuracy:",
    round(best_score * 100, 2), "%"
)


# ==========================================
# 11. FINAL VALIDATION CONFUSION MATRIX
# ==========================================

y_best = (
    y_prob >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_true,
    y_best,
    labels=[0, 1]
)

print("\nBest Threshold Confusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_best,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

df.to_csv(
    "results/threshold_comparison_v2.csv",
    index=False
)

with open(
    "results/best_threshold_v2.txt",
    "w"
) as file:

    file.write(str(best_threshold))


# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title(
    f"CNN V2 - Threshold {best_threshold}"
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks([0, 1], CLASS_NAMES)

plt.yticks([0, 1], CLASS_NAMES)

for i in range(2):
    for j in range(2):

        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/threshold_confusion_v2.png"
)

plt.show()


# ==========================================
# 14. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("THRESHOLD TUNING COMPLETED")
print("======================================")

print("Validation images:", len(y_true))
print("FAKE images:", fake_count)
print("REAL images:", real_count)

print("Best threshold:", best_threshold)

print("\nSaved files:")
print("- threshold_comparison_v2.csv")
print("- best_threshold_v2.txt")
print("- threshold_confusion_v2.png")

print("\nTest dataset was not used.")
