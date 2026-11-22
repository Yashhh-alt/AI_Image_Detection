
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64

THRESHOLD = 0.070

MODEL_PATH = "models/best_model_v2.keras"
TEST_DIR = Path("dataset/test")

CLASS_NAMES = ["FAKE", "REAL"]

os.makedirs("results", exist_ok=True)

print("FINAL CNN V2 EVALUATION")
print("=" * 45)


# ==========================================
# 2. COLLECT TEST IMAGES
# ==========================================

image_paths = []
y_true = []

for label, class_name in enumerate(CLASS_NAMES):

    folder = TEST_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(folder)

    files = sorted([
        str(file)
        for file in folder.iterdir()
        if file.suffix.lower() in
        [".jpg", ".jpeg", ".png"]
    ])

    image_paths.extend(files)
    y_true.extend([label] * len(files))

image_paths = np.array(image_paths)
y_true = np.array(y_true)

print("\nTest Distribution:")
print("Total:", len(y_true))
print("FAKE:", np.sum(y_true == 0))
print("REAL:", np.sum(y_true == 1))

# Safety checks

assert len(y_true) == 20000
assert np.sum(y_true == 0) == 10000
assert np.sum(y_true == 1) == 10000

print("\nTest dataset verification passed!")


# ==========================================
# 3. LOAD CNN V2
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nCNN V2 loaded successfully!")


# ==========================================
# 4. IMAGE PREPROCESSING
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


test_data = tf.data.Dataset.from_tensor_slices(
    image_paths
)

test_data = test_data.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_data = test_data.batch(BATCH_SIZE)

test_data = test_data.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating test predictions...")

y_prob = model.predict(
    test_data,
    verbose=1
).ravel()

assert len(y_prob) == len(y_true)

assert np.all(np.isfinite(y_prob))

assert np.all(
    (y_prob >= 0) & (y_prob <= 1)
)

print("\nPrediction verification passed!")


# ==========================================
# 6. APPLY SELECTED THRESHOLD
# ==========================================

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ==========================================
# 7. CALCULATE METRICS
# ==========================================

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


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL TEST RESULTS ==========")

print("Threshold:", THRESHOLD)

print("Accuracy:", round(accuracy * 100, 2), "%")

print(
    "Balanced Accuracy:",
    round(balanced_acc * 100, 2), "%"
)

print("Macro F1:", round(macro_f1 * 100, 2), "%")

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

plt.imshow(cm, cmap="Blues")

plt.title(
    f"Final CNN V2 - Threshold {THRESHOLD}"
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
    "results/final_confusion_matrix.png"
)

plt.show()


# ==========================================
# 10. SAVE FINAL METRICS
# ==========================================

metrics = pd.DataFrame([{

    "Model": "CNN V2",

    "Threshold": THRESHOLD,

    "Accuracy": accuracy,

    "Balanced Accuracy": balanced_acc,

    "Macro F1": macro_f1,

    "Precision REAL": precision,

    "Recall REAL": recall,

    "F1 Score REAL": f1

}])

metrics.to_csv(
    "results/final_metrics.csv",
    index=False
)


# ==========================================
# 11. SAVE PREDICTIONS
# ==========================================

predictions = pd.DataFrame({

    "Image": image_paths,

    "Actual": [
        CLASS_NAMES[x] for x in y_true
    ],

    "Predicted": [
        CLASS_NAMES[x] for x in y_pred
    ],

    "REAL Probability": y_prob,

    "Threshold": THRESHOLD

})

predictions.to_csv(
    "results/final_predictions.csv",
    index=False
)


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n===================================")

print("FINAL EVALUATION COMPLETED!")

print("Selected Threshold:", THRESHOLD)

print("Results saved in results folder.")

print("===================================")

