
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================
# 1. CONFIGURATION
# =====================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64

TRAIN_DIR = "dataset/train"
MODEL_PATH = "models/best_model.keras"

os.makedirs("results", exist_ok=True)

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

print("Loading CNN V1 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =====================================
# 3. LOAD VALIDATION DATA
# =====================================

val_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=False
)

class_names = val_data.class_names

print("Classes:", class_names)

# =====================================
# 4. GET TRUE LABELS
# =====================================

y_true = np.concatenate([
    labels.numpy().ravel()
    for images, labels in val_data
]).astype(int)

# =====================================
# 5. GET MODEL PREDICTIONS
# =====================================

print("\nGenerating predictions...")

y_prob = model.predict(val_data).ravel()

# =====================================
# 6. TEST DIFFERENT THRESHOLDS
# =====================================

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true, y_pred
    )

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# =====================================
# 7. DISPLAY RESULTS
# =====================================

df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON\n")

print(df.round(4).to_string(index=False))

# =====================================
# 8. SELECT BEST THRESHOLD
# =====================================

best_index = df["Balanced Accuracy"].idxmax()

best_threshold = df.loc[best_index, "Threshold"]

print("\nBest Threshold:", best_threshold)

print(
    "Best Balanced Accuracy:",
    round(df.loc[best_index, "Balanced Accuracy"] * 100, 2),
    "%"
)

# =====================================
# 9. CONFUSION MATRIX
# =====================================

y_best = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_best)

print("\nConfusion Matrix:")
print(cm)

# =====================================
# 10. SAVE RESULTS
# =====================================

df.to_csv("results/threshold_comparison.csv", index=False)

with open("results/best_threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("\nResults saved successfully!")
print("Location: results/threshold_comparison.csv")
print("Threshold saved: results/best_threshold.txt")
