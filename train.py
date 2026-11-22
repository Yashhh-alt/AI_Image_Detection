
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# =====================================
# 1. CONFIGURATION
# =====================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 15

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("TensorFlow Version:", tf.__version__)

# =====================================
# 2. LOAD TRAINING DATA
# =====================================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    validation_split=0.2,
    subset="training",
    seed=42
)

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
    seed=42
)

# =====================================
# 4. LOAD TEST DATA
# =====================================

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_data.class_names

print("\nClass Names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# =====================================
# 5. DATA AUGMENTATION
# =====================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# =====================================
# 6. BUILD CNN MODEL
# =====================================

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    layers.Rescaling(1./255),

    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# =====================================
# 7. COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# 8. CALLBACKS
# =====================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# =====================================
# 9. TRAIN MODEL
# =====================================

print("\nStarting CNN Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================
# 10. SAVE MODEL
# =====================================

model.save("models/ai_image_detector.keras")

print("\nModel saved successfully!")

# =====================================
# 11. EVALUATE ON TEST DATA
# =====================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)

# =====================================
# 12. CLASSIFICATION REPORT
# =====================================

y_true = np.concatenate([
    labels.numpy().ravel()
    for images, labels in test_data
])

y_prob = model.predict(test_data, verbose=1).ravel()

y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# =====================================
# 13. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

# =====================================
# 14. ACCURACY AND LOSS GRAPHS
# =====================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_graphs.png")
plt.show()

print("\nTraining and evaluation completed!")
