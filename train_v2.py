
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# 1. CONFIGURATION
# ==========================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 15
RANDOM_STATE = 42

TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

CLASS_NAMES = ["FAKE", "REAL"]

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

tf.keras.utils.set_random_seed(RANDOM_STATE)

print("TensorFlow Version:", tf.__version__)


# ==========================================
# 2. COLLECT IMAGE PATHS AND LABELS
# ==========================================

def collect_images(directory):

    image_paths = []
    image_labels = []

    for label, class_name in enumerate(CLASS_NAMES):

        folder = directory / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder}"
            )

        files = sorted([
            str(file)
            for file in folder.iterdir()
            if file.suffix.lower() in
            [".jpg", ".jpeg", ".png"]
        ])

        image_paths.extend(files)
        image_labels.extend([label] * len(files))

    return (
        np.array(image_paths),
        np.array(image_labels, dtype=np.int32)
    )


image_paths, image_labels = collect_images(TRAIN_DIR)

print("\nTotal training images:", len(image_paths))

print("Total FAKE:", np.sum(image_labels == 0))
print("Total REAL:", np.sum(image_labels == 1))


# ==========================================
# 3. STRATIFIED TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(

    image_paths,
    image_labels,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=image_labels

)

print("\n========== DATASET DISTRIBUTION ==========")

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

print("\nTraining class counts:")
print("FAKE:", np.sum(y_train == 0))
print("REAL:", np.sum(y_train == 1))

print("\nValidation class counts:")
print("FAKE:", np.sum(y_val == 0))
print("REAL:", np.sum(y_val == 1))


# ==========================================
# 4. VALIDATION SAFETY CHECKS
# ==========================================

fake_count = int(np.sum(y_val == 0))
real_count = int(np.sum(y_val == 1))

assert len(X_val) == 20000, \
    "Incorrect validation image count!"

assert fake_count == 10000, \
    "Incorrect FAKE validation count!"

assert real_count == 10000, \
    "Incorrect REAL validation count!"

assert set(np.unique(y_val)) == {0, 1}, \
    "Both classes must exist in validation data!"

print("\nVALIDATION CHECK PASSED!")

print("Both classes are present.")
print("Validation dataset is balanced.")


# ==========================================
# 5. SAVE VALIDATION MANIFEST
# ==========================================

validation_manifest = pd.DataFrame({

    "image_path": X_val,

    "label": y_val,

    "class_name": [
        CLASS_NAMES[label]
        for label in y_val
    ]

})

validation_manifest.to_csv(

    "results/validation_manifest.csv",

    index=False

)

print("\nValidation manifest saved!")

print(
    "Manifest images:",
    len(validation_manifest)
)


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

def load_image(path, label):

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

    return image, label


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(

            buffer_size=10000,

            seed=RANDOM_STATE,

            reshuffle_each_iteration=True

        )

    dataset = dataset.map(

        load_image,

        num_parallel_calls=tf.data.AUTOTUNE

    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_data = create_dataset(
    X_train,
    y_train,
    training=True
)

val_data = create_dataset(
    X_val,
    y_val
)


# ==========================================
# 7. LOAD TEST DATA
# ==========================================

test_paths, test_labels = collect_images(TEST_DIR)

print("\n========== TEST DATA DISTRIBUTION ==========")

print("Total test images:", len(test_paths))

print("FAKE:", np.sum(test_labels == 0))
print("REAL:", np.sum(test_labels == 1))

assert len(test_paths) == 20000, \
    "Incorrect test image count!"

assert np.sum(test_labels == 0) == 10000, \
    "Incorrect test FAKE count!"

assert np.sum(test_labels == 1) == 10000, \
    "Incorrect test REAL count!"

test_data = create_dataset(
    test_paths,
    test_labels
)

print("\nTest dataset verification passed!")


# ==========================================
# 8. DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.1)

])


# ==========================================
# 9. BUILD CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    layers.Rescaling(1./255),

    data_augmentation,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# ==========================================
# 10. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# ==========================================
# 11. CALLBACKS
# ==========================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True

    ),

    ModelCheckpoint(

        "models/best_model_v2.keras",

        monitor="val_loss",

        save_best_only=True

    )

]


# ==========================================
# 12. TRAIN CNN V2
# ==========================================

print("\nStarting CNN V2 Training...\n")

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=callbacks

)


# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

model.save("models/cnn_v2.keras")

print("\nCNN V2 saved successfully!")


# ==========================================
# 14. LOAD BEST MODEL
# ==========================================

best_model = tf.keras.models.load_model(
    "models/best_model_v2.keras"
)


# ==========================================
# 15. TEST EVALUATION
# ==========================================

test_loss, test_accuracy = best_model.evaluate(
    test_data
)

print("\n========== TEST RESULTS ==========")

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("Test Loss:", test_loss)


# ==========================================
# 16. CLASSIFICATION REPORT
# ==========================================

y_true = test_labels

y_prob = best_model.predict(
    test_data
).ravel()

y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:\n")

print(

    classification_report(

        y_true,

        y_pred,

        labels=[0, 1],

        target_names=CLASS_NAMES,

        zero_division=0

    )

)


# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=[0, 1]

)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title("CNN V2 Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks([0, 1], CLASS_NAMES)

plt.yticks([0, 1], CLASS_NAMES)

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix_v2.png"
)

plt.show()


# ==========================================
# 18. TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(12, 5))

# Accuracy graph

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.title("CNN V2 Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()


# Loss graph

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.title("CNN V2 Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_graphs_v2.png"
)

plt.show()


# ==========================================
# 19. FINAL VERIFICATION
# ==========================================

print("\n================================")

print("CNN V2 TRAINING COMPLETED!")

print("Validation manifest verified.")

print("Model saved successfully.")

print("Evaluation completed.")

print("================================")
