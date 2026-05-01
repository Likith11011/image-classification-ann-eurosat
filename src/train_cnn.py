import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "data/EuroSAT"
IMG_SIZE = 64

# ==============================
# LOAD DATA
# ==============================
print("Loading dataset...")

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

class_map = {name: i for i, name in enumerate(class_names)}

X, y = [], []

for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)

    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)

        if not os.path.isfile(fpath):
            continue

        try:
            img = Image.open(fpath).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            y.append(class_map[class_name])
        except:
            pass

    print(f"Loaded {class_name}")

X = np.array(X, dtype='float32') / 255.0
y = np.array(y)

# ==============================
# SPLIT
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train = to_categorical(y_train, len(class_names))
y_val   = to_categorical(y_val, len(class_names))

# ==============================
# CNN MODEL
# ==============================
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "best_cnn_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# ==============================
# TRAIN
# ==============================
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

# ==============================
# SAVE FINAL MODEL
# ==============================
model.save("final_cnn_model.keras")

print("CNN Training Completed!")
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("CNN Accuracy")
plt.savefig("results/cnn_accuracy.png")
plt.close()
