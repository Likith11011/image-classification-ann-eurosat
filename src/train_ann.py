import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

DATASET_PATH = "EuroSAT"   
IMG_SIZE = 64


print("Loading dataset...\n")

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

print("\nTotal images:", len(X))


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


mean = np.mean(X_train)
std  = np.std(X_train)

X_train = (X_train - mean) / std
X_val   = (X_val - mean) / std


with open("scaler.pkl", "wb") as f:
    pickle.dump((mean, std), f)

print("Scaler saved!")


y_train = to_categorical(y_train, len(class_names))
y_val   = to_categorical(y_val, len(class_names))


model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Flatten(),

    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),


    layers.Dense(len(class_names), activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()


early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    "best_ann_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)


print("\nTraining started...\n")

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    shuffle=True
)


model.save("final_ann_model.keras")
print("Model saved!")


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()
