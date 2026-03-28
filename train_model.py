import os
import numpy as np
from preprocess import extract_features
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET = "human-nonhuman"

labels = {
    "human": 0,
    "nonhuman": 1,
    "possible": 2
}

X = []
y = []

print("Loading dataset...")

for label in labels:
    folder = os.path.join(DATASET, label)
    if not os.path.exists(folder):
        print("❌ Missing:", folder)
        continue
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features = extract_features(path, augment=True)  # apply augmentation during loading
        if features is not None:
            X.append(features)
            y.append(labels[label])
        else:
            print("Error:", file)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)

# CNN needs 4D input
X = X[..., np.newaxis]

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Improved CNN Model ---
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(3,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3)
]

print("Training model...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy*100)

model.save("model.h5")
print("✅ Model saved!")
