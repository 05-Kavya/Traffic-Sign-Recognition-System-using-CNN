import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATASET_PATH = r"D:\Traffic_Sign_Recognition\dataset\GTSRB\Training" # Path of the Dataset

images = []
labels = []

print("Loading dataset...")

#Loading images


for class_id in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, class_id)

    if not os.path.isdir(class_path):
        continue

    print("Loading class:", class_id)

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (32, 32))

        images.append(image)
        labels.append(int(class_id))


print("\nDataset loaded successfully!")
print("Total images:", len(images))
print("Total classes:", len(set(labels)))


#Converting to NumPy arrays


X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Normalizing pixel values
X = X / 255.0

# Converting labels to categorical format
y = to_categorical(y, num_classes=43)


#Spliting the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

#Creating CNN model

model = Sequential([

    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(43, activation="softmax")
])



# Compile model


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#Training model


print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)



# Evaluate model

loss, accuracy = model.evaluate(X_test, y_test)

print("\n==============================")
print("Model Accuracy:", accuracy * 100, "%")
print("==============================")

# Save model


model.save("traffic_sign_model.keras")

print("\nModel saved successfully!")
print("File: traffic_sign_model.keras")