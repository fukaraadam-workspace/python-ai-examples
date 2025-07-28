"""
Digit Recognition using a Convolutional Neural Network (CNN)

This script trains a CNN on the MNIST dataset to classify handwritten digits (0-9).
"""

## Importing necessary modules
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.api.models import Sequential

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize image pixel values by dividing by 255 (grayscale)
gray_scale = 255
x_train = x_train.astype("float32") / gray_scale
x_test = x_test.astype("float32") / gray_scale

# Reshape the data to add a channel dimension (for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)  # Shape: (n_samples, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)  # Shape: (n_samples, 28, 28, 1)

# Checking the shape of feature and target matrices
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)

# Visualizing 100 images from the training data
fig, ax = plt.subplots(10, 10, figsize=(10, 10))
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28), aspect="auto", cmap="gray")
        ax[i][j].axis("off")  # Hide axes for better visualization
        k += 1
plt.suptitle("Sample Images from MNIST Dataset", fontsize=16)
plt.show()

# Building the CNN model
model = Sequential(
    [
        # Convolutional layer 1
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        # MaxPooling layer 1
        MaxPooling2D(pool_size=(2, 2)),
        # Convolutional layer 2
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # MaxPooling layer 2
        MaxPooling2D(pool_size=(2, 2)),
        # Flatten the feature maps
        Flatten(),
        # Fully connected Dense layer 1
        Dense(128, activation="relu"),
        # Output layer (10 classes)
        Dense(10, activation="softmax"),
    ]
)

# Compiling the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# Training the model with training data
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluating the model on test data
results = model.evaluate(x_test, y_test, verbose=0)
print("Test loss, Test accuracy:", results)

# Visualization of Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 5))

# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
plt.title("Training and Validation Accuracy", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True)

# Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
plt.title("Training and Validation Loss", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True)

plt.suptitle("Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()
