"""
Digit Recognition using a Single Layer Perceptron (SLP)

This script trains a simple neural network on the MNIST dataset to classify handwritten digits (0-9).


Credit: https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/
"""

import keras
import matplotlib.pyplot as plt

# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"len(x_train): {len(x_train)}")
print(f"len(x_test): {len(x_test)}")
print(f"x_train[0].shape: {x_train[0].shape}")
plt.matshow(x_train[0])
plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_flatten = x_train.reshape(len(x_train), 28 * 28)
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)
print(f"x_train_flatten.shape: {x_train_flatten.shape}")

# Build the model
model = keras.Sequential(
    [keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")]
)

model.summary()

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(x_train_flatten, y_train, epochs=30)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_flatten, y_test)
print(f"\nTest accuracy: {test_acc}")
