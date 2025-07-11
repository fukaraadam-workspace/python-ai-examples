"""
Digit Recognition using a Fully Connected Neural Network (FCNN)

This script trains a simple neural network on the MNIST dataset to classify handwritten digits (0-9).
The model consists of a Flatten layer, one hidden Dense layer, and an output Dense layer with softmax activation.

Credit: https://www.geeksforgeeks.org/feedforward-neural-network/
"""

import keras
from keras.api.layers import Dense, Flatten
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.metrics import SparseCategoricalAccuracy
from keras.api.models import Sequential
from keras.api.optimizers import Adam

# Load and prepare the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

print(model.summary())

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()],
)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc}")
