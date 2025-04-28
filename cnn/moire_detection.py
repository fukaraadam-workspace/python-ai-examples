from pathlib import Path
from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
import pywt
import tensorflow as tf
from keras import layers


def debug_batch_data(dataset: list[Any], debug_name: str, output_dir: str):
    i = 0
    for images, labels in dataset.take(2):
        print(f"---{debug_name} {i}---")
        print(f"image batch shape: {images.shape}")
        print(f"label batch shape: {labels.shape}")
        print(
            f"image shape: {images[0].shape}, min: {int(np.min(images[0]))}, max: {int(np.max(images[0]))}, mean: {int(np.mean(images[0]))}"
        )
        # for k in range(len(labels)):
        #     if labels[k] == 1:
        #         print(f"Label ex_image: {class_names[labels[k]]}")
        #         ex_image = images[k].numpy()
        #         break
        plot_batch(images, labels, class_names, output_dir / f"{debug_name}_{i}")
        i += 1


def plot_batch(
    image_batch: list[Any],
    labels_batch: list[int],
    class_names: list[str],
    output_path: str,
):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(image_batch))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels_batch[i])])
        plt.axis("off")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_history(history: Any, output_path: str):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_conv2d_output(
    model: keras.Model, layer_name: str, input_image: list[Any], output_path: str
):
    """
    Visualize the output of a Conv2D layer for a given input image.

    Parameters:
    ----------
    model : keras.Model
        The trained Keras model.
    layer_name : str
        The name of the Conv2D layer to visualize.
    input_image : np.ndarray
        The input image (must match the model's input shape).
    output_path : str
        The path to save the output image.

    Returns:
    -------
    None
    """
    # Create a sub-model that outputs the activations of the specified layer
    layer_output = model.get_layer(layer_name).output
    activation_model = keras.Model(inputs=model.inputs, outputs=layer_output)

    # Get the activations for the input image
    activations = activation_model.predict(np.expand_dims(input_image, axis=0))

    # Plot the feature maps
    num_filters = activations.shape[-1]  # Number of filters in the layer
    # size = activations.shape[1]  # Spatial size of the feature map

    plt.figure(figsize=(15, 15))
    for i in range(num_filters):
        plt.subplot(int(np.sqrt(num_filters)), int(np.sqrt(num_filters)), i + 1)
        plt.imshow(activations[0, :, :, i], cmap="viridis")
        plt.axis("off")
    plt.suptitle(f"Feature Maps from Layer: {layer_name}", fontsize=16)
    plt.savefig(output_path, dpi=300)


class VisualizeConv2DCallback(keras.callbacks.Callback):
    def __init__(self, layer_name, input_image, output_dir):
        super().__init__()
        self.layer_name = layer_name
        self.input_image = input_image
        self.output_dir = output_dir

    def on_epoch_end(self, epoch: int, logs=None):
        plot_conv2d_output(
            self.model,
            self.layer_name,
            self.input_image,
            self.output_dir / f"conv2d_output_epoch_{epoch + 1}.png",
        )


def wavelet_transform(image: list[Any], wavelet="haar"):
    # scaled_image = image.numpy().astype("uint8")
    scaled_image = np.squeeze(image.numpy()).astype("float32")

    coeffs = pywt.dwt2(scaled_image, wavelet)
    cA, (cH, cV, cD) = coeffs
    # Normalize each coefficient to [0, 1]
    cA = cA / np.max(np.abs(cA))
    cH = cH / np.max(np.abs(cH))
    cV = cV / np.max(np.abs(cV))
    cD = cD / np.max(np.abs(cD))
    # resized_image = tf.image.resize(image, size=cA.shape)
    # print(f"resized_image shape: {resized_image[:, :, 0].shape}")
    # print(f"cA shape: {cA.shape}")
    combined = np.stack([cA, cH, cV, cD], axis=-1)
    return combined


def tf_wavelet_transform(images, wavelet_size):
    def tf_wavelet_transform_single(image):
        transformed_components = tf.py_function(
            func=wavelet_transform,
            inp=[image],
            Tout=tf.float32,
        )
        transformed_components.set_shape(wavelet_size)
        return transformed_components

    transformed_batch = tf.map_fn(
        tf_wavelet_transform_single,
        images,
        fn_output_signature=tf.float32,
    )
    return transformed_batch


def se_block(in_block, ch, ratio=16):
    # Squeeze: Global Average Pooling converts each channel to a single numerical value
    y = layers.GlobalAveragePooling2D()(in_block)

    # Excitation: Two Dense blocks transform the n values to n weights for each channel
    # The first layer with a ReLU activation
    y = layers.Dense(ch // ratio, activation="relu")(y)
    # The second (last) layer with a sigmoid activation (acting as a smooth gating function)
    y = layers.Dense(ch, activation="sigmoid")(y)

    # Scale and Combine: Apply weights to the channels by element-wise multiplication
    return layers.multiply([in_block, y])


def build_model(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)

    def conv_branch(input_tensor):
        x = layers.Conv2D(
            32,
            (5, 5),
            activation="relu",
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.01),
        )(input_tensor)
        x = layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="same")(x)
        x = layers.Conv2D(64, (7, 7), activation="relu", padding="same")(x)
        x = layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="same")(x)
        # x = layers.Conv2D(128, (7, 7), activation="relu", padding="same")(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        return x

    base_input = layers.Lambda(lambda t: t[:, :, :, 0:4])(inputs)
    # LL_input = layers.Lambda(lambda t: t[:, :, :, 0:1])(inputs)
    LH_input = layers.Lambda(lambda t: t[:, :, :, 1:2])(inputs)
    HL_input = layers.Lambda(lambda t: t[:, :, :, 2:3])(inputs)
    HH_input = layers.Lambda(lambda t: t[:, :, :, 3:4])(inputs)
    base_features = conv_branch(base_input)
    # LL_features = conv_branch(LL_input)
    LH_features = conv_branch(LH_input)
    HL_features = conv_branch(HL_input)
    HH_features = conv_branch(HH_input)

    x = layers.Concatenate()([base_features, LH_features, HL_features, HH_features])
    # x = layers.Dense(128, activation="sigmoid")(merged_features)
    # Convolutional Layers
    # x = layers.Conv2D(
    #     16,
    #     (5, 5),
    #     strides=(2, 2),
    #     padding="same",
    #     activation="relu",
    #     kernel_regularizer=keras.regularizers.l2(0.001),
    # )(inputs)
    # x = se_block(x, ch=16)  # Add SE block
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    # x = se_block(x, ch=64)  # Add SE block
    # x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and Dense Layers
    x = layers.Dropout(0.6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(
        32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_model_sequential(input_shape, num_classes=2):
    model = keras.models.Sequential()

    model.add(layers.Input(shape=input_shape))
    # Convolutional Layers
    model.add(
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    )
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense Layers
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Credit: https://github.com/AmadeusITGroup/Moire-Pattern-Detection/blob/master/src/mCNN.py
def createModel(input_shape, num_classes):
    #     num_epochs = 20 # 50 26 200 # we iterate 200 times over the entire training set
    kernel_size_1 = 7  # we will use 7x7 kernels
    kernel_size_2 = 3  # we will use 3x3 kernels
    pool_size = 2  # we will use 2x2 pooling throughout
    conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 16  # ...switching to 16 after the first pooling layer
    drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
    hidden_size = 32  # 128 512 the FC layer will have 512 neurons

    # # depth goes last in TensorFlow back-end (first in Theano)
    # inpLL = layers.Input(shape=(height, width, depth))
    # # depth goes last in TensorFlow back-end (first in Theano)
    # inpLH = layers.Input(shape=(height, width, depth))
    # # depth goes last in TensorFlow back-end (first in Theano)
    # inpHL = layers.Input(shape=(height, width, depth))
    # # depth goes last in TensorFlow back-end (first in Theano)
    # inpHH = layers.Input(shape=(height, width, depth))
    inputs = layers.Input(shape=input_shape)
    inpLL = layers.Lambda(lambda t: t[:, :, :, 0:1])(inputs)
    inpLH = layers.Lambda(lambda t: t[:, :, :, 1:2])(inputs)
    inpHL = layers.Lambda(lambda t: t[:, :, :, 2:3])(inputs)
    inpHH = layers.Lambda(lambda t: t[:, :, :, 3:4])(inputs)

    # conv_1_LL = layers.Convolution2D(
    #     conv_depth_1, (kernel_size_1, kernel_size_1), padding="same", activation="relu"
    # )(inpLL)
    conv_1_LH = layers.Convolution2D(
        conv_depth_1, (kernel_size_1, kernel_size_1), padding="same", activation="relu"
    )(inpLH)
    conv_1_HL = layers.Convolution2D(
        conv_depth_1, (kernel_size_1, kernel_size_1), padding="same", activation="relu"
    )(inpHL)
    conv_1_HH = layers.Convolution2D(
        conv_depth_1, (kernel_size_1, kernel_size_1), padding="same", activation="relu"
    )(inpHH)
    # pool_1_LL = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LL)
    pool_1_LH = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LH)
    pool_1_HL = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HL)
    pool_1_HH = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)

    inp_merged = layers.Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    # inp_merged = layers.Multiply()([pool_1_LL, avg_LH_HL_HH])
    C4 = layers.Convolution2D(
        conv_depth_2, (kernel_size_2, kernel_size_2), padding="same", activation="relu"
    )(inp_merged)
    S2 = layers.MaxPooling2D(pool_size=(4, 4))(C4)
    drop_1 = layers.Dropout(drop_prob_1)(S2)
    C5 = layers.Convolution2D(
        conv_depth_1, (kernel_size_2, kernel_size_2), padding="same", activation="relu"
    )(drop_1)
    S3 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = layers.Convolution2D(
        conv_depth_1, (kernel_size_2, kernel_size_2), padding="same", activation="relu"
    )(S3)
    S4 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = layers.Dropout(drop_prob_1)(S4)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = layers.Flatten()(drop_2)
    hidden = layers.Dense(hidden_size, activation="relu")(flat)
    drop_3 = layers.Dropout(drop_prob_2)(hidden)
    outputs = layers.Dense(num_classes, activation="softmax")(drop_3)

    # model = layers.Model(
    #     inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out
    # )  # To define a model, just specify its input and output layers

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    DATASET_DIR = Path.home() / "dataset"
    OUTPUT_DIR = Path.cwd() / ".volume" / "moire_detection"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATASET_DIR / "moire_classification"
    test_dir = DATASET_DIR / "private_moire"

    img_size = (640, 480)
    wavelet_size = (img_size[0] // 2, img_size[1] // 2, 4)
    input_shape = wavelet_size
    # input_shape = img_size + (1,)
    batch_size = 8

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=797,
        validation_split=0.2,
        subset="both",
    )
    class_names = train_ds.class_names

    test_ds = keras.utils.image_dataset_from_directory(
        directory=test_dir,
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=797,
    )

    # Enable dataset caching and prefetching for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds_preprocessed = train_ds.map(
        lambda x, y: (tf_wavelet_transform(x, wavelet_size), y)
    )
    val_ds_preprocessed = val_ds.map(
        lambda x, y: (tf_wavelet_transform(x, wavelet_size), y)
    )
    test_ds_preprocessed = test_ds.map(
        lambda x, y: (tf_wavelet_transform(x, wavelet_size), y)
    )

    # train_ds_preprocessed = train_ds
    # val_ds_preprocessed = val_ds
    # test_ds_preprocessed = test_ds

    model = createModel(input_shape, num_classes=2)
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "weights_epoch_{epoch:02d}.keras"),
        save_best_only=False,  # Save model at each epoch
        save_weights_only=False,
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=3
    )

    print(f"Class names: {class_names}")
    debug_batch_data(train_ds, "original_batch", OUTPUT_DIR)

    i = 0
    for images, labels in train_ds_preprocessed.take(2):
        for k in range(len(labels)):
            if labels[k] == 1:
                ex_image = images[k].numpy()
                break
        # visualize_wavelet(images[0], f"batch{i}_image{0}")
        # visualize_wavelet(images[1], f"batch{i}_image{1}")
        # visualize_wavelet(images[2], f"batch{i}_image{2}")
        i += 1

    visualize_callback = VisualizeConv2DCallback(
        layer_name="conv2d",  # Name of the Conv2D layer to visualize
        input_image=ex_image,  # Example input image
        output_dir=OUTPUT_DIR,  # Directory to save visualizations
    )

    history = model.fit(
        train_ds_preprocessed,
        validation_data=val_ds_preprocessed,
        epochs=40,
        callbacks=[model_checkpoint, early_stopping, lr_schedule],
    )

    # plt.figure(figsize=(5, 5))
    # plt.imshow(ex_image[:, :, 0].squeeze(), cmap="gray")  # Display the original image
    # plt.title("Original Image")
    # plt.axis("off")
    # plt.show(block=False)
    # plot_conv2d_output(model, layer_name="conv2d", input_image=ex_image[:, :, 0])

    plot_training_history(history, OUTPUT_DIR / "training_history.png")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_ds_preprocessed)
    print(f"Test Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")

    # # Save the model for future use
    model.save(OUTPUT_DIR / "moire_detection.keras")
