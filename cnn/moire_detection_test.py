from pathlib import Path

import cv2
import keras
import numpy as np
import pywt

# Load the trained model
MODEL_PATH = (
    Path(__file__).parent.parent
    / ".volume"
    / "moire_detection"
    / "moire_detection.keras"
)


model = keras.models.load_model(MODEL_PATH, safe_mode=False)


# Define the wavelet transform function
def wavelet_transform(image, wavelet="haar"):
    """
    Perform 2D wavelet decomposition on a grayscale image.
    """
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    cA = cA / np.max(np.abs(cA))
    cH = cH / np.max(np.abs(cH))
    cV = cV / np.max(np.abs(cV))
    cD = cD / np.max(np.abs(cD))
    combined = np.stack([cA, cH, cV, cD], axis=-1)
    return combined


# Preprocess the input image
def preprocess_image(image_path, img_size=(480, 640)):
    """
    Load and preprocess a single image for prediction.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Resize the image to the expected size
    image = cv2.resize(image, img_size)

    # Normalize the image to the range [0, 1]
    # image = image.astype("float32") / 255.0

    # Apply the wavelet transform
    transformed_image = wavelet_transform(image)

    # Add batch dimension (model expects a batch of images)
    transformed_image = np.expand_dims(transformed_image, axis=0)
    return transformed_image


# Predict the class of a single image
def predict_single_image(image_path, class_names):
    preprocessed_image = preprocess_image(image_path)

    # Make a prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")


# Main script
if __name__ == "__main__":
    IMAGE_DIR_PATH = (
        Path(__file__).parent.parent.parent.parent
        / "dataset"
        / "private_moire"
        / "spoof"
    )

    CLASS_NAMES = ["real_world", "spoof"]

    for image_path in IMAGE_DIR_PATH.glob("*.*"):
        if "identity" not in image_path.name:
            print(f"Processing image: {image_path}")
            predict_single_image(image_path, CLASS_NAMES)
