"""
Wheat Variety Classification - Inference Script
Usage: python inference.py <path_to_image>
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

MODEL_PATH = "../models/best_model.keras"
CLASS_NAMES = ['Akbar', 'Dilkash', 'urooj']  # must match training order (alphabetical)
IMG_SIZE = (256, 256)


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype(np.float32)
    img_preprocessed = preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_class_id = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_id]
    confidence = predictions[predicted_class_id] * 100

    return predicted_class, confidence, predictions


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading model...")
    model = load_model()

    print(f"Predicting on: {image_path}")
    predicted_class, confidence, all_probs = predict_image(model, image_path)

    print("\n" + "=" * 40)
    print(f"Predicted Wheat Variety: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 40)

    print("\nAll class probabilities:")
    for cls, prob in zip(CLASS_NAMES, all_probs):
        print(f"  {cls}: {prob*100:.2f}%")


if __name__ == "__main__":
    main()