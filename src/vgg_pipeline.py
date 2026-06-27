import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "vgg16_fire_classifier.h5"

# loading model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 224


# Fire Prediction using VGG16


def calibrate_probability(p):

    return float(1 / (1 + np.exp(-3*(p-0.5))))

def predict_vgg(image_path):

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)

    if isinstance(pred, list):
        pred = pred[0]

    no_fire_prob = float(pred[0][0])

    # converting to fire probability
    fire_prob = 1 - no_fire_prob

    return calibrate_probability(fire_prob)