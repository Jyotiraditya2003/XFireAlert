import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fire_classifier.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 224

# Fire Probability Prediction

def predict_image(image_path):

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Image not found or path incorrect")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)

    # handles keras list output
    if isinstance(pred, list):
        pred = pred[0]

    no_fire_prob = float(pred[0][0])

    fire_prob = 1 - no_fire_prob

    return calibrate_probability(fire_prob)

def calibrate_probability(p):
    return float(1 / (1 + np.exp(-4*(p-0.5))))


def classify_fire(prob):
    if prob > 0.7:
        return "FIRE"
    elif prob > 0.4:
        return "POSSIBLE_FIRE"
    else:
        return "NO_FIRE"

# Grad-CAM Heatmap
def generate_gradcam(image_path, output_path="gradcam.jpg"):

    # loading image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Image not found or path incorrect")

    original = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    # locates last conv layer safely
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in model")

    last_conv_layer = model.get_layer(last_conv_layer)

    # grad model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_channel = 1 - predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # resizing heatmap
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)

    cv2.imwrite(output_path, superimposed)

    return output_path
