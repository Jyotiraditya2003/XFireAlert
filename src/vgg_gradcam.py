import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "vgg16_fire_classifier.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
IMG_SIZE = 224

def generate_vgg_gradcam(image_path, output_path="vgg_gradcam.jpg"):

    img = cv2.imread(str(image_path))
    original = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    # last conv layer of VGG16
    last_conv_layer = model.get_layer("block5_conv3")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if isinstance(predictions, list):
            predictions = predictions[0]

        # fire probability
        class_channel = 1 - predictions[:, 0]

    # gradients
    grads = tape.gradient(class_channel, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, superimposed)

    return output_path