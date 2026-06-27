import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# project path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# import pipelines
from image_pipeline import predict_image
from vgg_pipeline import predict_vgg
from fusion_engine import fuse_predictions

# dataset path
DATASET_PATH = BASE_DIR / "data" / "images" / "ForestFireDataset" / "train"

fire_path = DATASET_PATH / "fire"
nofire_path = DATASET_PATH / "nofire"

print("Fire folder:", fire_path)
print("NoFire folder:", nofire_path)
print("Fire exists:", fire_path.exists())
print("NoFire exists:", nofire_path.exists())

images = []
labels = []

# load dataset
for img in fire_path.glob("*"):
    images.append(str(img))
    labels.append(1)

for img in nofire_path.glob("*"):
    images.append(str(img))
    labels.append(0)

print("Total test images:", len(images))

# store predictions
mobilenet_preds = []
vgg_preds = []
fusion_preds = []

for img, label in zip(images, labels):

    m_prob = predict_image(img)
    v_prob = predict_vgg(img)

    m_class = 1 if m_prob > 0.5 else 0
    v_class = 1 if v_prob > 0.5 else 0

    # simulate weather risk for fusion evaluation
    weather_prob = np.random.uniform(0.3,0.9)

    fusion_prob = fuse_predictions(weather_prob, m_prob)
    fusion_class = 1 if fusion_prob > 0.5 else 0

    mobilenet_preds.append(m_class)
    vgg_preds.append(v_class)
    fusion_preds.append(fusion_class)

y_true = np.array(labels)

# ---------------- MobileNet results ----------------

print("\nMobileNetV2 Results")
print(classification_report(y_true, mobilenet_preds))

cm = confusion_matrix(y_true, mobilenet_preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("MobileNetV2 Confusion Matrix")
plt.savefig("mobilenet_confusion.png")

# ---------------- VGG results ----------------

print("\nVGG16 Results")
print(classification_report(y_true, vgg_preds))

cm = confusion_matrix(y_true, vgg_preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("VGG16 Confusion Matrix")
plt.savefig("vgg_confusion.png")

# ---------------- Fusion results ----------------

fusion_accuracy = accuracy_score(y_true, fusion_preds)

print("\nFusion Accuracy:", fusion_accuracy)

# ---------------- Comparison table ----------------

mobilenet_acc = accuracy_score(y_true, mobilenet_preds)
vgg_acc = accuracy_score(y_true, vgg_preds)

results = pd.DataFrame({
    "Model": ["MobileNetV2", "VGG16", "Hybrid Fusion"],
    "Accuracy": [mobilenet_acc, vgg_acc, fusion_accuracy]
})

print("\nModel Comparison")
print(results)

results.to_csv("model_comparison.csv", index=False)