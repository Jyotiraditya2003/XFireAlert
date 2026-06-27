import os
import numpy as np
from sklearn.metrics import accuracy_score
from image_pipeline import predict_image
from risk_pipeline import predict_risk

fire_dir = "data/images/ForestFireDataset/train/fire"
nofire_dir = "data/images/ForestFireDataset/train/nofire"

image_paths = []
labels = []

for img in os.listdir(fire_dir):
    image_paths.append(os.path.join(fire_dir, img))
    labels.append(1)

for img in os.listdir(nofire_dir):
    image_paths.append(os.path.join(nofire_dir, img))
    labels.append(0)

fusion_preds = []

for img_path in image_paths:

    p_det = predict_image(img_path)

    weather = {
        "Temperature": np.random.uniform(25,45),
        "RH": np.random.uniform(20,80),
        "Ws": np.random.uniform(5,20),
        "Rain": 0,
        "FFMC": np.random.uniform(70,95),
        "DMC": np.random.uniform(10,80),
        "DC": np.random.uniform(50,200),
        "ISI": np.random.uniform(2,15),
        "BUI": np.random.uniform(20,80)
    }

    p_risk = predict_risk(weather)

    p_final = 0.4*p_risk + 0.6*p_det

    fusion_preds.append(1 if p_final > 0.5 else 0)

print("Fusion Accuracy:", accuracy_score(labels, fusion_preds))