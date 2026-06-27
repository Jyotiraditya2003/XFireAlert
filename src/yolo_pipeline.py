from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo_fire.pt"

# loading model
model = YOLO(str(MODEL_PATH))


def detect_fire(image_path, output_path="yolo_output.jpg"):
    """
    Runs YOLO fire detection and draws bounding box
    """

    results = model.predict(source=image_path, conf=0.35, save=False)

    result = results[0]
    img = cv2.imread(image_path)

    fire_conf = 0.0

    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            fire_conf = max(fire_conf, conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"FIRE {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)

    return fire_conf, output_path