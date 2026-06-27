from yolo_pipeline import detect_fire

prob, img = detect_fire("sample_fire.jpg")

print("YOLO Fire Confidence:", prob)
print("Output image:", img)