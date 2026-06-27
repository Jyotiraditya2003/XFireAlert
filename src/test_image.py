from image_pipeline import predict_image, generate_gradcam

image_path = "sample_fire.jpg"  # put any fire image in project root

prob = predict_image(image_path)
print("Fire Detection Probability:", round(prob,3))

heatmap = generate_gradcam(image_path)
print("Heatmap saved at:", heatmap)
