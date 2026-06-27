from vgg_pipeline import predict_vgg

image = "sample_fire.jpg"

prob = predict_vgg(image)

print("VGG16 Fire Probability:", round(prob,3))