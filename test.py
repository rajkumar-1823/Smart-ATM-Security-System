from ultralytics import YOLO
# Load a model
model = YOLO("mask.pt")  # load an official model


# Predict with the model
source = "https://thumbs.dreamstime.com/b/thief-balaclava-his-head-bank-robbery-thief-balaclava-gun-his-head-bank-robbery-ai-313453151.jpg"  # predict on an image

# Run inference on the source
results = model(source, save=True, show=True) 