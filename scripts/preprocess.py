import json
import os
from PIL import Image
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess(dataset_path):
    with open(os.path.join(dataset_path, "annotations.json")) as f:
        data = json.load(f)

    image_inputs = []
    texts = []

    for item in data:
        image = Image.open(item["image"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        image_inputs.append(inputs)
        texts.append(item["text"])

    return image_inputs, texts