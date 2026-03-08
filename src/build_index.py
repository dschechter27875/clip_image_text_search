import os
import numpy as np
import torch

from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPProcessor

PROJECT_DIR = "/content/clip_image_text_search"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INDEX_DIR = os.path.join(PROJECT_DIR, "index")

os.makedirs(INDEX_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading CIFAR10 dataset...")
dataset = CIFAR10(root=DATA_DIR, train=False, download=True)

class_names = dataset.classes

num_images = 200
images = []
labels = []

for i in range(num_images):
    image, label = dataset[i]
    images.append(image)
    labels.append(label)

print("Computing image embeddings...")

embeddings = []
batch_size = 32

for start in range(0, len(images), batch_size):

    batch = images[start:start+batch_size]

    inputs = processor(images=batch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        features = model.visual_projection(pooled)

    features = features / features.norm(dim=-1, keepdim=True)

    embeddings.append(features.cpu().numpy())

embeddings = np.vstack(embeddings)

print("Embedding matrix shape:", embeddings.shape)

np.save(os.path.join(INDEX_DIR, "image_embeddings.npy"), embeddings)
np.save(os.path.join(INDEX_DIR, "labels.npy"), np.array(labels))

print("Index saved.")
