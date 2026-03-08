import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = "/content/clip_image_text_search"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load pretrained CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

print(f"Using device: {device}")
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Load CIFAR-10
print("Loading CIFAR-10...")
dataset = CIFAR10(root=DATA_DIR, train=False, download=True)

class_names = dataset.classes
print("CIFAR-10 classes:", class_names)

# Take a small subset for fast retrieval
num_images = 200
images = []
labels = []

for i in range(num_images):
    image, label = dataset[i]
    images.append(image)
    labels.append(label)

print(f"Loaded {len(images)} images.")

# Build image embeddings
print("Building image embeddings...")
image_embeddings = []

batch_size = 32
for start in range(0, len(images), batch_size):
    batch_images = images[start:start + batch_size]

    inputs = processor(
        images=batch_images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        features = model.visual_projection(pooled_output)

    features = features / features.norm(dim=-1, keepdim=True)
    image_embeddings.append(features.detach().cpu().numpy())

image_embeddings = np.vstack(image_embeddings)
print("Image embeddings shape:", image_embeddings.shape)

def search_images(text_query, top_k=5):
    inputs = processor(
        text=[text_query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_embedding = text_features.detach().cpu().numpy()

    sims = cosine_similarity(text_embedding, image_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    return top_indices, sims[top_indices]

def show_results(query, top_k=5):
    top_indices, top_scores = search_images(query, top_k=top_k)

    print(f"\nTop {top_k} results for query: {query}")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        true_label = class_names[labels[idx]]
        print(f"{rank}. index={idx}, label={true_label}, score={score:.4f}")

    plt.figure(figsize=(15, 3))
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        plt.subplot(1, top_k, i + 1)
        plt.imshow(images[idx])
        plt.axis("off")
        true_label = class_names[labels[idx]]
        plt.title(f"{true_label}\nscore={score:.3f}")

    plt.suptitle(f"Query: {query}", fontsize=14)
    plt.tight_layout()

    safe_query = query.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(RESULTS_DIR, f"{safe_query}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved result to: {out_path}")

# Starter demo queries
demo_queries = [
    "a frog",
    "a truck",
    "a ship",
    "a dog"
]

for q in demo_queries:
    print("=" * 60)
    show_results(q, top_k=5)

# Interactive loop
while True:
    query = input("\nEnter a text query (or type 'exit' to stop): ").strip()

    if query.lower() == "exit":
        print("Exiting search.")
        break

    if query == "":
        print("Please enter a non-empty query.")
        continue

    print("=" * 60)
    show_results(query, top_k=5)
