import os
import difflib
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = "/content/clip_image_text_search"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INDEX_DIR = os.path.join(PROJECT_DIR, "index")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading embeddings...")
image_embeddings = np.load(os.path.join(INDEX_DIR, "image_embeddings.npy"))
labels = np.load(os.path.join(INDEX_DIR, "labels.npy"))

dataset = CIFAR10(root=DATA_DIR, train=False, download=True)
class_names = dataset.classes

def is_exit_command(text):
    text = text.strip().lower()
    if text == "exit":
        return True

    close_matches = difflib.get_close_matches(text, ["exit"], n=1, cutoff=0.75)
    return len(close_matches) > 0

def search_images(query, top_k=5):
    inputs = processor(
        text=[query],
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
        pooled = text_outputs.pooler_output
        text_features = model.text_projection(pooled)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_embedding = text_features.cpu().numpy()

    sims = cosine_similarity(text_embedding, image_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    return top_idx, sims[top_idx]

def show_results(query, top_k=5):
    idx, scores = search_images(query, top_k=top_k)

    print(f"\nTop {top_k} results for query: {query}")
    for rank, (j, score) in enumerate(zip(idx, scores), start=1):
        label_name = class_names[labels[j]]
        print(f"{rank}. index={j}, label={label_name}, score={score:.4f}")

    plt.figure(figsize=(3 * top_k, 3))

    for i, (j, score) in enumerate(zip(idx, scores)):
        img, label = dataset[j]

        plt.subplot(1, top_k, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{class_names[label]}\n{score:.3f}")

    plt.suptitle(f"Query: {query}")
    plt.tight_layout()

    safe_query = query.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(RESULTS_DIR, f"{safe_query}_top{top_k}_search.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

    print(f"Saved result to: {out_path}")

while True:
    query = input("Enter query (or exit): ").strip()

    if is_exit_command(query):
        print("Exiting search.")
        break

    if query == "":
        print("Please enter a non-empty query.")
        continue

    top_k_input = input("How many results would you like to see? (default 5): ").strip()

    if top_k_input == "":
        top_k = 5
    else:
        try:
            top_k = int(top_k_input)
            if top_k < 1:
                print("top_k must be at least 1. Using 5 instead.")
                top_k = 5
            elif top_k > 10:
                print("For readability, capping top_k at 10.")
                top_k = 10
        except ValueError:
            print("Invalid number. Using default top_k=5.")
            top_k = 5

    show_results(query, top_k=top_k)
