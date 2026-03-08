import os
import numpy as np
import torch
import gradio as gr

from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = "."
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INDEX_DIR = os.path.join(PROJECT_DIR, "index")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading saved image embeddings...")
image_embeddings = np.load(os.path.join(INDEX_DIR, "image_embeddings.npy"))
labels = np.load(os.path.join(INDEX_DIR, "labels.npy"))

print("Loading CIFAR-10 dataset...")
dataset = CIFAR10(root=DATA_DIR, train=False, download=True)
class_names = dataset.classes

def search_images(query, top_k):
    top_k = int(top_k)

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

    gallery = []
    ranking_lines = []

    for rank, idx in enumerate(top_idx, start=1):
        img, _ = dataset[idx]
        score = sims[idx]
        label_name = class_names[labels[idx]]
        caption = f"{rank}. {label_name} | score={score:.4f}"
        gallery.append((img, caption))
        ranking_lines.append(caption)

    return gallery, "\n".join(ranking_lines)

example_queries = [
    ["a green animal", 5],
    ["something that flies", 5],
    ["a fast car", 5],
    ["a vehicle", 5],
]

with gr.Blocks() as demo:
    gr.Markdown("# CLIP Image-Text Semantic Search")
    gr.Markdown(
        "Type a natural-language query and retrieve the most semantically similar CIFAR-10 images."
    )

    with gr.Row():
        query = gr.Textbox(
            label="Text Query",
            placeholder="e.g. a green animal"
        )
        top_k = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of results"
        )

    search_btn = gr.Button("Search")

    gallery = gr.Gallery(
        label="Retrieved Images",
        columns=5,
        height="auto"
    )
    rankings = gr.Textbox(label="Ranked Results", lines=8)

    search_btn.click(
        fn=search_images,
        inputs=[query, top_k],
        outputs=[gallery, rankings]
    )

    gr.Examples(
        examples=example_queries,
        inputs=[query, top_k]
    )

demo.launch()
