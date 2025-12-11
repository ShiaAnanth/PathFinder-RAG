import json
import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np

# Load program overviews
with open("program_overviews.json", "r", encoding="utf-8") as f:
    programs = json.load(f)

# Convert dict → list of entries we can rank
program_entries = []
for prog, desc in programs.items():
    program_entries.append({
        "program": prog,
        "description": desc
    })

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Build search function
def find_best_major(question):
    q_emb = embedder.encode([question])[0]

    best_match = None
    best_score = -1

    for entry in program_entries:
        text = entry["program"] + " " + entry["description"]
        emb = embedder.encode([text])[0]

        score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))

        if score > best_score:
            best_score = score
            best_match = entry["program"]

    return f"**Recommended Major:** {best_match}"

# Gradio UI
iface = gr.Interface(
    fn=find_best_major,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Markdown(label="Result"),
    title="Pathfinder Demo",
    description="Small demo showing simple embedding retrieval over program descriptions."
)

iface.launch()
