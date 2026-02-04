import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
from sentence_transformers import SentenceTransformer


OUTPUT_DIR = "visualizations"
FAISS_INDEX_PATH = "data/faiss_index.index"
EMBEDDINGS_PATH = "data/embeddings.npy"
CHUNKS_PATH = "data/data_chunk.csv"
MODEL_PATH = "models/sentence_transformer_model"
SAMPLE_QUERY = "Apa itu machine learning?"
TOP_K = 10


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    chunks_df = pd.read_csv(CHUNKS_PATH)
    model = SentenceTransformer(MODEL_PATH)
    return index, embeddings, chunks_df, model


def draw_box(ax, xy, text, box_color="#0f1419", text_color="#d4d8dd"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), 0.24, 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=box_color,
        edgecolor="#17d4d4",
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + 0.12, y + 0.06, text, ha="center", va="center", fontsize=9, color=text_color)


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, color="#17d4d4", linewidth=1.4)
    ax.add_patch(arrow)


def visualize_faiss_search_process():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.02, 0.4), "Query")
    draw_box(ax, (0.30, 0.4), "Encode\nEmbedding")
    draw_box(ax, (0.58, 0.4), "FAISS\nSearch")
    draw_box(ax, (0.82, 0.4), "Top-K\nResults")

    draw_arrow(ax, (0.26, 0.46), (0.30, 0.46))
    draw_arrow(ax, (0.54, 0.46), (0.58, 0.46))
    draw_arrow(ax, (0.78, 0.46), (0.82, 0.46))

    ax.set_title("Visualisasi Proses Pencarian FAISS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faiss_search_process.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_system_flowchart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.75), "User\nQuery")
    draw_box(ax, (0.35, 0.75), "Flask API\n/app.py")
    draw_box(ax, (0.65, 0.75), "Retrieval\nSystem")
    draw_box(ax, (0.05, 0.45), "Sentence\nTransformer")
    draw_box(ax, (0.35, 0.45), "FAISS\nIndex")
    draw_box(ax, (0.65, 0.45), "Chunks\nCSV")
    draw_box(ax, (0.35, 0.15), "Generation\n(Ollama)")
    draw_box(ax, (0.65, 0.15), "Answer\n+ Citations")

    draw_arrow(ax, (0.29, 0.81), (0.35, 0.81))
    draw_arrow(ax, (0.59, 0.81), (0.65, 0.81))
    draw_arrow(ax, (0.47, 0.70), (0.17, 0.57))
    draw_arrow(ax, (0.47, 0.70), (0.47, 0.57))
    draw_arrow(ax, (0.47, 0.70), (0.77, 0.57))
    draw_arrow(ax, (0.47, 0.39), (0.47, 0.27))
    draw_arrow(ax, (0.59, 0.21), (0.65, 0.21))

    ax.set_title("Arsitektur Sistem RAG (Flowchart)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "system_flowchart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_faiss_metadata(index, embeddings):
    file_size_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024**2)
    bytes_per_vector = (file_size_mb * 1024 * 1024) / index.ntotal

    metadata = [
        ["Index Type", type(index).__name__],
        ["Total Vectors", f"{index.ntotal:,}"],
        ["Dimension", f"{index.d}"],
        ["Index File Size", f"{file_size_mb:.2f} MB"],
        ["Bytes / Vector", f"{bytes_per_vector:.2f} bytes"],
        ["Embeddings Shape", f"{embeddings.shape}"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=metadata,
        colLabels=["Metadata", "Value"],
        loc="center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Tabel Metadata FAISS Index", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faiss_metadata_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_similarity_distribution(index, model):
    query_vector = model.encode([SAMPLE_QUERY], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
    distances, _ = index.search(query_vector, k=index.ntotal)
    similarities = distances[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(similarities, bins=60, color="#17d4d4", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=similarities.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {similarities.mean():.4f}")
    ax.axvline(x=np.median(similarities), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(similarities):.4f}")
    ax.set_title(f"Distribusi Similarity Scores\nQuery: '{SAMPLE_QUERY}'", fontsize=12, fontweight="bold")
    ax.set_xlabel("Similarity Score", fontsize=10)
    ax.set_ylabel("Frekuensi", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faiss_similarity_distribution_single_query.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_top_k_results(index, model, chunks_df):
    query_vector = model.encode([SAMPLE_QUERY], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vector, k=TOP_K)

    similarities = distances[0]
    labels = [f"{i+1}: {chunks_df.iloc[idx]['judul'][:25]}" for i, idx in enumerate(indices[0])]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, TOP_K))
    ax.barh(range(TOP_K), similarities, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity Score", fontsize=10)
    ax.set_title(f"Top-{TOP_K} Results для Query: '{SAMPLE_QUERY}'", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faiss_top_k_results_single_query.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dir()
    index, embeddings, chunks_df, model = load_resources()

    visualize_faiss_search_process()
    visualize_system_flowchart()
    visualize_faiss_metadata(index, embeddings)
    visualize_similarity_distribution(index, model)
    visualize_top_k_results(index, model, chunks_df)

    print("\n✅ Visualisasi selesai! File tersimpan di folder: visualizations/")
    print("  1. faiss_search_process.png")
    print("  2. system_flowchart.png")
    print("  3. faiss_metadata_table.png")
    print("  4. faiss_similarity_distribution_single_query.png")
    print("  5. faiss_top_k_results_single_query.png")


if __name__ == "__main__":
    main()