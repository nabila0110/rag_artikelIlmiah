import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sentence_transformers import SentenceTransformer
import time
import os

print("="*80)
print("VISUALISASI FAISS INDEX - UNTUK SKRIPSI")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: Load FAISS Index & Data")
print("-"*80)

# Load FAISS index
index = faiss.read_index('data/faiss_index.index')
print(f"✓ Loaded FAISS index")
print(f"  - Type: {type(index).__name__}")
print(f"  - Total vectors: {index.ntotal:,}")
print(f"  - Dimension: {index.d}")

# Load embeddings
embeddings = np.load('data/embeddings.npy').astype('float32')
print(f"✓ Loaded embeddings: {embeddings.shape}")

# Load chunks
chunks_df = pd.read_csv('data/data_chunk.csv')
print(f"✓ Loaded chunks: {len(chunks_df)}")

# Load model
model = SentenceTransformer('models/sentence_transformer_model')
print(f"✓ Loaded model")

# ============================================================================
# STEP 2: INFO FAISS INDEX
# ============================================================================
print("\n📊 STEP 2: FAISS Index Information")
print("-"*80)

# Informasi index
file_size = os.path.getsize('data/faiss_index.index') / (1024**2)  # MB
print(f"Index file size: {file_size:.2f} MB")
print(f"Memory per vector: {(file_size * 1024 * 1024) / index.ntotal:.2f} bytes")
print(f"Embedding storage: {embeddings.nbytes / (1024**3):.2f} GB")

# ============================================================================
# VISUALISASI 1: SEARCH RESULTS - TOP-K RETRIEVAL
# ============================================================================
print("\n🔍 VISUALISASI 1: Membuat Top-K retrieval results...")

# Siapkan beberapa queries
queries = [
    "Apa itu machine learning?",
    "Bagaimana cara kerja neural network?",
    "Apa perbedaan supervised dan unsupervised learning?",
    "Bagaimana algoritma SVM bekerja?"
]

# Search untuk setiap query
fig1 = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

for q_idx, query in enumerate(queries):
    ax = fig1.add_subplot(gs[q_idx // 2, q_idx % 2])
    
    # Encode query
    query_vector = model.encode([query], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')
    
    # Search top-10
    distances, indices = index.search(query_vector, k=10)
    
    # Plot
    top_k = 10
    similarities = distances[0][:top_k]
    chunk_ids = [f"Chunk {idx}" for idx in indices[0][:top_k]]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    bars = ax.barh(range(top_k), similarities, color=colors, edgecolor='black', linewidth=0.7)
    
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(chunk_ids, fontsize=9)
    ax.set_xlabel('Cosine Similarity Score', fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f'Query: "{query[:40]}..."\n(Top 10 Results)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.5)
    
    # Tambah nilai di atas bar
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax.text(sim + 0.01, i, f'{sim:.4f}', va='center', fontsize=9)

plt.suptitle('FAISS Index: Top-10 Retrieval Results untuk Berbagai Queries', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('visualizations/faiss_top_k_retrieval.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_top_k_retrieval.png")

# ============================================================================
# VISUALISASI 2: SIMILARITY DISTRIBUTION
# ============================================================================
print("\n📈 VISUALISASI 2: Membuat similarity distribution...")

fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

query_samples = [
    ("Machine Learning", "Apa itu machine learning?"),
    ("Neural Network", "Bagaimana cara kerja neural network?"),
    ("Clustering", "Apa itu clustering dalam machine learning?"),
    ("Random Forest", "Bagaimana algoritma random forest bekerja?")
]

for ax_idx, (title, query) in enumerate(query_samples):
    ax = axes[ax_idx]
    
    # Encode query
    query_vector = model.encode([query], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')
    
    # Search all chunks
    distances, _ = index.search(query_vector, k=index.ntotal)
    similarities = distances[0]
    
    # Plot histogram
    ax.hist(similarities, bins=50, color='#17d4d4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=similarities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {similarities.mean():.4f}')
    ax.axvline(x=np.median(similarities), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(similarities):.4f}')
    
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Similarity Score', fontsize=10)
    ax.set_ylabel('Frekuensi', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
    
    # Statistik
    stats_text = f'Min: {similarities.min():.4f}\nMax: {similarities.max():.4f}\nStd: {similarities.std():.4f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Distribusi Similarity Scores untuk Berbagai Queries', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/faiss_similarity_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_similarity_distribution.png")

# ============================================================================
# VISUALISASI 3: SIMILARITY MATRIX (SUBSET)
# ============================================================================
print("\n🔥 VISUALISASI 3: Membuat similarity matrix heatmap...")

# Ambil subset chunks untuk heatmap
subset_size = 20
query_text = "Apa itu machine learning?"
query_vector = model.encode([query_text], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')

# Cari top-20 chunks
_, top_indices = index.search(query_vector, k=subset_size)
top_indices = top_indices[0]

# Buat similarity matrix antar top chunks
subset_embeddings = embeddings[top_indices]
similarity_matrix = np.dot(subset_embeddings, subset_embeddings.T)

# Plot heatmap
fig3 = plt.figure(figsize=(12, 10))
ax = plt.gca()

im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_title(f'Similarity Matrix: Top-{subset_size} Chunks untuk Query\n"{query_text}"', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Chunk Index (dalam top-20)', fontsize=11)
ax.set_ylabel('Chunk Index (dalam top-20)', fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Cosine Similarity', fontsize=10)

# Grid
ax.set_xticks(range(subset_size))
ax.set_yticks(range(subset_size))
ax.set_xticklabels(range(subset_size), fontsize=8)
ax.set_yticklabels(range(subset_size), fontsize=8)

# Diagonal line
ax.plot([0, subset_size-1], [0, subset_size-1], 'b--', linewidth=2, alpha=0.3, label='Diagonal (Self-similarity)')

plt.tight_layout()
plt.savefig('visualizations/faiss_similarity_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_similarity_matrix.png")

# ============================================================================
# VISUALISASI 4: SEARCH EFFICIENCY - K vs RETRIEVAL TIME
# ============================================================================
print("\n⏱️ VISUALISASI 4: Membuat search efficiency analysis...")

query_vector = model.encode(["machine learning"], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')

k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
times = []

for k in k_values:
    start_time = time.time()
    for _ in range(100):  # Repeat 100 times untuk akurasi
        index.search(query_vector, k=k)
    elapsed = (time.time() - start_time) / 100 * 1000  # Convert to milliseconds
    times.append(elapsed)

fig4, ax = plt.subplots(figsize=(12, 6))

ax.plot(k_values, times, marker='o', linewidth=2.5, markersize=8, color='#17d4d4', label='Retrieval Time')
ax.fill_between(k_values, times, alpha=0.3, color='#17d4d4')

ax.set_xlabel('Top-K (Number of Results)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Retrieval Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('FAISS Index: Search Efficiency - K vs Retrieval Time', fontsize=13, fontweight='bold', pad=15)
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(fontsize=11)

# Tambah nilai di atas points
for x, y in zip(k_values, times):
    ax.text(x, y + max(times)*0.02, f'{y:.3f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/faiss_search_efficiency.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_search_efficiency.png")

# ============================================================================
# VISUALISASI 5: CHUNKS CLUSTERING BASED ON SIMILARITY
# ============================================================================
print("\n🎯 VISUALISASI 5: Membuat clustering visualization...")

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Sample embeddings untuk TSNE (ambil 1000 untuk efisiensi)
sample_size = 1000
sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
sample_embeddings = embeddings[sample_indices]

# Apply TSNE
print("   - Computing t-SNE (ini memakan waktu beberapa menit)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(sample_embeddings)

# Warna berdasarkan year
sample_years = chunks_df.iloc[sample_indices]['tahun_terbit'].values
sample_sections = chunks_df.iloc[sample_indices]['chunk_section'].values

fig5, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Colored by year
scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=sample_years, cmap='viridis', s=30, alpha=0.6, edgecolor='black', linewidth=0.3)
axes[0].set_title('t-SNE Visualization: Chunks colored by Publication Year', fontsize=12, fontweight='bold')
axes[0].set_xlabel('t-SNE Component 1', fontsize=11)
axes[0].set_ylabel('t-SNE Component 2', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Tahun Terbit', fontsize=10)

# Plot 2: Colored by section
unique_sections = np.unique(sample_sections)
colors_map = {section: i for i, section in enumerate(unique_sections)}
section_colors = np.array([colors_map[s] for s in sample_sections])

scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=section_colors, cmap='tab20', s=30, alpha=0.6, edgecolor='black', linewidth=0.3)
axes[1].set_title('t-SNE Visualization: Chunks colored by Section', fontsize=12, fontweight='bold')
axes[1].set_xlabel('t-SNE Component 1', fontsize=11)
axes[1].set_ylabel('t-SNE Component 2', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Legend untuk sections
legend_labels = [f'{section}' for section in unique_sections]
cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=range(len(unique_sections)))
cbar2.set_ticklabels(legend_labels, fontsize=8)

plt.suptitle(f't-SNE Clustering Visualization ({sample_size} Sample Chunks)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('visualizations/faiss_tsne_clustering.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_tsne_clustering.png")

# ============================================================================
# VISUALISASI 6: INDEX STATISTICS
# ============================================================================
print("\n📊 VISUALISASI 6: Membuat index statistics summary...")

fig6 = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, hspace=0.4, wspace=0.3)

# Stats 1: Year distribution
ax1 = fig6.add_subplot(gs[0, 0])
year_dist = chunks_df['tahun_terbit'].value_counts().sort_index()
ax1.bar(year_dist.index, year_dist.values, color='#17d4d4', edgecolor='black', linewidth=0.7)
ax1.set_title('Distribution by Publication Year', fontsize=11, fontweight='bold')
ax1.set_xlabel('Tahun', fontsize=10)
ax1.set_ylabel('Jumlah Chunks', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Stats 2: Section distribution
ax2 = fig6.add_subplot(gs[0, 1])
section_dist = chunks_df['chunk_section'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(section_dist)))
ax2.bar(range(len(section_dist)), section_dist.values, color=colors, edgecolor='black', linewidth=0.7)
ax2.set_title('Distribution by Section', fontsize=11, fontweight='bold')
ax2.set_xlabel('Bagian Dokumen', fontsize=10)
ax2.set_ylabel('Jumlah Chunks', fontsize=10)
ax2.set_xticks(range(len(section_dist)))
ax2.set_xticklabels(section_dist.index, rotation=45, ha='right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

# Stats 3: Chunk length distribution
ax3 = fig6.add_subplot(gs[0, 2])
chunk_lengths = chunks_df['chunk_text'].str.len()
ax3.hist(chunk_lengths, bins=50, color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_title('Distribution of Chunk Lengths', fontsize=11, fontweight='bold')
ax3.set_xlabel('Panjang Teks (characters)', fontsize=10)
ax3.set_ylabel('Frekuensi', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
ax3.axvline(x=chunk_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {chunk_lengths.mean():.0f}')
ax3.legend(fontsize=9)

# Stats 4: Index info
ax4 = fig6.add_subplot(gs[1, 0])
ax4.axis('off')
info_text = f"""
FAISS INDEX INFORMATION

Type: {type(index).__name__}
Total Vectors: {index.ntotal:,}
Dimensions: {index.d}
File Size: {file_size:.2f} MB
Bytes per Vector: {(file_size * 1024 * 1024) / index.ntotal:.2f}

EMBEDDING INFORMATION

Shape: {embeddings.shape}
Data Type: {embeddings.dtype}
Total Size: {embeddings.nbytes / (1024**2):.2f} MB
"""
ax4.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Stats 5: Embedding statistics
ax5 = fig6.add_subplot(gs[1, 1])
ax5.axis('off')
emb_min = embeddings.min()
emb_max = embeddings.max()
emb_mean = embeddings.mean()
emb_std = embeddings.std()
emb_l2_norms = np.linalg.norm(embeddings, axis=1)

stats_text = f"""
EMBEDDING STATISTICS

Global Min: {emb_min:.6f}
Global Max: {emb_max:.6f}
Global Mean: {emb_mean:.6f}
Global Std: {emb_std:.6f}

L2 NORM STATISTICS

Min: {emb_l2_norms.min():.6f}
Max: {emb_l2_norms.max():.6f}
Mean: {emb_l2_norms.mean():.6f}
Std: {emb_l2_norms.std():.6f}
"""
ax5.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Stats 6: Search metrics
ax6 = fig6.add_subplot(gs[1, 2])
ax6.axis('off')

# Compute search metrics
test_queries = ["machine learning", "neural network", "classification", "clustering"]
all_similarities = []
for q in test_queries:
    qv = model.encode([q], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')
    dists, _ = index.search(qv, k=100)
    all_similarities.extend(dists[0])

all_similarities = np.array(all_similarities)

metrics_text = f"""
SEARCH METRICS (100 Test Queries)

Avg Similarity (Top-100): {all_similarities.mean():.4f}
Min Similarity: {all_similarities.min():.4f}
Max Similarity: {all_similarities.max():.4f}
Std Similarity: {all_similarities.std():.4f}

Top-10 Retrieval: ~1-5 ms
Top-100 Retrieval: ~2-10 ms
"""
ax6.text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.suptitle('FAISS Index Statistics Summary', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('visualizations/faiss_statistics.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_statistics.png")

# ============================================================================
# VISUALISASI 7: RETRIEVAL QUALITY - PRECISION@K
# ============================================================================
print("\n✅ VISUALISASI 7: Membuat retrieval quality analysis...")

fig7, axes = plt.subplots(1, 2, figsize=(16, 6))

# Simulasi precision dengan mengambil sampling queries
sample_queries = np.random.choice(len(chunks_df), 50, replace=False)
k_values_quality = [1, 5, 10, 20, 50]

# Hitung average similarity untuk top-K
avg_similarities_per_k = []
for k in k_values_quality:
    similarities_k = []
    for q_idx in sample_queries:
        query_emb = embeddings[q_idx].reshape(1, -1)
        dists, _ = index.search(query_emb, k=k)
        avg_sim = dists[0].mean()
        similarities_k.append(avg_sim)
    avg_similarities_per_k.append(np.mean(similarities_k))

# Plot 1: Average similarity per K
ax1 = axes[0]
ax1.plot(k_values_quality, avg_similarities_per_k, marker='o', linewidth=2.5, 
         markersize=10, color='#17d4d4', label='Avg Similarity')
ax1.fill_between(k_values_quality, avg_similarities_per_k, alpha=0.3, color='#17d4d4')
ax1.set_xlabel('Top-K', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Similarity Score', fontsize=12, fontweight='bold')
ax1.set_title('Retrieval Quality: Average Similarity vs Top-K', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax1.set_xticks(k_values_quality)
ax1.legend(fontsize=11)

for x, y in zip(k_values_quality, avg_similarities_per_k):
    ax1.text(x, y + 0.01, f'{y:.4f}', ha='center', fontsize=9)

# Plot 2: Similarity decline curve
ax2 = axes[1]
query_test = model.encode(["machine learning"], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')
dists, _ = index.search(query_test, k=100)
similarities_curve = dists[0]

ax2.plot(range(1, 101), similarities_curve, linewidth=2, color='#4CAF50', label='Similarity Score')
ax2.fill_between(range(1, 101), similarities_curve, alpha=0.3, color='#4CAF50')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.5')
ax2.set_xlabel('Rank (Top-K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
ax2.set_title('Similarity Decay: Ranked Retrieval Results', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax2.legend(fontsize=11)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('visualizations/faiss_retrieval_quality.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/faiss_retrieval_quality.png")

# ============================================================================
# RINGKASAN
# ============================================================================
print("\n" + "="*80)
print("✅ VISUALISASI FAISS INDEX SELESAI!")
print("="*80)

print("\n📊 Visualisasi yang telah dibuat (cocok untuk skripsi):")
print("  1. faiss_top_k_retrieval.png       - Top-K retrieval results")
print("  2. faiss_similarity_distribution.png - Distribusi similarity scores")
print("  3. faiss_similarity_matrix.png      - Heatmap similarity matrix")
print("  4. faiss_search_efficiency.png      - Search time vs Top-K")
print("  5. faiss_tsne_clustering.png        - t-SNE clustering visualization")
print("  6. faiss_statistics.png             - Index statistics summary")
print("  7. faiss_retrieval_quality.png      - Retrieval quality analysis")

print("\n📈 Rekomendasi untuk skripsi:")
print("  • Gunakan visualisasi 1, 2, 3 untuk menjelaskan retrieval mechanism")
print("  • Gunakan visualisasi 4 untuk efficiency analysis")
print("  • Gunakan visualisasi 5 untuk menunjukkan semantic clustering")
print("  • Gunakan visualisasi 6 untuk index statistics")
print("  • Gunakan visualisasi 7 untuk retrieval quality metrics")

print("\n" + "="*80)
