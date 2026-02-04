import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

print("="*80)
print("VISUALISASI EMBEDDING - SEBELUM NORMALISASI")
print("="*80)

# Load data
print("\n📂 Loading data...")
embeddings_before = np.load('data/bn_embeddings.npy')
embeddings_after = np.load('data/embeddings.npy')
chunks_df = pd.read_csv('data/data_chunk.csv')

print(f"✓ Loaded {embeddings_before.shape[0]} embeddings")
print(f"✓ Dimensions: {embeddings_before.shape[1]}")

# Pilih 1 vektor untuk visualisasi detail
VECTOR_IDX = 0
vector_before = embeddings_before[VECTOR_IDX]
vector_after = embeddings_after[VECTOR_IDX]

print(f"\n📌 Visualisasi vektor index: {VECTOR_IDX}")
print(f"📌 Chunk text: {chunks_df.iloc[VECTOR_IDX]['chunk_text'][:100]}...")

# Statistik
l2_before = np.linalg.norm(vector_before)
l2_after = np.linalg.norm(vector_after)

print(f"\n📊 Statistik vektor sebelum normalisasi:")
print(f"   - Min: {vector_before.min():.6f}")
print(f"   - Max: {vector_before.max():.6f}")
print(f"   - Mean: {vector_before.mean():.6f}")
print(f"   - Std: {vector_before.std():.6f}")
print(f"   - L2 norm: {l2_before:.6f}")

print(f"\n📊 Statistik vektor setelah normalisasi:")
print(f"   - Min: {vector_after.min():.6f}")
print(f"   - Max: {vector_after.max():.6f}")
print(f"   - Mean: {vector_after.mean():.6f}")
print(f"   - Std: {vector_after.std():.6f}")
print(f"   - L2 norm: {l2_after:.6f}")

# ============================================================================
# VISUALISASI 1: LINE PLOT - NILAI PER DIMENSI (LENGKAP 768)
# ============================================================================
print("\n📈 Membuat visualisasi 1: Line plot (768 dimensi lengkap)...")

fig1 = plt.figure(figsize=(20, 6))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# Plot 1: Sebelum normalisasi
ax1 = fig1.add_subplot(gs[0])
ax1.plot(range(768), vector_before, linewidth=0.8, color='#17d4d4', alpha=0.8)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.set_title(f'Embedding Vector SEBELUM Normalisasi (L2 norm = {l2_before:.4f})', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Dimensi ke-i', fontsize=11)
ax1.set_ylabel('Nilai', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax1.set_xlim(0, 767)

# Plot 2: Setelah normalisasi
ax2 = fig1.add_subplot(gs[1])
ax2.plot(range(768), vector_after, linewidth=0.8, color='#4CAF50', alpha=0.8)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.set_title(f'Embedding Vector SETELAH Normalisasi (L2 norm = {l2_after:.4f})', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Dimensi ke-i', fontsize=11)
ax2.set_ylabel('Nilai', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax2.set_xlim(0, 767)

plt.tight_layout()
plt.savefig('visualizations/embedding_lineplot.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_lineplot.png")

# ============================================================================
# VISUALISASI 2: HEATMAP - FULL 768 DIMENSI
# ============================================================================
print("\n🔥 Membuat visualisasi 2: Heatmap (768 dimensi lengkap)...")

fig2, axes = plt.subplots(2, 1, figsize=(24, 4))

# Heatmap 1: Sebelum normalisasi
im1 = axes[0].imshow(vector_before.reshape(1, -1), aspect='auto', cmap='coolwarm', 
                      interpolation='nearest', vmin=vector_before.min(), vmax=vector_before.max())
axes[0].set_title(f'Heatmap SEBELUM Normalisasi (1x768) - L2 norm = {l2_before:.4f}', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dimensi', fontsize=11)
axes[0].set_yticks([])
axes[0].set_ylabel('Vector', fontsize=11)
cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1, aspect=50)
cbar1.set_label('Nilai', fontsize=10)

# Heatmap 2: Setelah normalisasi
im2 = axes[1].imshow(vector_after.reshape(1, -1), aspect='auto', cmap='coolwarm', 
                      interpolation='nearest', vmin=vector_after.min(), vmax=vector_after.max())
axes[1].set_title(f'Heatmap SETELAH Normalisasi (1x768) - L2 norm = {l2_after:.4f}', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dimensi', fontsize=11)
axes[1].set_yticks([])
axes[1].set_ylabel('Vector', fontsize=11)
cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1, aspect=50)
cbar2.set_label('Nilai', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/embedding_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_heatmap.png")

# ============================================================================
# VISUALISASI 3: DISTRIBUSI NILAI (HISTOGRAM)
# ============================================================================
print("\n📊 Membuat visualisasi 3: Histogram distribusi nilai...")

fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram 1: Sebelum normalisasi
axes[0].hist(vector_before, bins=50, color='#17d4d4', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].axvline(x=vector_before.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {vector_before.mean():.4f}')
axes[0].axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
axes[0].set_title(f'Distribusi Nilai SEBELUM Normalisasi\nL2 norm = {l2_before:.4f}', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Nilai', fontsize=11)
axes[0].set_ylabel('Frekuensi', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[0].text(0.02, 0.98, f'Min: {vector_before.min():.4f}\nMax: {vector_before.max():.4f}\nStd: {vector_before.std():.4f}',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Histogram 2: Setelah normalisasi
axes[1].hist(vector_after, bins=50, color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].axvline(x=vector_after.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {vector_after.mean():.4f}')
axes[1].axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
axes[1].set_title(f'Distribusi Nilai SETELAH Normalisasi\nL2 norm = {l2_after:.4f}', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Nilai', fontsize=11)
axes[1].set_ylabel('Frekuensi', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
axes[1].text(0.02, 0.98, f'Min: {vector_after.min():.4f}\nMax: {vector_after.max():.4f}\nStd: {vector_after.std():.4f}',
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/embedding_histogram.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_histogram.png")

# ============================================================================
# VISUALISASI 4: PERBANDINGAN LANGSUNG (OVERLAY)
# ============================================================================
print("\n🔀 Membuat visualisasi 4: Perbandingan overlay...")

fig4 = plt.figure(figsize=(20, 8))

# Plot overlay
plt.plot(range(768), vector_before, linewidth=1, color='#17d4d4', alpha=0.7, label=f'Sebelum Norm (L2={l2_before:.4f})')
plt.plot(range(768), vector_after, linewidth=1, color='#4CAF50', alpha=0.7, label=f'Setelah Norm (L2={l2_after:.4f})')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.title('Perbandingan Embedding: Sebelum vs Setelah Normalisasi (768 Dimensi Lengkap)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Dimensi ke-i', fontsize=12)
plt.ylabel('Nilai', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
plt.xlim(0, 767)

plt.tight_layout()
plt.savefig('visualizations/embedding_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_comparison.png")

# ============================================================================
# VISUALISASI 5: BOXPLOT STATISTIK
# ============================================================================
print("\n📦 Membuat visualisasi 5: Boxplot statistik...")

fig5, ax = plt.subplots(figsize=(10, 6))

data_to_plot = [vector_before, vector_after]
labels = [f'Sebelum Norm\n(L2={l2_before:.4f})', f'Setelah Norm\n(L2={l2_after:.4f})']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6,
                boxprops=dict(facecolor='#17d4d4', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Warna berbeda untuk setiap box
bp['boxes'][0].set_facecolor('#17d4d4')
bp['boxes'][1].set_facecolor('#4CAF50')

ax.set_title('Perbandingan Statistik: Sebelum vs Setelah Normalisasi', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Nilai', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('visualizations/embedding_boxplot.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_boxplot.png")

# ============================================================================
# VISUALISASI 6: DETAIL SEGMEN (Zoom in beberapa bagian)
# ============================================================================
print("\n🔍 Membuat visualisasi 6: Detail segmen (zoom)...")

fig6 = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 2, hspace=0.4, wspace=0.3)

segments = [
    (0, 128, "Dimensi 0-128 (Awal)"),
    (128, 256, "Dimensi 128-256"),
    (256, 384, "Dimensi 256-384"),
    (384, 512, "Dimensi 384-512"),
    (512, 640, "Dimensi 512-640"),
    (640, 768, "Dimensi 640-768 (Akhir)")
]

for idx, (start, end, title) in enumerate(segments):
    ax = fig6.add_subplot(gs[idx // 2, idx % 2])
    
    x_range = range(start, end)
    ax.plot(x_range, vector_before[start:end], linewidth=1.5, color='#17d4d4', 
            alpha=0.8, label='Sebelum Norm')
    ax.plot(x_range, vector_after[start:end], linewidth=1.5, color='#4CAF50', 
            alpha=0.8, label='Setelah Norm')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimensi', fontsize=10)
    ax.set_ylabel('Nilai', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

plt.suptitle('Detail Embedding per Segmen (768 Dimensi Terbagi 6 Bagian)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/embedding_segments.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/embedding_segments.png")

# ============================================================================
# PRINT NILAI LENGKAP (OPSIONAL - KE FILE TXT)
# ============================================================================
print("\n📝 Menyimpan nilai lengkap ke file teks...")

with open('visualizations/embedding_values.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("NILAI EMBEDDING LENGKAP (768 DIMENSI)\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Chunk Index: {VECTOR_IDX}\n")
    f.write(f"Chunk Text: {chunks_df.iloc[VECTOR_IDX]['chunk_text']}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("SEBELUM NORMALISASI:\n")
    f.write("-"*80 + "\n")
    f.write(f"L2 Norm: {l2_before:.10f}\n")
    f.write(f"Min: {vector_before.min():.10f}\n")
    f.write(f"Max: {vector_before.max():.10f}\n")
    f.write(f"Mean: {vector_before.mean():.10f}\n")
    f.write(f"Std: {vector_before.std():.10f}\n\n")
    
    f.write("Nilai per dimensi (768 dimensi lengkap):\n")
    for i in range(768):
        f.write(f"v[{i:3d}] = {vector_before[i]:12.8f}\n")
    
    f.write("\n\n" + "-"*80 + "\n")
    f.write("SETELAH NORMALISASI:\n")
    f.write("-"*80 + "\n")
    f.write(f"L2 Norm: {l2_after:.10f}\n")
    f.write(f"Min: {vector_after.min():.10f}\n")
    f.write(f"Max: {vector_after.max():.10f}\n")
    f.write(f"Mean: {vector_after.mean():.10f}\n")
    f.write(f"Std: {vector_after.std():.10f}\n\n")
    
    f.write("Nilai per dimensi (768 dimensi lengkap):\n")
    for i in range(768):
        f.write(f"v[{i:3d}] = {vector_after[i]:12.8f}\n")

print("   ✓ Saved: visualizations/embedding_values.txt")

# ============================================================================
# SELESAI
# ============================================================================
print("\n" + "="*80)
print("✅ VISUALISASI SELESAI!")
print("="*80)
print("\nFile yang dihasilkan:")
print("  1. visualizations/embedding_lineplot.png     - Line plot 768 dimensi")
print("  2. visualizations/embedding_heatmap.png      - Heatmap 1x768")
print("  3. visualizations/embedding_histogram.png    - Distribusi nilai")
print("  4. visualizations/embedding_comparison.png   - Overlay perbandingan")
print("  5. visualizations/embedding_boxplot.png      - Boxplot statistik")
print("  6. visualizations/embedding_segments.png     - Detail per segmen")
print("  7. visualizations/embedding_values.txt       - Nilai lengkap 768 dimensi")
print("\n" + "="*80)

# Show plot
print("\n💡 Membuka plot...")
plt.show()
