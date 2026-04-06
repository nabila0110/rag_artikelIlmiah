import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns


embeddings_before = np.load('data/bn_embeddings.npy')
embeddings_after = np.load('data/embeddings.npy')
chunks_df = pd.read_csv('data/data_chunk.csv')


# Pilih 1 vektor untuk visualisasi detail
VECTOR_IDX = 0
vector_before = embeddings_before[VECTOR_IDX]
vector_after = embeddings_after[VECTOR_IDX]

print(f"\n Visualisasi vektor index: {VECTOR_IDX}")
print(f" Chunk text: {chunks_df.iloc[VECTOR_IDX]['chunk_text'][:100]}...")

# Statistik
l2_before = np.linalg.norm(vector_before)
l2_after = np.linalg.norm(vector_after)

print(f"\n Statistik vektor sebelum normalisasi:")
print(f"   - Min: {vector_before.min():.6f}")
print(f"   - Max: {vector_before.max():.6f}")
print(f"   - Mean: {vector_before.mean():.6f}")
print(f"   - Std: {vector_before.std():.6f}")
print(f"   - L2 norm: {l2_before:.6f}")

print(f"\n Statistik vektor setelah normalisasi:")
print(f"   - Min: {vector_after.min():.6f}")
print(f"   - Max: {vector_after.max():.6f}")
print(f"   - Mean: {vector_after.mean():.6f}")
print(f"   - Std: {vector_after.std():.6f}")
print(f"   - L2 norm: {l2_after:.6f}")

# ============================================================================
# VISUALISASI 3: DISTRIBUSI NILAI (HISTOGRAM)
# ============================================================================
print("\n Membuat visualisasi 3: Histogram distribusi nilai...")

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
# PRINT NILAI LENGKAP (OPSIONAL - KE FILE TXT)
# ============================================================================
print("\n Menyimpan nilai lengkap ke file teks...")

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

print("    Saved: visualizations/embedding_values.txt")

# ============================================================================
# SELESAI
# ============================================================================
print("\n" + "="*80)
print(" VISUALISASI SELESAI!")
print("="*80)
# Show plot
print("\n Membuka plot...")
plt.show()
