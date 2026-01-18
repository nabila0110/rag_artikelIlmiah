import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformers
from pathlib import Path
import shutil
def prepare_data():
    #paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    models_dir = base_dis / 'models'

    #buat directories
    data_dir.mkdir(exis_ok=True)
    models_dir.mkdir(exis_ok=True)

    print('\n[1/5] Checking source files...')

    #Check source files
    source_chunks = 'data_chunk.csv'  
    
    if not Path(source_chunks).exists():
        print(f"Error: {source_chunks} not found!")
        print("Please ensure you have the chunks CSV file.")
        return False
    
    print(f"Found: {source_chunks}")
    
    #Copy chunks file
    print("\n[2/5] Copying chunks file...")
    shutil.copy(source_chunks, data_dir / 'data_chunk.csv')
    print("Chunks copied")
    
    #Load chunks
    print("\n[3/5] Loading chunks...")
    chunks_df = pd.read_csv(data_dir / 'data_chunk.csv')
    print(f"Loaded {len(chunks_df)} chunks")
    
    #Create embeddings
    print("\n[4/5] Creating embeddings...")
    print("This may take a few minutes...")
    
    model_name = 'intfloat/multilingual-e5-base'
    model = SentenceTransformer(model_name)
    
    embeddings = model.encode(
        chunks_df['chunk_text'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"Embeddings created: {embeddings.shape}")
    
    #Normalize untuk cosine similarity
    faiss.normalize_L2(embeddings)
    
    #Create FAISS index
    print("\n[5/5] Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"FAISS index created: {index.ntotal} vectors")
    
    #Save files
    print("\nSaving files...")
    
    #Save FAISS index
    faiss.write_index(index, str(data_dir / 'faiss_index.index'))
    print(f"Saved: {data_dir / 'faiss_index.index'}")
    
    #Save embeddings
    np.save(data_dir / 'embeddings.npy', embeddings)
    print(f"Saved: {data_dir / 'embeddings.npy'}")
    
    #Save model
    model.save(str(models_dir / 'sentence_transformer_model'))
    print(f"Saved: {models_dir / 'sentence_transformer_model'}")
    
    #Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print(f"  - {data_dir / 'data_chunk.csv'}")
    print(f"  - {data_dir / 'faiss_index.index'}")
    print(f"  - {data_dir / 'embeddings.npy'}")
    print(f"  - {models_dir / 'sentence_transformer_model'}")
    
    return True


if __name__ == '__main__':
    try:
        prepare_data()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
