import torch
import librosa
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from muq import MuQ

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
# Force CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️ No GPU detected, using CPU (this will be slow)")

sr = 24000  # MuQ uses 24kHz
DATA_DIR = 'data/fma_small'
METADATA_DIR = 'data/fma_metadata'

print(f"Using device: {device}")

# -----------------------------
# 2️⃣ Load MuQ model
# -----------------------------
print("\nLoading MuQ model...")
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

# Enable memory efficient inference if on GPU
if device == 'cuda':
    muq = muq.half()  # Use FP16 for faster inference on GPU
    print("✅ MuQ model loaded successfully with FP16 precision!")
else:
    print("✅ MuQ model loaded successfully!")
    
# Clear GPU cache
if device == 'cuda':
    torch.cuda.empty_cache()

# -----------------------------
# 3️⃣ Load FMA metadata
# -----------------------------
print("\nLoading FMA metadata...")
metadata = pd.read_csv(f'{METADATA_DIR}/tracks.csv', index_col=0, header=[0,1])

# Keep only tracks that belong to fma_small
subset = metadata['set']['subset']
small_track_ids = subset[subset == 'small'].index

# Keep main genre and drop missing
tracks = metadata['track']['genre_top'].dropna()
tracks = tracks[tracks.index.isin(small_track_ids)]

# -----------------------------
# 4️⃣ Map track IDs to file paths
# -----------------------------
def get_track_path(track_id, base_dir=DATA_DIR):
    """Convert track ID to file path"""
    folder = f"{track_id:06d}"[:3]  # first 3 digits
    filename = f"{track_id:06d}.mp3"
    return os.path.join(base_dir, folder, filename)

df = pd.DataFrame({'track_id': tracks.index, 'genre': tracks.values})
df['file_path'] = df['track_id'].apply(get_track_path)

# Filter only existing files
df = df[df['file_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Found {len(df)} valid audio files")
print(f"Genres: {df['genre'].unique()}")

# -----------------------------
# 5️⃣ Extract MuQ embeddings
# -----------------------------
def extract_muq_embedding(file_path):
    """Extract embedding using MuQ model on GPU"""
    try:
        # Load audio at 24kHz (MuQ's sampling rate)
        wav, _ = librosa.load(file_path, sr=sr, duration=30.0, mono=True)
        
        # Convert to torch tensor and move to GPU
        wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Convert to FP16 if model is in FP16
        if device == 'cuda':
            wavs = wavs.half()
        
        # Extract embedding using GPU
        with torch.no_grad():
            output = muq(wavs)
        
        # Average pooling over time dimension (keep on GPU until final conversion)
        emb = output.last_hidden_state.squeeze(0).mean(dim=0)
        
        # Move back to CPU only for storage
        return emb.float().cpu().numpy()
    
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Extract embeddings for all tracks
print("\nExtracting MuQ embeddings...")
embeddings = []
genres = []
track_ids = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tracks"):
    emb = extract_muq_embedding(row['file_path'])
    
    if emb is None:
        continue
    
    embeddings.append(emb)
    genres.append(row['genre'])
    track_ids.append(row['track_id'])
    
    # Log every 100 tracks
    if len(embeddings) % 100 == 0:
        print(f"✅ Processed {len(embeddings)} tracks | Current: {row['track_id']} | Genre: {row['genre']}")
        # Clear GPU cache periodically
        if device == 'cuda':
            torch.cuda.empty_cache()

# Convert to arrays
embeddings = np.array(embeddings)
genres = np.array(genres)
track_ids = np.array(track_ids)

print(f"\n✅ Extraction complete!")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Number of tracks: {len(genres)}")

# -----------------------------
# 6️⃣ Save embeddings
# -----------------------------
print("\nSaving embeddings...")

# Save as CSV
df_embeddings = pd.DataFrame(embeddings)
df_embeddings['genre'] = genres
df_embeddings['track_id'] = track_ids
df_embeddings.to_csv("fma_muq_embeddings.csv", index=False)

# Save as PyTorch format
torch.save({
    'embeddings': torch.tensor(embeddings),
    'genres': genres,
    'track_ids': track_ids
}, "fma_muq_embeddings.pt")

print("✅ Embeddings saved to 'fma_muq_embeddings.csv' and 'fma_muq_embeddings.pt'")
print(f"\nFinal embedding dimensions: {embeddings.shape}")
print("Done! 🎵")