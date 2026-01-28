import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import shutil  
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Device utilisé : {device}")

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
DEAM_AUDIO_DIR = 'deam/DEAM_audio/MEMD_audio/' 

# ============================================================================
# 1. UTILITAIRES & MODÈLE
# ============================================================================
def ensure_fixed_length(tensor, target_len=1292):
    current_len = tensor.shape[2]
    if current_len > target_len:
        return tensor[:, :, :target_len]
    elif current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len))
    else:
        return tensor

class MultiTaskCNN(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        self.pool4x = nn.MaxPool2d(4); self.pool2x = nn.MaxPool2d(2)
        
        self.fc_shared = nn.Linear(512*3, 2048)
        self.dropout = nn.Dropout(0.5)

        self.head_genre = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, num_genres))
        self.head_emotion = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 2))

    def forward(self, x, task='embedding'):
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        mean = x.mean(dim=2); max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2)
        
        emb = F.relu(self.fc_shared(x))
        if task == 'embedding': return emb
        if task == 'genre': return self.head_genre(self.dropout(emb))
        elif task == 'emotion': return self.head_emotion(self.dropout(emb))

# ============================================================================
# 2. DATASETS 
# ============================================================================
class GTZANFullDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        self.target_len = int(np.ceil(30 * sr / 512)) # ~1292
        self.data = [] 
        
        print(f"⏳ Chargement liste GTZAN...")
        for idx, row in self.df.iterrows():
            fpath = os.path.join(audio_dir, row['label'], row['filename'])
            if os.path.exists(fpath):
                self.data.append({
                    'path': fpath, 'label': row['label'], 'filename': row['filename'], 'source': 'GTZAN'
                })
            
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            y, _ = librosa.load(item['path'], sr=self.sr, duration=self.duration)
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels), ref=np.max)
            S_norm = (S - S.min()) / (S.max() - S.min() + 1e-6)
            X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
            X = ensure_fixed_length(X, self.target_len)
            return {'tensor': X, 'label': item['label'], 'filename': item['filename'], 'source': 'GTZAN'}
        except:
            new_idx = np.random.randint(len(self.data))
            return self.__getitem__(new_idx)

class DEAMCachedFullDataset(Dataset):
    def __init__(self, root_dir, cache_dir):
        csv_path = os.path.join(root_dir, "metadata.csv")
        self.cache_dir = cache_dir
        self.df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        self.target_len = int(np.ceil(30 * 22050 / 512))
        self.valid_files = []
        
        print(f"⏳ Indexation DEAM...")
        for idx, row in self.df.iterrows():
            cache_path = os.path.join(self.cache_dir, f"{row['filename']}.pt")
            if os.path.exists(cache_path):
                self.valid_files.append({
                    'path': cache_path, 'filename': row['filename'],
                    'valence': row['valence'], 'arousal': row['arousal']
                })

    def __len__(self): return len(self.valid_files)
    
    def __getitem__(self, idx):
        try:
            item = self.valid_files[idx]
            X = torch.load(item['path'])
            X = ensure_fixed_length(X, self.target_len)
            
            v, a = item['valence'], item['arousal']
            if v >= 5.0 and a >= 5.0:
                emo_label = "Q1 Happy"
            elif v < 5.0 and a >= 5.0:
                emo_label = "Q2 Angry"
            elif v < 5.0 and a < 5.0:
                emo_label = "Q3 Sad"
            else:
                emo_label = "Q4 Relax"
            
            return {'tensor': X, 'label': emo_label, 'filename': item['filename'], 'source': 'DEAM'}
        except:
            new_idx = np.random.randint(len(self.valid_files))
            return self.__getitem__(new_idx)

# ============================================================================
# 3. EXTRACTION ET ANALYSE
# ============================================================================
def extract_test_embeddings(model, dataloader):
    model.eval()
    embeddings, metadata = [], []
    
    print("\n🚀 Extraction des embeddings sur le Test Set...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['tensor'].to(device)
            emb = model(inputs, task='embedding').cpu().numpy()
            
            for j in range(len(emb)):
                embeddings.append(emb[j])
                metadata.append({
                    'label': batch['label'][j],
                    'filename': batch['filename'][j],
                    'source': batch['source'][j]
                })
            print(f"Batch {i+1}/{len(dataloader)} traité...", end='\r')
            
    return np.array(embeddings), pd.DataFrame(metadata)

def generate_and_analyze(embeddings, meta_df):
    print("\n\n🎨 Calcul du t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    coords = tsne.fit_transform(embeddings)
    
    print("🔗 Indexation des voisins...")
    nn_model = NearestNeighbors(n_neighbors=40, metric='euclidean')
    nn_model.fit(coords)
    
    def create_playlist(seed_idx, size=5, diversity=0.3):
        playlist = [seed_idx]
        available = set(range(len(embeddings))) - {seed_idx}
        while len(playlist) < size and available:
            last = playlist[-1]
            last_coord = coords[last].reshape(1, -1)
            dists, indices = nn_model.kneighbors(last_coord, n_neighbors=25)
            candidates = [(i, d) for i, d in zip(indices[0], dists[0]) if i in available]
            if not candidates: break
            
            if np.random.random() < diversity and len(candidates) > 2:
                sel = candidates[np.random.randint(0, min(15, len(candidates)))]
            else:
                sel = candidates[0]
            playlist.append(sel[0]); available.remove(sel[0])
        return playlist

    playlists = {}
    
    # 1. Metal Journey
    metal_idx = meta_df[meta_df['label'] == 'metal'].index
    if len(metal_idx) > 0: playlists['Metal Journey'] = create_playlist(metal_idx[0], size=12)
        
    # 2. Relaxing/Jazz (Cherche un Jazz ou un Relax)
    relax_idx = meta_df[meta_df['label'].isin(['jazz', 'Q4 Relax'])].index
    if len(relax_idx) > 0: playlists['Chill Vibes'] = create_playlist(relax_idx[0], size=12)

    # 3. Happy Flow
    happy_idx = meta_df[meta_df['label'] == 'Q1 Happy'].index
    if len(happy_idx) > 0: playlists['Happy Flow'] = create_playlist(happy_idx[0], size=12)

    print("\n" + "="*70)
    print("📊 ANALYSE DES PLAYLISTS GÉNÉRÉES")
    print("="*70)

    for name, indices in playlists.items():
        print(f"\n🎵 Playlist : {name}")
        print("-" * 35)
        pl_data = meta_df.iloc[indices].copy()
        pl_data['Track #'] = range(1, len(indices) + 1)
        print(pl_data[['Track #', 'source', 'label', 'filename']].to_string(index=False))

    plt.figure(figsize=(16, 10))
    
    gtzan_df = meta_df[meta_df['source'] == 'GTZAN']
    for g in gtzan_df['label'].unique():
        idx = meta_df[meta_df['label'] == g].index
        plt.scatter(coords[idx, 0], coords[idx, 1], label=g, s=20, alpha=0.5)
        
    deam_df = meta_df[meta_df['source'] == 'DEAM']
    quad_colors = {"Q1 Happy": "red", "Q2 Angry": "orange", "Q3 Sad": "blue", "Q4 Relax": "green"}
    for q in deam_df['label'].unique():
        idx = meta_df[meta_df['label'] == q].index
        c = quad_colors.get(q, 'gray')
        plt.scatter(coords[idx, 0], coords[idx, 1], c=c, label=q, s=20, alpha=0.3)

    colors_pl = ['black', 'darkblue', 'purple']
    for i, (name, indices) in enumerate(playlists.items()):
        path = coords[indices]
        c = colors_pl[i % len(colors_pl)]
        plt.plot(path[:, 0], path[:, 1], c=c, linewidth=2, label=f"PL: {name}", alpha=0.8)
        plt.scatter(path[:, 0], path[:, 1], c=c, s=50, edgecolors='white', zorder=10)
        plt.text(path[0, 0], path[0, 1], "START", fontsize=8, fontweight='bold', color=c)

    plt.title("Espace Latent Musical : Genres (GTZAN) & Émotions (DEAM)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("CNN/tsne_plot/visual_playlist_final.png")
    
    return playlists

# ============================================================================
# 4. FONCTION D'EXPORT PORTABLE
# ============================================================================
def export_portable_playlists(playlists, meta_df, gtzan_root_dir):
    """
    Crée un dossier 'Exported_Playlists', crée des sous-dossiers,
    et COPIE physiquement les fichiers audio dedans.
    """
    base_output_dir = "Exported_Playlists"
    if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)
    
    print(f"\n💾 EXPORT PORTABLE (Copie physique dans {base_output_dir})")
    print("-" * 60)

    for name, indices in playlists.items():
        safe_name = name.replace(" ", "_")
        playlist_folder = os.path.join(base_output_dir, safe_name)
        if not os.path.exists(playlist_folder): os.makedirs(playlist_folder)
        
        m3u_path = os.path.join(playlist_folder, f"PLAY_{safe_name}.m3u")
        
        with open(m3u_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            
            for i, idx in enumerate(indices):
                row = meta_df.iloc[idx]
                fname = row['filename']
                source = row['source']
                label = row['label']
                
                if source == "GTZAN":
                    original_path = os.path.join(gtzan_root_dir, label, fname)
                else:
                    original_path = os.path.join(DEAM_AUDIO_DIR, fname)
                
                if os.path.exists(original_path):
                    new_filename = f"{i+1:02d}_{fname}" 
                    destination_path = os.path.join(playlist_folder, new_filename)
                    
                    try:
                        shutil.copy2(original_path, destination_path)
                        f.write(f"#EXTINF:-1,{source} - {label}\n")
                        f.write(f"{new_filename}\n")
                    except Exception as e:
                        print(f"⚠️ Erreur copie : {fname} -> {e}")
                else:
                    print(f"⚠️ Fichier source introuvable : {original_path}")

        print(f"✅ Playlist '{name}' copiée dans : {playlist_folder}/")

# ============================================================================
# 5. MAIN
# ============================================================================
if __name__ == "__main__":
    WEIGHTS = "CNN/weights_multitask/best_ccc_model.pth"
    if not os.path.exists(WEIGHTS): exit("❌ Poids introuvables.")
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")

    ds_g = GTZANFullDataset('Data/features_30_sec.csv', 'Data/genres_original/')
    ds_d = DEAMCachedFullDataset('DEAM_Regression', 'cache_reg')
    
    if len(ds_g) == 0 and len(ds_d) == 0: exit("❌ Aucun dataset chargé.")

    full_dataset = ConcatDataset([ds_g, ds_d])
    
    _, test_dataset = random_split(full_dataset, [int(0.8*len(full_dataset)), len(full_dataset)-int(0.8*len(full_dataset))], 
                                   generator=torch.Generator().manual_seed(42))
    
    print(f"\n📊 Taille du TEST SET : {len(test_dataset)} morceaux")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultiTaskCNN(num_genres=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))

    embs, meta = extract_test_embeddings(model, test_loader)
    
    playlists = generate_and_analyze(embs, meta)
    
    export_portable_playlists(playlists, meta, 'Data/genres_original/')