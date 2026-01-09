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
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Device utilisé : {device}")

# ============================================================================
# 1. UTILITAIRES & MODÈLE
# ============================================================================
def ensure_fixed_length(tensor, target_len=1292):
    """Force une taille fixe pour éviter les erreurs de batching"""
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
# 2. DATASETS (ROBUSTES AUX ERREURS)
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
                    'path': fpath,
                    'label': row['label'],
                    'filename': row['filename'],
                    'source': 'GTZAN'
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
        except Exception as e:
            # En cas de fichier corrompu (ex: jazz.00054.wav), on en prend un autre
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
                    'path': cache_path,
                    'filename': row['filename'],
                    'valence': row['valence'],
                    'arousal': row['arousal']
                })

    def __len__(self): return len(self.valid_files)
    
    def __getitem__(self, idx):
        try:
            item = self.valid_files[idx]
            X = torch.load(item['path'])
            X = ensure_fixed_length(X, self.target_len)
            return {'tensor': X, 'label': 'DEAM_Emotion', 'filename': item['filename'], 'source': 'DEAM'}
        except:
            new_idx = np.random.randint(len(self.valid_files))
            return self.__getitem__(new_idx)

# ============================================================================
# 3. EXTRACTION, GENERATION VISUELLE ET AFFICHAGE
# ============================================================================
def extract_test_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    metadata = []
    
    print("\n🚀 Extraction des embeddings sur le Test Set...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['tensor'].to(device)
            emb = model(inputs, task='embedding')
            
            emb_np = emb.cpu().numpy()
            
            for j in range(len(emb_np)):
                embeddings.append(emb_np[j])
                metadata.append({
                    'label': batch['label'][j],
                    'filename': batch['filename'][j],
                    'source': batch['source'][j]
                })
            
            print(f"Batch {i+1}/{len(dataloader)} traité...", end='\r')
            
    return np.array(embeddings), pd.DataFrame(metadata)

def generate_and_analyze(embeddings, meta_df):
    # 1. Calcul t-SNE (D'ABORD)
    print("\n\n🎨 Calcul du t-SNE (Projection 2048D -> 2D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    coords = tsne.fit_transform(embeddings)
    
    # 2. Modèle KNN pour les playlists (SUR LE T-SNE - Distance Euclidienne)
    print("🔗 Indexation des voisins basée sur la PROJECTION VISUELLE (2D)...")
    nn_model = NearestNeighbors(n_neighbors=20, metric='euclidean')
    nn_model.fit(coords) # On fit sur les coordonnées 2D
    
    def create_playlist(seed_idx, size=5, diversity=0.2):
        playlist = [seed_idx]
        available = set(range(len(embeddings))) - {seed_idx}
        
        while len(playlist) < size and available:
            last = playlist[-1]
            # On cherche les voisins de la coordonnée 2D
            last_coord = coords[last].reshape(1, -1)
            
            dists, indices = nn_model.kneighbors(last_coord, n_neighbors=15)
            
            candidates = [(i, d) for i, d in zip(indices[0], dists[0]) if i in available]
            if not candidates: break
            
            if np.random.random() < diversity and len(candidates) > 2:
                sel = candidates[np.random.randint(0, min(5, len(candidates)))]
            else:
                sel = candidates[0]
            
            playlist.append(sel[0]); available.remove(sel[0])
        return playlist

    # 3. Génération des Playlists
    playlists = {}
    
    # Métal
    metal_idx = meta_df[meta_df['label'] == 'metal'].index
    if len(metal_idx) > 0:
        playlists['Metal Journey'] = create_playlist(metal_idx[0], size=8, diversity=0.1)
        
    # Jazz
    jazz_idx = meta_df[meta_df['label'] == 'jazz'].index
    if len(jazz_idx) > 0:
        playlists['Jazz Vibes'] = create_playlist(jazz_idx[0], size=8, diversity=0.2)
        
    # DEAM (Emotion)
    deam_idx = meta_df[meta_df['source'] == 'DEAM'].index
    if len(deam_idx) > 0:
        playlists['Emotion Flow'] = create_playlist(deam_idx[0], size=8, diversity=0.3)

    # --- 4. AFFICHAGE DÉTAILLÉ (TEXTE) ---
    print("\n" + "="*60)
    print("📊 ANALYSE DÉTAILLÉE (BASÉE SUR PROXIMITÉ VISUELLE)")
    print("="*60)

    for name, indices in playlists.items():
        print(f"\n🎵 Playlist : {name}")
        print("-" * 30)
        
        pl_data = meta_df.iloc[indices].copy()
        pl_data['Track #'] = range(1, len(indices) + 1)
        cols = ['Track #', 'source', 'label', 'filename'] 
        print(pl_data[cols].to_string(index=False))

    # --- 5. VISUALISATION GRAPHIQUE ---
    plt.figure(figsize=(16, 10))
    
    # Fond
    gtzan_df = meta_df[meta_df['source'] == 'GTZAN']
    unique_genres = gtzan_df['label'].unique()
    cmap = plt.get_cmap('tab10')
    genre_to_color = {g: cmap(i/10) for i, g in enumerate(unique_genres)}
    
    for genre in unique_genres:
        idx = meta_df[meta_df['label'] == genre].index
        plt.scatter(coords[idx, 0], coords[idx, 1], c=[genre_to_color[genre]], label=genre, s=20, alpha=0.5)
        
    deam_idx = meta_df[meta_df['source'] == 'DEAM'].index
    plt.scatter(coords[deam_idx, 0], coords[deam_idx, 1], c='lightgray', label='DEAM (Emotion)', s=20, alpha=0.3, zorder=0)

    # Playlists
    colors_pl = ['black', 'darkblue', 'darkred']
    for i, (name, indices) in enumerate(playlists.items()):
        path = coords[indices]
        c = colors_pl[i % len(colors_pl)]
        
        plt.plot(path[:, 0], path[:, 1], c=c, linewidth=2, label=f"PL: {name}", alpha=0.9)
        plt.scatter(path[:, 0], path[:, 1], c=c, s=60, edgecolors='white', zorder=10)
        
        plt.text(path[0, 0], path[0, 1], "START", fontsize=9, fontweight='bold', color=c, zorder=11)
        plt.text(path[-1, 0], path[-1, 1], "END", fontsize=9, fontweight='bold', color=c, zorder=11)

    plt.title("Visualisation t-SNE (Playlists basées sur la distance visuelle)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = "CNN/tsne_plot/test_set_visual_playlist.png"
    plt.savefig(out_path)
    print(f"\n✅ Graphique sauvegardé : {out_path}")
    plt.show()

# ============================================================================
# 4. MAIN
# ============================================================================
if __name__ == "__main__":
    WEIGHTS = "CNN/weights_multitask/best_ccc_model.pth"
    if not os.path.exists(WEIGHTS): exit("❌ Poids introuvables.")
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")

    ds_g = GTZANFullDataset('Data/features_30_sec.csv', 'Data/genres_original/')
    ds_d = DEAMCachedFullDataset('DEAM_Regression', 'cache_reg')
    
    if len(ds_g) == 0 and len(ds_d) == 0: exit("❌ Aucun dataset chargé.")

    full_dataset = ConcatDataset([ds_g, ds_d])
    total_len = len(full_dataset)
    train_sz = int(0.7 * total_len); val_sz = int(0.15 * total_len)
    test_sz = total_len - train_sz - val_sz
    
    _, _, test_dataset = random_split(full_dataset, [train_sz, val_sz, test_sz], 
                                      generator=torch.Generator().manual_seed(42))
    
    print(f"\n📊 Taille du TEST SET : {len(test_dataset)} morceaux")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultiTaskCNN(num_genres=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))

    embs, meta = extract_test_embeddings(model, test_loader)
    generate_and_analyze(embs, meta)