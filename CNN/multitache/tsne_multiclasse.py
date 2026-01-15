import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Device utilisé : {device}")

# ============================================================================
# 1. MODÈLE (DOIT ÊTRE IDENTIQUE À L'ENTRAÎNEMENT)
# ============================================================================
def ensure_fixed_length(tensor, target_len=1292):
    current_len = tensor.shape[2]
    if current_len > target_len: return tensor[:, :, :target_len]
    elif current_len < target_len: return F.pad(tensor, (0, target_len - current_len))
    else: return tensor

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

        self.head_genre = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_genres)
        )
        self.head_emotion = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 2) 
        )

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
# 2. DATASETS (ROBUSTES)
# ============================================================================
class GTZANFullDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        self.target_len = int(np.ceil(30 * sr / 512))
        self.data = [] 
        print(f"⏳ Indexation GTZAN...")
        for idx, row in self.df.iterrows():
            fpath = os.path.join(audio_dir, row['label'], row['filename'])
            if os.path.exists(fpath):
                self.data.append({'path': fpath, 'label': row['label'], 'source': 'GTZAN'})
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            y, _ = librosa.load(item['path'], sr=self.sr, duration=self.duration)
            if len(y) < 1000: raise ValueError("Trop court")
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels), ref=np.max)
            S_norm = (S - S.min()) / (S.max() - S.min() + 1e-6)
            X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
            X = ensure_fixed_length(X, self.target_len)
            return {'tensor': X, 'label': item['label'], 'source': 'GTZAN', 'valence': 0, 'arousal': 0}
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

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
                    'path': cache_path, 'valence': row['valence'], 'arousal': row['arousal']
                })

    def __len__(self): return len(self.valid_files)
    def __getitem__(self, idx):
        try:
            item = self.valid_files[idx]
            X = torch.load(item['path'])
            X = ensure_fixed_length(X, self.target_len)
            # Normalisation 0-1 pour les quadrants
            val = item['valence'] / 10.0
            aro = item['arousal'] / 10.0
            return {'tensor': X, 'label': 'DEAM', 'source': 'DEAM', 'valence': val, 'arousal': aro}
        except:
            return self.__getitem__(np.random.randint(len(self.valid_files)))

# ============================================================================
# 3. EXTRACTION ET PLOT SIDE-BY-SIDE
# ============================================================================
def extract_and_plot_side_by_side(model, dataloader):
    print("\n🚀 Extraction des embeddings du Test Set...")
    model.eval()
    
    embeddings = []
    labels = []    # Pour Genre
    sources = []   # GTZAN ou DEAM
    quadrants = [] # Pour Emotion
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['tensor'].to(device)
            emb = model(inputs, task='embedding')
            emb_np = emb.cpu().numpy()
            
            for j in range(len(emb_np)):
                embeddings.append(emb_np[j])
                sources.append(batch['source'][j])
                labels.append(batch['label'][j])
                
                # Calcul Quadrant (Seulement utile pour DEAM)
                v = batch['valence'][j].item()
                a = batch['arousal'][j].item()
                
                if v >= 0.5 and a >= 0.5: q = "Q1 Happy"
                elif v < 0.5 and a >= 0.5: q = "Q2 Angry"
                elif v < 0.5 and a < 0.5: q = "Q3 Sad"
                else: q = "Q4 Relax"
                quadrants.append(q)
                
            print(f"Batch {i+1}/{len(dataloader)}...", end='\r')

    # --- t-SNE ---
    print("\n\n🎨 Calcul du t-SNE (Projection 2D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    X_2d = tsne.fit_transform(np.array(embeddings))
    
    # --- PRÉPARATION PLOT ---
    idx_gtzan = [i for i, s in enumerate(sources) if s == 'GTZAN']
    idx_deam = [i for i, s in enumerate(sources) if s == 'DEAM']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)
    
    # === GRAPHE 1 : GENRE (Focus GTZAN) ===
    # Fond : DEAM en gris
    ax1.scatter(X_2d[idx_deam, 0], X_2d[idx_deam, 1], c='lightgray', s=10, alpha=0.2)
    
    unique_genres = sorted(list(set([labels[i] for i in idx_gtzan])))
    cmap = plt.get_cmap('tab10')
    
    for i, genre in enumerate(unique_genres):
        # Indices locaux pour ce genre
        indices = [k for k in idx_gtzan if labels[k] == genre]
        color = cmap(i / 10)
        ax1.scatter(X_2d[indices, 0], X_2d[indices, 1], c=[color], label=genre, s=25, alpha=0.7, edgecolors='white', linewidth=0.3)
        
    ax1.set_title("Espace Latent coloré par GENRE (GTZAN)", fontsize=14, fontweight='bold')
    ax1.legend(title="Genres", loc='upper left')
    ax1.grid(alpha=0.3)

    # === GRAPHE 2 : EMOTION (Focus DEAM - Quadrants) ===
    # Fond : GTZAN en gris
    ax2.scatter(X_2d[idx_gtzan, 0], X_2d[idx_gtzan, 1], c='lightgray', s=10, alpha=0.2)
    
    quad_colors = {
        "Q1 Happy": "red",
        "Q2 Angry": "orange",
        "Q3 Sad": "blue",
        "Q4 Relax": "green"
    }
    
    for q_name, q_color in quad_colors.items():
        # Indices locaux pour ce quadrant (parmi DEAM)
        indices = [k for k in idx_deam if quadrants[k] == q_name]
        if len(indices) > 0:
            ax2.scatter(X_2d[indices, 0], X_2d[indices, 1], c=q_color, label=q_name, s=25, alpha=0.7, edgecolors='white', linewidth=0.3)
            
    ax2.set_title("Espace Latent coloré par ÉMOTION (DEAM - Russell's Model)", fontsize=14, fontweight='bold')
    ax2.legend(title="Quadrants", loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = "CNN/tsne_plot/final_side_by_side.png"
    plt.savefig(out_path)
    print(f"\n✅ Graphique sauvegardé : {out_path}")
    plt.show()

# ============================================================================
# 4. MAIN
# ============================================================================
if __name__ == "__main__":
    WEIGHTS = "CNN/weights_multitask/best_ccc_model.pth"
    if not os.path.exists(WEIGHTS): exit(f"❌ Poids introuvables : {WEIGHTS}")
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")

    ds_g = GTZANFullDataset('Data/features_30_sec.csv', 'Data/genres_original/')
    ds_d = DEAMCachedFullDataset('DEAM_Regression', 'cache_reg')
    
    if len(ds_g) == 0 and len(ds_d) == 0: exit("❌ Erreur Dataset.")

    # Concaténation et Split Test
    full_dataset = ConcatDataset([ds_g, ds_d])
    total_len = len(full_dataset)
    train_sz = int(0.7 * total_len); val_sz = int(0.15 * total_len)
    test_sz = total_len - train_sz - val_sz
    
    _, _, test_dataset = random_split(full_dataset, [train_sz, val_sz, test_sz], 
                                      generator=torch.Generator().manual_seed(42))
    
    print(f"\n📊 Visualisation sur le TEST SET ({len(test_dataset)} morceaux)")
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultiTaskCNN(num_genres=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))

    extract_and_plot_side_by_side(model, loader)