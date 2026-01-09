import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# 🎛️ CONFIGURATION DE L'EXPÉRIENCE
# ============================================================================
# CHOISIS ICI : "GENRE" ou "EMOTION"
# TRAIN_MODE = "GENRE" 
TRAIN_MODE = "EMOTION"

print(f"\n🔬 MODE EXPÉRIMENTAL : Entraînement sur {TRAIN_MODE} uniquement (avec CCC Loss).")
print("   (L'autre dataset sera projeté en 'Zero-Shot' pour voir l'alignement)")

# ============================================================================
# 0. LA LOSS MAGIQUE : CCC (Concordance Correlation Coefficient)
# ============================================================================
class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        # x: prediction, y: target (batch, 2)
        ccc_v = self.get_ccc(x[:, 0], y[:, 0])
        ccc_a = self.get_ccc(x[:, 1], y[:, 1])
        return 1.0 - (ccc_v + ccc_a) / 2.0

    def get_ccc(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        x_std = torch.std(x); y_std = torch.std(y)
        x_mean = torch.mean(x); y_mean = torch.mean(y)
        return (2 * rho * x_std * y_std) / (x_std ** 2 + y_std ** 2 + (x_mean - y_mean) ** 2 + 1e-8)

# ============================================================================
# 1. MODÈLE PARTAGÉ
# ============================================================================
class SharedCNN(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        # Tronc commun
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        self.pool4x = nn.MaxPool2d(4); self.pool2x = nn.MaxPool2d(2)
        
        self.fc_shared = nn.Linear(512*3, 2048)
        self.dropout = nn.Dropout(0.5)

        # Têtes
        self.head_genre = nn.Linear(2048, num_genres)
        self.head_emotion = nn.Linear(2048, 2) 

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
class GTZANRamDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        cat = pd.Categorical(self.df['label'])
        self.classes = cat.categories
        self.label_to_index = {c: i for i, c in enumerate(self.classes)}
        self.data = [] 
        print(f"⏳ Chargement GTZAN en RAM...")
        for idx, row in self.df.iterrows():
            try:
                fpath = os.path.join(audio_dir, row['label'], row['filename'])
                if not os.path.exists(fpath): continue 
                y, _ = librosa.load(fpath, sr=sr, duration=duration)
                if len(y) < 1000: continue
                S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels), ref=np.max)
                S_norm = (S - S.min()) / (S.max() - S.min() + 1e-6)
                tgt_len = int(np.ceil(duration * sr / 512))
                if S_norm.shape[1] < tgt_len: S_norm = np.pad(S_norm, ((0,0), (0, tgt_len - S_norm.shape[1])))
                else: S_norm = S_norm[:, :tgt_len]
                X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(self.label_to_index[row['label']], dtype=torch.long)
                self.data.append((X, y))
            except: pass
        print(f"✅ GTZAN chargé : {len(self.data)} fichiers.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class DEAMDiskDataset(Dataset):
    def __init__(self, root_dir, cache_dir):
        csv_path = os.path.join(root_dir, "metadata.csv")
        self.cache_dir = cache_dir
        if not os.path.exists(csv_path): self.df = []
        else: self.df = pd.read_csv(csv_path)
        print(f"✅ DEAM indexé : {len(self.df)} fichiers.")
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = torch.tensor([row['valence'] / 10.0, row['arousal'] / 10.0], dtype=torch.float32)
        cache_path = os.path.join(self.cache_dir, f"{row['filename']}.pt")
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), target
            except: pass
        return torch.zeros(1, 128, 1200), target

# ============================================================================
# 3. TRAINING LOOP UNILATÉRALE
# ============================================================================
def run_experiment():
    GTZAN_CSV = 'Data/features_30_sec.csv'; GTZAN_AUDIO = 'Data/genres_original/'
    DEAM_ROOT = 'DEAM_Regression'; DEAM_CACHE = 'cache_reg'
    
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")
    
    # 1. Load Data
    full_gtzan = GTZANRamDataset(GTZAN_CSV, GTZAN_AUDIO)
    full_deam = DEAMDiskDataset(DEAM_ROOT, DEAM_CACHE)
    if len(full_gtzan) == 0 or len(full_deam) == 0: return

    # Split (On entraîne sur 85%, on garde 15% pour tester la performance de la tâche)
    train_size = int(0.85 * len(full_gtzan))
    test_size = len(full_gtzan) - train_size
    train_g, test_g = random_split(full_gtzan, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_size_d = int(0.85 * len(full_deam))
    test_size_d = len(full_deam) - train_size_d
    train_d, test_d = random_split(full_deam, [train_size_d, test_size_d], generator=torch.Generator().manual_seed(42))

    ldr_g_train = DataLoader(train_g, batch_size=16, shuffle=True)
    ldr_d_train = DataLoader(train_d, batch_size=16, shuffle=True)
    
    # Pour la visu, on utilise les Sets de TEST (données jamais vues)
    # Et l'autre dataset complet ou partiel (puisqu'il n'a jamais été vu du tout)
    ldr_g_visu = DataLoader(test_g, batch_size=16, shuffle=False)
    ldr_d_visu = DataLoader(test_d, batch_size=16, shuffle=False)

    model = SharedCNN(num_genres=len(full_gtzan.classes)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = CCCLoss() # Utilisation de la CCC

    print(f"\n🚀 DÉBUT ENTRAÎNEMENT CIBLÉ SUR : {TRAIN_MODE}")
    epochs = 25

    for ep in range(epochs):
        model.train()
        total_loss = 0
        
        if TRAIN_MODE == "GENRE":
            for X, y in ldr_g_train:
                optim.zero_grad()
                out = model(X.to(device), 'genre')
                loss = crit_cls(out, y.to(device))
                loss.backward(); optim.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(ldr_g_train)
            print(f"Ep {ep+1:02d} | Loss GENRE: {avg_loss:.4f}")
            
        elif TRAIN_MODE == "EMOTION":
            for X, y in ldr_d_train:
                optim.zero_grad()
                out = model(X.to(device), 'emotion')
                loss = crit_reg(out, y.to(device)) # CCC Loss
                loss.backward(); optim.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(ldr_d_train)
            print(f"Ep {ep+1:02d} | Loss EMOTION (CCC): {avg_loss:.4f}")

    # ========================================================================
    # 4. VISUALISATION CROISÉE
    # ========================================================================
    print("\n🎨 Calcul du t-SNE Croisé...")
    model.eval()
    embeddings, labels_visu, colors_visu = [], [], []
    deam_quads = [] # Pour stocker la classe émotionnelle
    
    # 1. Extraction GTZAN
    with torch.no_grad():
        for X, y in ldr_g_visu:
            emb = model(X.to(device), task='embedding').cpu().numpy()
            embeddings.extend(emb)
            for lbl in y:
                labels_visu.append(full_gtzan.classes[lbl.item()])
                colors_visu.append(f"C{lbl.item()}")
                deam_quads.append("Not DEAM")

    # 2. Extraction DEAM
    with torch.no_grad():
        for X, y in ldr_d_visu:
            emb = model(X.to(device), task='embedding').cpu().numpy()
            embeddings.extend(emb)
            # Quadrants (Basé sur médiane 0.5 car normalisé)
            for val_ar in y:
                v, a = val_ar[0].item(), val_ar[1].item()
                if v>=0.5 and a>=0.5: q, c = "Q1 Happy", "red"
                elif v<0.5 and a>=0.5: q, c = "Q2 Angry", "orange"
                elif v<0.5 and a<0.5: q, c = "Q3 Sad", "blue"
                else: q, c = "Q4 Relax", "green"
                
                labels_visu.append("DEAM")
                colors_visu.append(c)
                deam_quads.append(q)

    # t-SNE Global
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    X_2d = tsne.fit_transform(np.array(embeddings))
    
    idx_gtzan = [i for i, l in enumerate(labels_visu) if l != "DEAM"]
    idx_deam = [i for i, l in enumerate(labels_visu) if l == "DEAM"]

    # --- DOUBLE PLOT ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    
    if TRAIN_MODE == "GENRE":
        fig.suptitle(f"Entraînement : GENRE (GTZAN) -> Projection : ÉMOTION (DEAM)", fontsize=16)
    else:
        fig.suptitle(f"Entraînement : ÉMOTION (DEAM with CCC) -> Projection : GENRE (GTZAN)", fontsize=16)

    # GAUCHE : GENRES
    # On met DEAM en fond gris
    ax1.scatter(X_2d[idx_deam, 0], X_2d[idx_deam, 1], c='lightgray', s=15, alpha=0.3)
    unique_genres = sorted(list(set([l for l in labels_visu if l != "DEAM"])))
    for g in unique_genres:
        local_idx = [i for i in idx_gtzan if labels_visu[i] == g]
        c = colors_visu[local_idx[0]]
        ax1.scatter(X_2d[local_idx, 0], X_2d[local_idx, 1], c=c, s=30, label=g, edgecolors='white', linewidth=0.5)
    ax1.set_title("Coloration par GENRE")
    ax1.legend(fontsize='x-small', loc='upper left')

    # DROITE : EMOTIONS
    # On met GTZAN en fond gris
    ax2.scatter(X_2d[idx_gtzan, 0], X_2d[idx_gtzan, 1], c='lightgray', s=15, alpha=0.3)
    unique_quads = ["Q1 Happy", "Q2 Angry", "Q3 Sad", "Q4 Relax"]
    quad_colors = {"Q1 Happy":"red", "Q2 Angry":"orange", "Q3 Sad":"blue", "Q4 Relax":"green"}
    
    for q in unique_quads:
        local_idx = [i for i in idx_deam if deam_quads[i] == q]
        if len(local_idx) > 0:
            ax2.scatter(X_2d[local_idx, 0], X_2d[local_idx, 1], c=quad_colors[q], s=30, label=q, edgecolors='white', linewidth=0.5)
    
    ax2.set_title("Coloration par ÉMOTION (Quadrants)")
    ax2.legend(fontsize='x-small', loc='upper left')

    plt.tight_layout()
    fname = f"CNN/tsne_plot/cross_analysis_{TRAIN_MODE}_ccc.png"
    plt.savefig(fname)
    plt.show()
    print(f"✅ Résultat sauvegardé : {fname}")

if __name__ == "__main__":
    run_experiment()