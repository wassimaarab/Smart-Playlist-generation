import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
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
# CONFIGURATION DE L'EXPÉRIENCE
# ============================================================================

TRAIN_MODE = "GENRE"

print(f"\n🔬 MODE EXPÉRIMENTAL : Entraînement sur {TRAIN_MODE} uniquement (avec Early Stopping).")

# ============================================================================
# 0. UTILITAIRES
# ============================================================================
def ensure_fixed_length(tensor, target_len=1292):
    current_len = tensor.shape[2]
    if current_len > target_len:
        return tensor[:, :, :target_len]
    elif current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len))
    else:
        return tensor

# ============================================================================
# 1. MODÈLE
# ============================================================================
class SharedCNN(nn.Module):
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
# 2. DATASETS
# ============================================================================
class GTZANRobustDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        cat = pd.Categorical(self.df['label'])
        self.classes = cat.categories
        self.label_to_index = {c: i for i, c in enumerate(self.classes)}
        
        self.valid_data = [] 
        print(f"⏳ Indexation GTZAN...")
        for idx, row in self.df.iterrows():
            fpath = os.path.join(audio_dir, row['label'], row['filename'])
            if os.path.exists(fpath):
                self.valid_data.append((fpath, row['label']))
        self.target_len = int(np.ceil(duration * sr / 512))

    def __len__(self): return len(self.valid_data)

    def __getitem__(self, idx):
        try:
            fpath, label = self.valid_data[idx]
            y, _ = librosa.load(fpath, sr=self.sr, duration=self.duration)
            if len(y) < 1000: raise ValueError("Audio trop court")
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels), ref=np.max)
            S_norm = (S - S.min()) / (S.max() - S.min() + 1e-6)
            X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
            X = ensure_fixed_length(X, self.target_len)
            y_idx = torch.tensor(self.label_to_index[label], dtype=torch.long)
            return X, y_idx
        except Exception as e:
            new_idx = np.random.randint(len(self.valid_data))
            return self.__getitem__(new_idx)

class DEAMRobustDataset(Dataset):
    def __init__(self, root_dir, cache_dir):
        csv_path = os.path.join(root_dir, "metadata.csv")
        self.cache_dir = cache_dir
        if not os.path.exists(csv_path): self.df = []
        else: self.df = pd.read_csv(csv_path)
        self.target_len = int(np.ceil(30 * 22050 / 512)) 
        self.valid_data = []
        print(f"⏳ Indexation DEAM...")
        for idx, row in self.df.iterrows():
            cache_path = os.path.join(self.cache_dir, f"{row['filename']}.pt")
            if os.path.exists(cache_path):
                self.valid_data.append((cache_path, row['valence'], row['arousal']))
    def __len__(self): return len(self.valid_data)
    def __getitem__(self, idx):
        try:
            path, val, aro = self.valid_data[idx]
            target = torch.tensor([val / 10.0, aro / 10.0], dtype=torch.float32)
            X = torch.load(path)
            X = ensure_fixed_length(X, self.target_len)
            return X, target
        except:
            new_idx = np.random.randint(len(self.valid_data))
            return self.__getitem__(new_idx)

# ============================================================================
# 3. FONCTIONS D'ÉVALUATION
# ============================================================================
def evaluate_genre(model, loader):
    model.eval()
    preds, targs, total_loss = [], [], 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X, 'genre')
            loss = crit(out, y)
            total_loss += loss.item()
            preds.extend(out.argmax(1).cpu().numpy())
            targs.extend(y.cpu().numpy())
    return total_loss / len(loader), accuracy_score(targs, preds)

def evaluate_emotion(model, loader, emo_weight=50.0):
    model.eval()
    preds, targs, total_loss = [], [], 0
    crit = nn.MSELoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X, 'emotion')
            loss = crit(out, y)
            total_loss += loss.item() * emo_weight # Scaling
            preds.extend(out.cpu().numpy())
            targs.extend(y.cpu().numpy())
    return total_loss / len(loader), r2_score(np.array(targs), np.array(preds))

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def run_experiment():
    GTZAN_CSV = 'Data/features_30_sec.csv'; GTZAN_AUDIO = 'Data/genres_original/'
    DEAM_ROOT = 'DEAM_Regression'; DEAM_CACHE = 'cache_reg'
    SAVE_DIR = "CNN/weights_multitask"
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_cross_{TRAIN_MODE.lower()}.pth")
    
    full_gtzan = GTZANRobustDataset(GTZAN_CSV, GTZAN_AUDIO)
    full_deam = DEAMRobustDataset(DEAM_ROOT, DEAM_CACHE)
    if len(full_gtzan) == 0 or len(full_deam) == 0: 
        print("❌ Erreur : Datasets vides."); return

    def create_splits(dataset):
        total = len(dataset)
        train_sz = int(0.7 * total)
        val_sz = int(0.15 * total)
        test_sz = total - train_sz - val_sz
        return random_split(dataset, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(42))

    train_g, val_g, test_g = create_splits(full_gtzan)
    train_d, val_d, test_d = create_splits(full_deam)

    ldr_g_train = DataLoader(train_g, batch_size=16, shuffle=True)
    ldr_g_val   = DataLoader(val_g, batch_size=16, shuffle=False)
    ldr_g_test  = DataLoader(test_g, batch_size=16, shuffle=False)

    ldr_d_train = DataLoader(train_d, batch_size=16, shuffle=True)
    ldr_d_val   = DataLoader(val_d, batch_size=16, shuffle=False)
    ldr_d_test  = DataLoader(test_d, batch_size=16, shuffle=False)

    model = SharedCNN(num_genres=len(full_gtzan.classes)).to(device)
    
    genre_params = list(map(id, model.head_genre.parameters()))
    emotion_params = list(map(id, model.head_emotion.parameters()))
    base_params = filter(lambda p: id(p) not in genre_params + emotion_params, model.parameters())

    optim = torch.optim.Adam([
        {'params': base_params, 'lr': 1e-4}, 
        {'params': model.head_genre.parameters(), 'lr': 1e-4}, 
        {'params': model.head_emotion.parameters(), 'lr': 1e-5} 
    ])
    
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss()
    EMO_WEIGHT = 50.0 

    print(f"\n🚀 DÉBUT ENTRAÎNEMENT CIBLÉ SUR : {TRAIN_MODE}")
    epochs = 40
    patience = 8
    counter = 0
    best_val_metric = float('inf')

    for ep in range(epochs):
        model.train()
        train_loss_accum = 0
        
        if TRAIN_MODE == "GENRE":
            for X, y in ldr_g_train:
                optim.zero_grad()
                out = model(X.to(device), 'genre')
                loss = crit_cls(out, y.to(device))
                loss.backward(); optim.step()
                train_loss_accum += loss.item()
            avg_train_loss = train_loss_accum / len(ldr_g_train)
            
            val_loss, val_acc = evaluate_genre(model, ldr_g_val)
            print(f"Ep {ep+1:02d} | Train Loss:{avg_train_loss:.4f} | Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.4f}", end="")
            
            current_val_score = val_loss 

        elif TRAIN_MODE == "EMOTION":
            for X, y in ldr_d_train:
                optim.zero_grad()
                out = model(X.to(device), 'emotion')
                loss = crit_reg(out, y.to(device))
                (loss * EMO_WEIGHT).backward(); optim.step()
                train_loss_accum += loss.item() * EMO_WEIGHT
            avg_train_loss = train_loss_accum / len(ldr_d_train)
            
            val_loss, val_r2 = evaluate_emotion(model, ldr_d_val, EMO_WEIGHT)
            print(f"Ep {ep+1:02d} | Train Loss:{avg_train_loss:.4f} | Val Loss:{val_loss:.4f} | Val R2:{val_r2:.4f}", end="")
            
            current_val_score = val_loss

        if current_val_score < best_val_metric:
            best_val_metric = current_val_score
            counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(" 💾 Best!")
        else:
            counter += 1
            print(f" ⏳ {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early Stopping Activé.")
                break

    # ========================================================================
    # 5. ÉVALUATION FINALE (Double Tâche)
    # ========================================================================
    print("\n" + "="*50)
    print("🏆 ÉVALUATION FINALE SUR LE TEST SET (Meilleur Modèle)")
    print("="*50)
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    _, test_acc = evaluate_genre(model, ldr_g_test)
    print(f"🎸 GENRE (GTZAN Test) Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    _, test_r2 = evaluate_emotion(model, ldr_d_test, EMO_WEIGHT)
    print(f"❤️ EMOTION (DEAM Test) R2 Score : {test_r2:.4f}")
    
    print("-" * 50)
    if TRAIN_MODE == "GENRE":
        print("Note : Le score Émotion est probablement bas car le modèle n'a appris que le Genre.")
    else:
        print("Note : Le score Genre est probablement bas car le modèle n'a appris que l'Émotion.")

    # ========================================================================
    # 6. VISUALISATION CROISÉE
    # ========================================================================
    ldr_visu_g = ldr_g_test
    ldr_visu_d = ldr_d_test
    
    print("\n🎨 Calcul du t-SNE Croisé...")
    model.eval()
    embeddings, labels_visu, colors_visu = [], [], []
    deam_quads = [] 
    
    with torch.no_grad():
        for X, y in ldr_visu_g:
            X = ensure_fixed_length(X)
            emb = model(X.to(device), task='embedding').cpu().numpy()
            embeddings.extend(emb)
            for lbl in y:
                labels_visu.append(full_gtzan.classes[lbl.item()])
                colors_visu.append(f"C{lbl.item()}")
                deam_quads.append("Not DEAM")

    with torch.no_grad():
        for X, y in ldr_visu_d:
            X = ensure_fixed_length(X)
            emb = model(X.to(device), task='embedding').cpu().numpy()
            embeddings.extend(emb)
            for val_ar in y:
                v, a = val_ar[0].item(), val_ar[1].item()
                if v>=0.5 and a>=0.5: q, c = "Q1 Happy", "red"
                elif v<0.5 and a>=0.5: q, c = "Q2 Angry", "orange"
                elif v<0.5 and a<0.5: q, c = "Q3 Sad", "blue"
                else: q, c = "Q4 Relax", "green"
                
                labels_visu.append("DEAM")
                colors_visu.append(c)
                deam_quads.append(q)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    X_2d = tsne.fit_transform(np.array(embeddings))
    
    idx_gtzan = [i for i, l in enumerate(labels_visu) if l != "DEAM"]
    idx_deam = [i for i, l in enumerate(labels_visu) if l == "DEAM"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    
    if TRAIN_MODE == "GENRE":
        fig.suptitle(f"Entraînement : GENRE (GTZAN) -> Projection : ÉMOTION (DEAM)", fontsize=16)
    else:
        fig.suptitle(f"Entraînement : ÉMOTION (DEAM with MSE) -> Projection : GENRE (GTZAN)", fontsize=16)

    ax1.scatter(X_2d[idx_deam, 0], X_2d[idx_deam, 1], c='lightgray', s=15, alpha=0.3) 
    unique_genres = sorted(list(set([l for l in labels_visu if l != "DEAM"])))
    for g in unique_genres:
        local_idx = [i for i in idx_gtzan if labels_visu[i] == g]
        if local_idx:
            c = colors_visu[local_idx[0]]
            ax1.scatter(X_2d[local_idx, 0], X_2d[local_idx, 1], c=c, s=30, label=g, edgecolors='white', linewidth=0.5)
    ax1.set_title("Coloration par GENRE")
    ax1.legend(fontsize='x-small', loc='upper left')

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
    fname = f"CNN/tsne_plot/cross_analysis_{TRAIN_MODE}_mse.png"
    plt.savefig(fname)
    plt.show()
    print(f"✅ Résultat sauvegardé : {fname}")

if __name__ == "__main__":
    run_experiment()