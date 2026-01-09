import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
import librosa
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# 1. PRÉPARATION (Mode Régression)
# ============================================================================
def prepare_deam_regression(raw_root, output_dir="DEAM_Regression"):
    csv_out_path = os.path.join(output_dir, "metadata.csv")
    audio_out_dir = os.path.join(output_dir, "audio")
    
    if os.path.exists(output_dir) and os.path.exists(csv_out_path) and len(os.listdir(audio_out_dir)) > 100:
        print(f"✅ Dataset Régression déjà prêt dans '{output_dir}'.")
        return pd.read_csv(csv_out_path)

    print(f"--- 🛠️  Préparation Dataset (Régression) ---")
    audio_src = os.path.join(raw_root, "DEAM_audio", "MEMD_audio")
    annot_path = os.path.join(raw_root, "DEAM_Annotations", "annotations", 
                              "annotations averaged per song", "song_level")
    
    csv_files = [
        os.path.join(annot_path, "static_annotations_averaged_songs_1_2000.csv"),
        os.path.join(annot_path, "static_annotations_averaged_songs_2000_2058.csv")
    ]
    
    try:
        dfs = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"❌ Erreur CSV: {e}"); exit()

    v_col = next(c for c in df.columns if 'valence' in c.lower() and 'mean' in c.lower())
    a_col = next(c for c in df.columns if 'arousal' in c.lower() and 'mean' in c.lower())
    id_col = next(c for c in df.columns if 'id' in c.lower())

    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(audio_out_dir)

    data_entries = []
    print("🚀 Copie des fichiers...")
    for _, row in df.iterrows():
        song_id = int(row[id_col])
        fname = f"{song_id}.mp3"
        src = os.path.join(audio_src, fname)
        if not os.path.exists(src): src = os.path.join(audio_src, f"{song_id:05d}.mp3")
        
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(audio_out_dir, fname))
            data_entries.append({
                'filename': fname,
                'valence': row[v_col],
                'arousal': row[a_col]
            })

    new_df = pd.DataFrame(data_entries)
    new_df.to_csv(csv_out_path, index=False)
    print(f"✅ Terminé ! {len(new_df)} fichiers.")
    return new_df

# ============================================================================
# 2. DATASET (Avec Augmentation)
# ============================================================================
class RegressionDataset(Dataset):
    def __init__(self, root_dir, dataframe, sr=22050, n_mels=128, duration=30, cache_dir="cache_reg", augment=False):
        self.root_dir = os.path.join(root_dir, "audio")
        self.df = dataframe.reset_index(drop=True)
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.cache_dir = cache_dir
        self.augment = augment 
        self.target_len = int(np.ceil(self.duration * self.sr / 512))
        
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        path = os.path.join(self.root_dir, fname)
        
        # Cibles (Normalisées 0.1 - 0.9)
        target = torch.tensor([row['valence'] / 10.0, row['arousal'] / 10.0], dtype=torch.float32)
        
        cache_path = os.path.join(self.cache_dir, f"{fname}.pt")
        
        # A. Si pas d'augmentation, on utilise le cache
        if not self.augment and os.path.exists(cache_path):
            try: return torch.load(cache_path), target
            except: pass
            
        # B. Sinon on calcule (pour augmenter)
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
            
            if self.augment:
                if np.random.rand() < 0.5: y = y + 0.005 * np.random.randn(len(y))
                if np.random.rand() < 0.5: 
                    shift = np.random.randint(self.sr * 5)
                    y = np.roll(y, shift)

            tgt_len = int(self.sr * self.duration)
            if len(y) < tgt_len: y = np.pad(y, (0, tgt_len - len(y)))
            else: y = y[:tgt_len]
            
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels), ref=np.max)
            S = (S - S.min()) / (S.max() - S.min() + 1e-6)
            
            if S.shape[1] < self.target_len: S = np.pad(S, ((0,0), (0, self.target_len - S.shape[1])))
            else: S = S[:, :self.target_len]
            
            if self.augment and np.random.rand() < 0.5:
                f_mask = np.random.randint(5, 15)
                f_pos = np.random.randint(0, self.n_mels - f_mask)
                S[f_pos:f_pos+f_mask, :] = 0

            X = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
            
            if not self.augment: torch.save(X, cache_path)
            return X, target
            
        except: return torch.zeros(1, 128, self.target_len), target

# ============================================================================
# 3. CNN REGRESSION
# ============================================================================
class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        
        self.pool4x = nn.MaxPool2d(4); self.pool2x = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(512*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_embedding=False):
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        
        mean = x.mean(dim=2); max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2)
        
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        emb = F.relu(self.fc2(x))
        if return_embedding: return emb
        
        emb = self.dropout(emb)
        return self.fc3(emb)

# ============================================================================
# 4. TRAINING LOOP (Avec Train R2 & Val R2)
# ============================================================================
def train_regression(root_dir, epochs=70, patience=10):
    df_meta = prepare_deam_regression("deam", root_dir)
    
    idx = np.arange(len(df_meta))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    
    train_ds = RegressionDataset(root_dir, df_meta.iloc[train_idx], augment=True)
    val_ds = RegressionDataset(root_dir, df_meta.iloc[val_idx], augment=False)
    test_ds = RegressionDataset(root_dir, df_meta.iloc[test_idx], augment=False)
    
    workers = 4 if os.name != 'nt' else 0
    loaders = {
        'train': DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=workers),
        'val': DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=workers),
        'test': DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=workers)
    }
    
    model = RegressionCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    crit = nn.MSELoss() 
    
    # Ajout de 'train_r2' dans l'historique
    H = {'loss':[], 'val_loss':[], 'train_r2':[], 'val_r2':[]}
    
    best_loss = float('inf') 
    early_stop_counter = 0
    model_path = "CNN/weights_balanced/best_reg_model.pth"
    
    print(f"\n🚀 Start Regression Training ({epochs} Epochs)...")
    
    for ep in range(epochs):
        # --- TRAIN ---
        model.train()
        l_sum = 0
        train_preds, train_targs = [], []
        
        for X, y in loaders['train']:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            l_sum += loss.item()
            
            # On stocke pour le R2
            train_preds.append(out.detach().cpu().numpy())
            train_targs.append(y.detach().cpu().numpy())
        
        avg_loss = l_sum / len(loaders['train'])
        train_r2 = r2_score(np.vstack(train_targs), np.vstack(train_preds)) # Calcul du R2 Train
        
        H['loss'].append(avg_loss)
        H['train_r2'].append(train_r2)
        
        # --- VALIDATION ---
        model.eval()
        v_l_sum = 0
        val_preds, val_targs = [], []
        with torch.no_grad():
            for X, y in loaders['val']:
                X, y = X.to(device), y.to(device)
                out = model(X)
                v_l_sum += crit(out, y).item()
                val_preds.append(out.cpu().numpy())
                val_targs.append(y.cpu().numpy())
        
        avg_v_loss = v_l_sum / len(loaders['val'])
        val_r2 = r2_score(np.vstack(val_targs), np.vstack(val_preds))
        
        H['val_loss'].append(avg_v_loss)
        H['val_r2'].append(val_r2)
        
        # --- EARLY STOPPING CHECK ---
        save_msg = ""
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            early_stop_counter = 0 
            torch.save(model.state_dict(), model_path)
            save_msg = "💾 New Best!"
        else:
            early_stop_counter += 1
            
        print(f"Ep {ep+1:02d} | Loss: {avg_loss:.4f} | R2: {train_r2:.3f} || Val Loss: {avg_v_loss:.4f} | Val R2: {val_r2:.3f} {save_msg}")
        
        if early_stop_counter >= patience:
            print(f"🛑 Arrêt anticipé après {ep+1} époques.")
            break

    # ============================================================
    # 5. EVALUATION FINALE
    # ============================================================
    print("\n🏁 Test Final...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    preds, targets, embs = [], [], []
    with torch.no_grad():
        for X, y in loaders['test']:
            X = X.to(device)
            out = model(X)
            preds.extend(out.cpu().numpy()); targets.extend(y.numpy())
            embs.extend(model(X, return_embedding=True).cpu().numpy())
    
    preds = np.array(preds); targets = np.array(targets)
    final_r2 = r2_score(targets, preds)
    print(f"🏆 Test R2 Score : {final_r2:.4f}")
    
    # --- COURBES (MODIFIÉES POUR AVOIR TRAIN ET VAL) ---
    plt.figure(figsize=(12,5))
    
    # Loss Plot
    plt.subplot(1,2,1)
    plt.plot(H['loss'], label='Train Loss', color='blue')
    plt.plot(H['val_loss'], label='Val Loss', color='orange')
    plt.title('MSE Loss (Train vs Val)')
    plt.legend()
    plt.grid(True)
    
    # R2 Plot (C'est ici qu'on a ajouté la courbe Train)
    plt.subplot(1,2,2)
    plt.plot(H['train_r2'], label='Train R2', color='blue', linestyle='--')
    plt.plot(H['val_r2'], label='Val R2', color='orange')
    plt.title('R2 Score (Train vs Val)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('CNN/tsne_plot/regression_curves.png')
    plt.show()
    
    # Scatter Plot
    plt.figure(figsize=(6,6))
    plt.scatter(targets[:,0], preds[:,0], alpha=0.5, label='Valence', color='blue')
    plt.scatter(targets[:,1], preds[:,1], alpha=0.5, label='Arousal', color='red')
    plt.plot([0,1], [0,1], 'k--', alpha=0.7)
    plt.xlabel("Réel"); plt.ylabel("Prédit"); plt.legend(); plt.grid(True)
    plt.savefig('CNN/tsne_plot/regression_scatter.png')
    plt.show()

    # t-SNE
    print("🎨 Calcul t-SNE...")
    # On détermine les quadrants pour les couleurs
    colors, labels = [], []
    for i in range(len(targets)):
        v, a = targets[i]
        # Seuils à 0.5 car normalisé
        if v >= 0.5 and a >= 0.5: l="Q1 Happy"; c="red"
        elif v < 0.5 and a >= 0.5: l="Q2 Angry"; c="orange"
        elif v < 0.5 and a < 0.5: l="Q3 Sad"; c="blue"
        else: l="Q4 Relax"; c="green"
        colors.append(c); labels.append(l)
        
    tsne = TSNE(n_components=2, perplexity=30, init='pca').fit_transform(np.array(embs))
    plt.figure(figsize=(10,8))
    unique_labels = list(set(labels))
    for ul in sorted(unique_labels):
        mask = np.array(labels) == ul
        if np.any(mask):
            c = colors[np.where(mask)[0][0]]
            plt.scatter(tsne[mask, 0], tsne[mask, 1], c=c, label=ul, alpha=0.7)
            
    plt.title(f"t-SNE Embedding (Regression) - Test R2={final_r2:.2f}")
    plt.legend()
    plt.savefig('CNN/tsne_plot/tsne_regression.png')
    plt.show()

if __name__ == "__main__":
    if not os.path.exists("CNN/weights_balanced"): os.makedirs("CNN/weights_balanced")
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")
    train_regression("DEAM_Regression", epochs=70)