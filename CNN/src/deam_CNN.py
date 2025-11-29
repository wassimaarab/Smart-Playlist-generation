import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import librosa
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# ETAPE 1 : PRÉPARATION ET ÉQUILIBRAGE (Logique Script)
# ============================================================
def prepare_and_balance_deam(raw_root, output_dir="DEAM_Balanced"):
    """
    Lit le DEAM brut, calcule l'équilibre parfait, et crée un dossier physique propre.
    """
    if os.path.exists(output_dir):
        # Vérification rapide si le dossier semble déjà rempli
        if len(os.listdir(output_dir)) == 4:
            print(f"✅ Dossier '{output_dir}' déjà existant. On l'utilise directement.")
            return

    print(f"--- 🛠️  Création du dataset équilibré depuis {raw_root} ---")
    
    # A. Localisation des fichiers (Audio & CSV)
    audio_dir = os.path.join(raw_root, "DEAM_audio", "MEMD_audio")
    annot_path = os.path.join(raw_root, "DEAM_Annotations", "annotations", 
                              "annotations averaged per song", "song_level")
    
    csv_files = [
        os.path.join(annot_path, "static_annotations_averaged_songs_1_2000.csv"),
        os.path.join(annot_path, "static_annotations_averaged_songs_2000_2058.csv")
    ]
    
    # B. Lecture et Fusion CSV
    try:
        dfs = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
        if not dfs: raise FileNotFoundError("Aucun CSV trouvé.")
        df = pd.concat(dfs, ignore_index=True)
        df.columns = [c.strip() for c in df.columns] # Clean noms colonnes
    except Exception as e:
        print(f"ERREUR CRITIQUE lecture CSV : {e}")
        return

    # Colonnes
    v_col = next((c for c in df.columns if 'valence' in c.lower() and 'mean' in c.lower()), None)
    a_col = next((c for c in df.columns if 'arousal' in c.lower() and 'mean' in c.lower()), None)
    id_col = next((c for c in df.columns if 'id' in c.lower()), None)

    # C. Calcul Médianes & Tri Virtuel
    median_v = df[v_col].median()
    median_a = df[a_col].median()
    print(f"📊 Seuils Médians -> Valence: {median_v:.2f}, Arousal: {median_a:.2f}")

    # Dictionnaire temporaire : { 'happy': [id1, id2...], 'sad': [...] }
    temp_files = {'happy': [], 'angry': [], 'sad': [], 'relax': []}
    
    for _, row in df.iterrows():
        song_id = int(row[id_col])
        v, a = row[v_col], row[a_col]
        
        # Vérif existence physique du mp3
        fname = f"{song_id}.mp3"
        src_path = os.path.join(audio_dir, fname)
        if not os.path.exists(src_path): 
            # Check padding (ex: 00005.mp3)
            src_path = os.path.join(audio_dir, f"{song_id:05d}.mp3")
            if not os.path.exists(src_path): continue
        
        # Classification
        if v >= median_v and a >= median_a: label = 'happy'
        elif v < median_v and a >= median_a: label = 'angry'
        elif v < median_v and a < median_a: label = 'sad'
        else: label = 'relax'
        
        temp_files[label].append(src_path)

    # D. Équilibrage (Undersampling)
    counts = [len(v) for v in temp_files.values()]
    min_count = min(counts)
    print(f"📊 Répartition brute : {[f'{k}:{len(v)}' for k,v in temp_files.items()]}")
    print(f"⚖️  Équilibrage forcé à {min_count} fichiers par classe.")

    # E. Copie Physique
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for cls in temp_files:
        os.makedirs(os.path.join(output_dir, cls))
        files = temp_files[cls]
        random.seed(42)
        random.shuffle(files) # Mélange pour ne pas prendre que les premiers
        
        selected = files[:min_count] # On coupe
        
        for src in selected:
            fname = os.path.basename(src)
            shutil.copy2(src, os.path.join(output_dir, cls, fname))
            
    print(f"✅ Terminé ! Dataset '{output_dir}' prêt ({min_count*4} fichiers).")


# ============================================================
# ETAPE 2 : DATASET PYTORCH (Sur le dossier équilibré)
# ============================================================
class CleanFolderDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=128, duration=30, cache_dir="cache_final"):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.cache_dir = cache_dir
        
        self.classes = sorted(os.listdir(root_dir)) # ['angry', 'happy', 'relax', 'sad']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.file_list = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.endswith('.mp3'):
                    self.file_list.append((os.path.join(cls_dir, f), self.class_to_idx[cls], f))
        
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        self.target_len = int(np.ceil(self.duration * self.sr / 512))

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path, label, fname = self.file_list[idx]
        
        # Cache ID
        clean_name = os.path.splitext(fname)[0]
        cache_path = os.path.join(self.cache_dir, f"{clean_name}.pt")
        
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), torch.tensor(label)
            except: pass
            
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
            target_s = int(self.sr * self.duration)
            if len(y) < target_s: y = np.pad(y, (0, target_s-len(y)))
            else: y = y[:target_s]
            
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels), ref=np.max)
            S = (S - S.min()) / (S.max() - S.min() + 1e-6)
            
            if S.shape[1] < self.target_len:
                S = np.pad(S, ((0,0), (0, self.target_len - S.shape[1])))
            else: S = S[:, :self.target_len]
            
            X = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
            torch.save(X, cache_path)
            return X, torch.tensor(label)
        except:
            return self.__getitem__(np.random.randint(len(self.file_list)))

# ============================================================
# ETAPE 3 : MODELE & TRAINING
# ============================================================
class TemporalCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        self.pool4 = nn.MaxPool2d(4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool4(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = torch.cat([x.mean(2), x.max(2)[0], torch.sqrt((x**2).sum(2))], 1)
        x = F.relu(self.fc1(x.mean(2)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_process(dataset_dir, epochs=30):
    dataset = CleanFolderDataset(dataset_dir)
    
    # Split
    idx = np.arange(len(dataset))
    lbl = [i[1] for i in dataset.file_list]
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=lbl, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=[lbl[i] for i in train_idx], random_state=42)
    
    loaders = {
        'train': DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True),
        'val': DataLoader(Subset(dataset, val_idx), batch_size=16),
        'test': DataLoader(Subset(dataset, test_idx), batch_size=16)
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device} | Size: {len(dataset)} files")
    
    model = TemporalCNN(4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    
    history = {'t_loss':[], 'v_loss':[], 't_acc':[], 'v_acc':[]}
    
    for ep in range(epochs):
        model.train()
        l_sum, c_sum, n = 0, 0, 0
        for X, y in loaders['train']:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            l_sum += loss.item()*len(y); c_sum += (out.argmax(1)==y).sum().item(); n += len(y)
        
        history['t_loss'].append(l_sum/n); history['t_acc'].append(c_sum/n)
        
        model.eval()
        l_sum, c_sum, n = 0, 0, 0
        with torch.no_grad():
            for X, y in loaders['val']:
                X, y = X.to(device), y.to(device)
                out = model(X)
                l_sum += crit(out, y).item()*len(y); c_sum += (out.argmax(1)==y).sum().item(); n += len(y)
        
        history['v_loss'].append(l_sum/n); history['v_acc'].append(c_sum/n)
        print(f"Ep {ep+1:02d} | T_Loss:{history['t_loss'][-1]:.3f} Acc:{history['t_acc'][-1]:.3f} | V_Loss:{history['v_loss'][-1]:.3f} Acc:{history['v_acc'][-1]:.3f}")

    # Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history['t_loss'], label='T'); plt.plot(history['v_loss'], label='V'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(history['t_acc'], label='T'); plt.plot(history['v_acc'], label='V'); plt.title('Accuracy'); plt.legend()
    plt.show()
    
    torch.save(model.state_dict(), "CNN/weights_balanced/final_model.pth")
    print("Modèle sauvegardé.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    RAW_SOURCE = "deam"          # Ton dossier brut actuel
    BALANCED_DIR = "DEAM_Balanced" # Le dossier propre qui sera créé
    
    # 1. Préparer
    prepare_and_balance_deam(RAW_SOURCE, BALANCED_DIR)
    
    # 2. Entraîner
    if os.path.exists(BALANCED_DIR):
        if not os.path.exists("CNN/weights_balanced"): os.makedirs("CNN/weights_balanced")
        train_process(BALANCED_DIR, epochs=40)