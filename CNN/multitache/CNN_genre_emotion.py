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
# 1. MODÈLE UNIFIÉ
# ============================================================================
class MultiTaskCNN(nn.Module):
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

        # Tête GENRE
        self.head_genre = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_genres)
        )

        # Tête EMOTION (Régression 2 valeurs)
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
        # Target normalisée entre 0.0 et 1.0 (car DEAM est 1-10)
        target = torch.tensor([row['valence'] / 10.0, row['arousal'] / 10.0], dtype=torch.float32)
        cache_path = os.path.join(self.cache_dir, f"{row['filename']}.pt")
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), target
            except: pass
        return torch.zeros(1, 128, 1200), target

# ============================================================================
# 3. TRAINING AVEC MSE LOSS
# ============================================================================
def train_and_evaluate():
    GTZAN_CSV = 'Data/features_30_sec.csv'
    GTZAN_AUDIO = 'Data/genres_original/'
    DEAM_ROOT = 'DEAM_Regression'
    DEAM_CACHE = 'cache_reg'
    
    WEIGHTS_DIR = "CNN/weights_multitask"
    PLOT_DIR = "CNN/training_plot"
    if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_ccc_model.pth")

    if not os.path.exists(GTZAN_AUDIO) or not os.path.exists(DEAM_CACHE): 
        print("❌ Chemins introuvables."); return

    full_gtzan = GTZANRamDataset(GTZAN_CSV, GTZAN_AUDIO)
    full_deam = DEAMDiskDataset(DEAM_ROOT, DEAM_CACHE)
    if len(full_gtzan) == 0 or len(full_deam) == 0: return

    def get_splits(dataset):
        size = len(dataset)
        train_sz = int(0.7 * size); val_sz = int(0.15 * size); test_sz = size - train_sz - val_sz
        return random_split(dataset, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(42))

    train_g, val_g, test_g = get_splits(full_gtzan)
    train_d, val_d, test_d = get_splits(full_deam)

    workers = 4 if os.name != 'nt' else 0
    ldr_g_train = DataLoader(train_g, batch_size=16, shuffle=True, num_workers=workers)
    ldr_d_train = DataLoader(train_d, batch_size=16, shuffle=True, num_workers=workers)
    ldr_g_val = DataLoader(val_g, batch_size=16, shuffle=False, num_workers=workers)
    ldr_d_val = DataLoader(val_d, batch_size=16, shuffle=False, num_workers=workers)
    ldr_g_test = DataLoader(test_g, batch_size=16, shuffle=False, num_workers=workers)
    ldr_d_test = DataLoader(test_d, batch_size=16, shuffle=False, num_workers=workers)

    model = MultiTaskCNN(num_genres=len(full_gtzan.classes)).to(device)
    # 1. On identifie les paramètres des têtes spécifiques
    genre_params = list(map(id, model.head_genre.parameters()))
    emotion_params = list(map(id, model.head_emotion.parameters()))
    
    # 2. On filtre pour récupérer le reste (le tronc commun : conv1, conv2, etc.)
    base_params = filter(lambda p: id(p) not in genre_params + emotion_params, 
                         model.parameters())

    # 3. On crée l'optimizer avec des groupes de paramètres
    optim = torch.optim.Adam([
        # --- TRONC COMMUN (Vitesse standard) ---
        {'params': base_params, 'lr': 1e-4}, 
        
        # --- GENRE (Vitesse Boostée) ---
        # On peut le laisser à 1e-4 ou le pousser un peu (ex: 5e-4) pour qu'il apprenne vite
        {'params': model.head_genre.parameters(), 'lr': 1e-4}, 
        
        # --- EMOTION (FREIN À MAIN) ---
        # On divise le learning rate par 10 ou 20 !
        # Ça va forcer cette partie à converger beaucoup plus lentement (ex: en 20 epochs au lieu de 5)
        {'params': model.head_emotion.parameters(), 'lr': 1e-5} 
    ])
    
    print("🔧 Optimizer configuré avec Learning Rates différenciés !")
    
    # --- CHANGEMENT ICI : RETOUR A MSE LOSS ---
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss() 

    print("\n🚀 DÉBUT ENTRAÎNEMENT HYBRIDE (CrossEntropy + MSE)...")
    # On augmente le poids car MSE renvoie des valeurs très petites (ex: 0.05) comparé à CE (ex: 1.5)
    EMO_WEIGHT = 50.0 

    epochs = 40; patience = 8; counter = 0; best_score = float('inf')
    iters = max(len(ldr_g_train), len(ldr_d_train))

    H = {
        'loss_g_train': [], 'loss_g_val': [], 'acc_g_train': [], 'acc_g_val': [],
        'loss_d_train': [], 'loss_d_val': [], 'r2_d_train': [], 'r2_d_val': []
    }

    for ep in range(epochs):
        model.train()
        tg_preds, tg_targs, tg_loss = [], [], 0
        td_preds, td_targs, td_loss = [], [], 0
        
        iter_g = iter(ldr_g_train); iter_d = iter(ldr_d_train)

        for _ in range(iters):
            optim.zero_grad()
            
            # 1. GTZAN
            try: batch_g = next(iter_g)
            except StopIteration: iter_g = iter(ldr_g_train); batch_g = next(iter_g)
            Xg, yg = batch_g
            if Xg.shape[0] > 1:
                out = model(Xg.to(device), 'genre')
                loss_g = crit_cls(out, yg.to(device))
                loss_g.backward()
                tg_loss += loss_g.item()
                tg_preds.extend(out.argmax(1).cpu().numpy()); tg_targs.extend(yg.numpy())

            # 2. DEAM (MSE)
            try: batch_d = next(iter_d)
            except StopIteration: iter_d = iter(ldr_d_train); batch_d = next(iter_d)
            Xd, yd = batch_d
            if Xd.shape[0] > 1:
                out = model(Xd.to(device), 'emotion')
                loss_d = crit_reg(out, yd.to(device))
                (loss_d * EMO_WEIGHT).backward() # Pondération x10
                td_loss += loss_d.item()
                td_preds.extend(out.detach().cpu().numpy()); td_targs.extend(yd.numpy())
            
            optim.step()

        # Metrics Train
        avg_tg_loss = tg_loss/iters
        avg_td_loss = td_loss/iters
        acc_g_train = accuracy_score(tg_targs, tg_preds)
        r2_d_train = r2_score(np.array(td_targs), np.array(td_preds))
        
        H['loss_g_train'].append(avg_tg_loss); H['acc_g_train'].append(acc_g_train)
        H['loss_d_train'].append(avg_td_loss); H['r2_d_train'].append(r2_d_train)

        # --- VALIDATION ---
        model.eval()
        vg_preds, vg_targs, vg_loss = [], [], 0
        vd_preds, vd_targs, vd_loss = [], [], 0
        
        with torch.no_grad():
            for X, y in ldr_g_val:
                out = model(X.to(device), 'genre')
                vg_loss += crit_cls(out, y.to(device)).item()
                vg_preds.extend(out.argmax(1).cpu().numpy()); vg_targs.extend(y.numpy())
                
            for X, y in ldr_d_val:
                out = model(X.to(device), 'emotion')
                vd_loss += crit_reg(out, y.to(device)).item()
                vd_preds.extend(out.cpu().numpy()); vd_targs.extend(y.numpy())

        # Metrics Val
        avg_vg_loss = vg_loss / len(ldr_g_val)
        avg_vd_loss = vd_loss / len(ldr_d_val)
        acc_g_val = accuracy_score(vg_targs, vg_preds)
        r2_d_val = r2_score(np.array(vd_targs), np.array(vd_preds))
        
        H['loss_g_val'].append(avg_vg_loss); H['acc_g_val'].append(acc_g_val)
        H['loss_d_val'].append(avg_vd_loss); H['r2_d_val'].append(r2_d_val)
        
        # --- EARLY STOPPING & SAVE ---
        total_val_loss = avg_vg_loss + (avg_vd_loss * EMO_WEIGHT)
        
        save_msg = ""
        if total_val_loss < best_score:
            best_score = total_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            counter = 0; save_msg = "💾 New Best!"
        else:
            counter += 1; save_msg = f"⏳ {counter}/{patience}"

        # --- AFFICHAGE COMPLET ---
        print(f"Ep {ep+1:02d} | "
              f"GENRE [L:{avg_tg_loss:.2f}->{avg_vg_loss:.2f} | Acc:{acc_g_train:.2f}->{acc_g_val:.2f}] | "
              f"EMO [L:{avg_td_loss:.3f}->{avg_vd_loss:.3f} | R2:{r2_d_train:.2f}->{r2_d_val:.2f}] {save_msg}")
        
        if counter >= patience: 
            print("🛑 Early Stopping"); break

    # ========================================================================
    # 4. PLOTTING
    # ========================================================================
    print("\n📊 Génération des courbes d'entraînement...")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1); plt.plot(H['loss_g_train'], label='Train'); plt.plot(H['loss_g_val'], label='Val'); plt.title('GENRE: CrossEntropy Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 2); plt.plot(H['acc_g_train'], label='Train'); plt.plot(H['acc_g_val'], label='Val'); plt.title('GENRE: Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 3); plt.plot(H['loss_d_train'], label='Train'); plt.plot(H['loss_d_val'], label='Val'); plt.title('EMOTION: MSE Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 4); plt.plot(H['r2_d_train'], label='Train'); plt.plot(H['r2_d_val'], label='Val'); plt.title('EMOTION: R2 Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves_multitask_mse.png"))
    plt.show()

    # ========================================================================
    # 5. TEST FINAL
    # ========================================================================
    print(f"\n🔄 Chargement Meilleur Modèle ({BEST_MODEL_PATH})...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    # On calcule juste les scores finaux pour vérifier
    test_g_preds, test_g_targs = [], []
    test_d_preds, test_d_targs = [], []

    with torch.no_grad():
        for X, y in ldr_g_test:
            out = model(X.to(device), 'genre')
            test_g_preds.extend(out.argmax(1).cpu().numpy()); test_g_targs.extend(y.numpy())
        for X, y in ldr_d_test:
            out = model(X.to(device), 'emotion')
            test_d_preds.extend(out.cpu().numpy()); test_d_targs.extend(y.numpy())

    final_acc = accuracy_score(test_g_targs, test_g_preds)
    final_r2 = r2_score(np.array(test_d_targs), np.array(test_d_preds))

    print("\n" + "="*40)
    print("🏆 RÉSULTATS FINAUX (TEST SET)")
    print("="*40)
    print(f"🎸 GENRE Accuracy : {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"❤️ EMOTION R2 Score : {final_r2:.4f}")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists("CNN/tsne_plot"): os.makedirs("CNN/tsne_plot")
    train_and_evaluate()