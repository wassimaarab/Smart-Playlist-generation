import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 1. Dataset (Cache + Folder Structure)
# ----------------------------
class FourQDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=128, duration=30, cache_dir="cache_4q"):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.cache_dir = cache_dir
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Dossier introuvable : {root_dir}")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.file_list = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.mp3'):
                    self.file_list.append((os.path.join(cls_dir, f), self.class_to_idx[cls], f))
        
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        self.target_len = int(np.ceil(self.duration * self.sr / 512))
        
        print(f"Classes détectées : {self.classes}")
        print(f"Total fichiers : {len(self.file_list)}")

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path, label, fname = self.file_list[idx]
        
        clean_id = os.path.splitext(fname)[0]
        cache_path = os.path.join(self.cache_dir, f"{clean_id}_{label}.pt")
        
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), torch.tensor(label)
            except: pass
            
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
            target_samples = int(self.sr * self.duration)
            if len(y) < target_samples: 
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')
            else: 
                y = y[:target_samples]
            
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels), ref=np.max)
            denom = S.max() - S.min()
            if denom == 0: denom = 1e-6
            S = (S - S.min()) / denom
            
            if S.shape[1] < self.target_len:
                S = np.pad(S, ((0,0), (0, self.target_len - S.shape[1])))
            else: 
                S = S[:, :self.target_len]
            
            X = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
            torch.save(X, cache_path)
            return X, torch.tensor(label)

        except Exception as e:
            return self.__getitem__(np.random.randint(len(self.file_list)))

# ----------------------------
# 2. Modèle STANDARD
# ----------------------------
class TemporalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(1,4), padding=1)
        
        self.pool4x = nn.MaxPool2d(kernel_size=(4))
        self.pool2x = nn.MaxPool2d(kernel_size=(2))
        
        self.fc1 = nn.Linear(512*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        
    def forward(self, x, return_embedding=False):
        x = F.relu(self.conv1(x))
        x = self.pool4x(x)
        x = F.relu(self.conv2(x))
        x = self.pool2x(x)
        x = F.relu(self.conv3(x))
        x = self.pool2x(x)
        x = F.relu(self.conv4(x))
        
        mean = x.mean(dim=2)
        max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        
        x = torch.cat([mean, max_, l2], dim=1)
        x = x.mean(dim=2)
        
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        
        if return_embedding:
            return embedding
            
        out = self.fc3(embedding)
        return out

# ----------------------------
# 3. Main Training (Modifié: Early Stopping + Best Model)
# ----------------------------
def train(dataset_dir, epochs=100, patience=7):
    print(f"--- Initialisation Dataset sur {dataset_dir} ---")
    dataset = FourQDataset(dataset_dir)
    
    idx = np.arange(len(dataset))
    lbl = [i[1] for i in dataset.file_list]
    
    # Split
    train_idx, test_idx = train_test_split(idx, test_size=0.15, stratify=lbl, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=[lbl[i] for i in train_idx], random_state=42)
    
    loaders = {
        'train': DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True),
        'val': DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False),
        'test': DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Start Training on {device}")
    
    model = TemporalCNN(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Historique pour les courbes
    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    
    # Variables pour Early Stopping et Sauvegarde
    best_val_loss = float('inf')
    early_stop_counter = 0
    save_dir = "CNN/weights_4q"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    best_model_path = os.path.join(save_dir, "best_model_4q.pth")
    
    for ep in range(epochs):
        # --- TRAIN ---
        model.train()
        l_sum, c_sum, n = 0, 0, 0
        for X, y in loaders['train']:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()*len(y); c_sum += (out.argmax(1)==y).sum().item(); n += len(y)
        
        train_loss = l_sum/n
        train_acc = c_sum/n
        history['t_loss'].append(train_loss)
        history['t_acc'].append(train_acc)
        
        # --- VALIDATION ---
        model.eval()
        l_sum, c_sum, n = 0, 0, 0
        with torch.no_grad():
            for X, y in loaders['val']:
                X, y = X.to(device), y.to(device)
                out = model(X)
                l_sum += criterion(out, y).item()*len(y); c_sum += (out.argmax(1)==y).sum().item(); n += len(y)
        
        val_loss = l_sum/n
        val_acc = c_sum/n
        history['v_loss'].append(val_loss)
        history['v_acc'].append(val_acc)
        
        print(f"Ep {ep+1:02d} | T_Loss:{train_loss:.3f} Acc:{train_acc:.3f} | V_Loss:{val_loss:.3f} Acc:{val_acc:.3f}")
        
        # --- CHECKPOINT & EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  --> 💾 Meilleur modèle sauvegardé (Validation Loss améliorée).")
        else:
            early_stop_counter += 1
            print(f"  --> Pas d'amélioration ({early_stop_counter}/{patience})")
            
        if early_stop_counter >= patience:
            print("🛑 Arrêt anticipé (Early Stopping) activé.")
            break
            
    # --- FIN DE L'ENTRAINEMENT ---
    
    # --- TRACÉ DES COURBES ---
    print("\n📊 Génération des courbes...")
    plt.figure(figsize=(12, 5))
    
    # Courbe de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Train Loss', color='blue')
    plt.plot(history['v_loss'], label='Val Loss', color='orange')
    plt.title('Loss par époque')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Courbe d'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['t_acc'], label='Train Acc', color='blue')
    plt.plot(history['v_acc'], label='Val Acc', color='orange')
    plt.title('Accuracy par époque')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("CNN/training_curves.png") # Sauvegarde l'image
    plt.show() # Affiche l'image
    
    # --- ÉVALUATION FINALE SUR TEST SET ---
    print("\n🏁 Évaluation finale sur le Test Set (avec le MEILLEUR modèle)...")
    
    # IMPORTANT : On recharge les poids du meilleur modèle, pas le dernier !
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    # On peut ajouter une matrice de confusion ici si on veut
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loaders['test']:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(1)
            
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    test_acc = test_correct/test_total
    print(f"🏆 Accuracy Test Final : {test_acc:.4f}")
    print(f"📁 Modèle sauvegardé sous : {best_model_path}")
    print(f"📉 Courbes sauvegardées sous : CNN/training_curves.png")

if __name__ == "__main__":
    DATA_DIR = "4Q/MER Audio" 
    
    if os.path.exists(DATA_DIR):
        train(DATA_DIR, epochs=100, patience=10) # Patience augmentée à 10
    else:
        print(f"❌ Dossier introuvable : {DATA_DIR}")