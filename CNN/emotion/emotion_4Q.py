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
# 2. Modèle STANDARD (Sans BN, Sans Dropout)
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
# 3. Main Training
# ----------------------------
def train(dataset_dir, epochs=30):
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
    
    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    
    for ep in range(epochs):
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
        
        history['t_loss'].append(l_sum/n); history['t_acc'].append(c_sum/n)
        
        model.eval()
        l_sum, c_sum, n = 0, 0, 0
        with torch.no_grad():
            for X, y in loaders['val']:
                X, y = X.to(device), y.to(device)
                out = model(X)
                l_sum += criterion(out, y).item()*len(y); c_sum += (out.argmax(1)==y).sum().item(); n += len(y)
        
        history['v_loss'].append(l_sum/n); history['v_acc'].append(c_sum/n)
        print(f"Ep {ep+1:02d} | T_Loss:{history['t_loss'][-1]:.3f} Acc:{history['t_acc'][-1]:.3f} | V_Loss:{history['v_loss'][-1]:.3f} Acc:{history['v_acc'][-1]:.3f}")
        
    # --- ÉVALUATION TEST FINAL (CE QUI MANQUAIT) ---
    print("\n🏁 Évaluation finale sur le Test Set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X, y in loaders['test']:
            X, y = X.to(device), y.to(device)
            out = model(X)
            test_correct += (out.argmax(1) == y).sum().item()
            test_total += y.size(0)
            
    print(f"🏆 Accuracy Test Final : {test_correct/test_total:.4f}")
    # -----------------------------------------------

    # Save & Plot
    if not os.path.exists("CNN/weights_4q"): os.makedirs("CNN/weights_4q")
    torch.save(model.state_dict(), "CNN/weights_4q/model_4q.pth")
    print("Modèle sauvegardé : CNN/weights_4q/model_4q.pth")
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history['t_loss'], label='Train'); plt.plot(history['v_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(history['t_acc'], label='Train'); plt.plot(history['v_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "4Q/MER Audio" 
    
    if os.path.exists(DATA_DIR):
        train(DATA_DIR, epochs=50)
    else:
        print(f"❌ Dossier introuvable : {DATA_DIR}")