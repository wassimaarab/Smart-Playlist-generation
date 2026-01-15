import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import librosa
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# 1. MODÈLE MONOTÂCHE (MÊME ARCHITECTURE, SANS BRANCHE ÉMOTION)
# ============================================================================
class TemporalCNN(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        
        # --- BLOC 1 : EXTRACTION DE FEATURES (IDENTIQUE AU TRONC COMMUN) ---
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        
        self.pool4x = nn.MaxPool2d(4)
        self.pool2x = nn.MaxPool2d(2)
        
        # --- BLOC 2 : CLASSIFICATION (ANCIENNE "TÊTE GENRE" INTÉGRÉE) ---
        # FC Shared (512*3 -> 2048)
        self.fc1 = nn.Linear(512*3, 2048)
        
        # Partie qui était dans la "tête genre" (2048 -> 1024 -> num_genres)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_genres)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 1. Convolutions
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        
        # 2. Global Statistics Pooling (Mean, Max, L2)
        mean = x.mean(dim=2)
        max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2) # [Batch, 512*3]
        
        # 3. Dense Layers (Linéaire)
        x = F.relu(self.fc1(x))      # -> 2048
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))      # -> 1024
        x = self.dropout(x)
        
        out = self.fc3(x)            # -> num_genres
        return out

# ============================================================================
# 2. DATASET
# ============================================================================
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.original_data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        
        cat = pd.Categorical(self.original_data['label'])
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}

        self.cached_specs = []
        self.data = self.load_data_to_cache() 
        
        cat = pd.Categorical(self.data['label'])
        self.labels = cat.codes
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}

    def __len__(self):
        return len(self.cached_specs)

    def load_data_to_cache(self):
        print("Mise en cache des spectrogrammes...")
        valid_indices = [] 
        for idx in range(len(self.original_data)):
            try:
                label_str = self.original_data.loc[idx, 'label']
                filename = self.original_data.loc[idx, 'filename']
                path = os.path.join(self.audio_dir, label_str, filename)

                y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
                if len(y) < 1000: continue
                S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels), ref=np.max)
                S_norm = (S - S.min()) / (S.max() - S.min() + 1e-6)
                
                tgt_len = int(np.ceil(self.duration * self.sr / 512))
                if S_norm.shape[1] < tgt_len: 
                    S_norm = np.pad(S_norm, ((0,0), (0, tgt_len - S_norm.shape[1])))
                else: S_norm = S_norm[:, :tgt_len]
                
                X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
                y_idx = self.label_to_index[label_str]
                
                self.cached_specs.append((X, torch.tensor(y_idx, dtype=torch.long)))
                valid_indices.append(idx)
            except Exception: continue
        print(f"Terminé : {len(self.cached_specs)} fichiers.")
        return self.original_data.iloc[valid_indices].reset_index(drop=True)

    def __getitem__(self, idx):
        return self.cached_specs[idx]

# ============================================================================
# 3. LOADER & SPLIT
# ============================================================================
def stratified_split(dataset, test_size=0.15, val_size=0.15, random_state=42):
    df = dataset.data
    train_val_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, stratify=df['label'], random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, stratify=df.loc[train_val_idx, 'label'], random_state=random_state)
    return train_idx, val_idx, test_idx

def get_loaders(dataset, batch_size=16):
    train_idx, val_idx, test_idx = stratified_split(dataset)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# ============================================================================
# 4. ENTRAÎNEMENT
# ============================================================================
def train_model(csv_file, audio_dir, epochs=50, batch_size=16, save_path="CNN/weights/temporal_cnn_gtzan.pth", patience=7):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    dataset = AudioDataset(csv_file, audio_dir, duration=28)
    train_loader, valid_loader, test_loader = get_loaders(dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_to_index)
    
    print(f"🚀 Démarrage entraînement (Monotâche) sur {device} | Classes: {num_classes}")
    
    # Modèle monotâche
    model = TemporalCNN(num_genres=num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = save_path.replace('.pth', '_best.pth')

    for epoch in range(epochs):
        model.train()
        train_loss, total_correct, total_samples = 0,0,0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X) # Appel simple, plus de 'task='
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum().item()
            total_samples += y.size(0)
        train_loss /= total_samples
        train_acc = total_correct / total_samples

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for Xv, yv in valid_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                outputs = model(Xv)
                loss_v = criterion(outputs, yv)
                val_loss += loss_v.item() * Xv.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == yv).sum().item()
                val_total += yv.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> ⭐ Saved Best")
        else:
            early_stop_counter += 1
            print(f"  --> No Improv ({early_stop_counter}/{patience})")
            
        if early_stop_counter >= patience:
            print("🛑 Early Stopping."); break

    # Test final
    print("\n🏁 Test sur le meilleur modèle...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            outputs = model(Xt)
            loss_t = criterion(outputs, yt)
            test_loss += loss_t.item() * Xt.size(0)
            preds = outputs.argmax(1)
            test_correct += (preds == yt).sum().item()
            test_total += yt.size(0)
            
    print(f"🏆 RÉSULTAT FINAL (TEST) : Loss: {test_loss/test_total:.4f} - Accuracy: {test_correct/test_total:.4f}")
    torch.save(model.state_dict(), save_path)

    # Plot
    plot_dir = "CNN/training_plot"
    os.makedirs(plot_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, train_losses, label='Train'); plt.plot(epochs_range, val_losses, label='Val'); plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(epochs_range, train_accs, label='Train'); plt.plot(epochs_range, val_accs, label='Val'); plt.title('Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "training_curves_monotask.png")); plt.show()

    return model

if __name__ == "__main__":
    csv_file = 'Data/features_30_sec.csv'
    audio_dir = 'Data/genres_original/'
    if os.path.exists(csv_file) and os.path.exists(audio_dir):
        train_model(csv_file, audio_dir, epochs=100, patience=10)
    else: print("❌ Erreur : Fichiers de données introuvables.")