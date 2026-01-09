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

# ----------------------------
# Dataset
# ----------------------------
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.original_data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        
        # Initialisation temporaire pour structure
        cat = pd.Categorical(self.original_data['label'])
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}

        self.cached_specs = []
        # Le DataFrame 'self.data' contient DÉSORMAIS uniquement les échantillons valides
        self.data = self.load_data_to_cache() 
        
        # Les labels et mappings sont basés sur les données validées
        cat = pd.Categorical(self.data['label'])
        self.labels = cat.codes
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}
        self.index_to_label = {i: cat.categories[i] for i in range(len(cat.categories))}

    def __len__(self):
        return len(self.cached_specs)

    def load_data_to_cache(self):
        print("Mise en cache des spectrogrammes...")
        valid_indices_original_df = [] 
        
        for idx in range(len(self.original_data)):
            label_str = self.original_data.loc[idx, 'label']
            filename = self.original_data.loc[idx, 'filename']
            path = os.path.join(self.audio_dir, label_str, filename)

            try:
                y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
                if y.size == 0: raise ValueError("Audio vide")
                    
                S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                denom = S_dB.max() - S_dB.min()
                if denom == 0: raise ValueError("Silence")
                    
                S_norm = (S_dB - S_dB.min()) / denom
                
                X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
                y_label_idx = self.label_to_index[label_str]
                y = torch.tensor(y_label_idx, dtype=torch.long)
                
                self.cached_specs.append((X, y))
                valid_indices_original_df.append(idx)
                
            except Exception as e:
                continue
                
        print(f"Mise en cache terminée. {len(self.cached_specs)} échantillons chargés.")
        return self.original_data.iloc[valid_indices_original_df].reset_index(drop=True)

    def __getitem__(self, idx):
        return self.cached_specs[idx]


# ----------------------------
# Stratified split & Loader
# ----------------------------
def stratified_split(dataset, test_size=0.15, val_size=0.15, random_state=42):
    df = dataset.data
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=df['label'], random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, stratify=df.loc[train_val_idx, 'label'], random_state=random_state
    )
    return train_idx, val_idx, test_idx

def get_loaders(dataset, batch_size=16):
    train_idx, val_idx, test_idx = stratified_split(dataset)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


# ----------------------------
# CNN Model
# ----------------------------
class TemporalCNN(nn.Module):
    def __init__(self, num_classes=10):
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
        out = self.fc3(embedding)
        if return_embedding: return embedding
        return out


# ----------------------------
# Entraînement (Modifié avec Early Stopping + Best Model)
# ----------------------------
def train_model(csv_file, audio_dir, epochs=50, batch_size=16, save_path="CNN/weights/temporal_cnn_gtzan.pth", patience=7):
    # Création du dossier de sauvegarde si inexistant
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    dataset = AudioDataset(csv_file, audio_dir, duration=28)
    train_loader, valid_loader, test_loader, _, _, _ = get_loaders(dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_to_index)
    
    print(f"🚀 Démarrage entraînement sur {device} | Classes: {num_classes}")
    
    model = TemporalCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Historique
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Variables Early Stopping & Best Model
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = save_path.replace('.pth', '_best.pth') # ex: cnn_best.pth

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss, total_correct, total_samples = 0,0,0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum().item()
            total_samples += y.size(0)
        train_loss /= total_samples
        train_acc = total_correct / total_samples

        # ---- VALIDATION ----
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

        # Sauvegarde historique
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | "
              f"V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}")
        
        # ---- LOGIQUE EARLY STOPPING & SAVE BEST ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> ⭐ Nouveau meilleur modèle sauvegardé (Val Loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  --> Pas d'amélioration ({early_stop_counter}/{patience})")
            
        if early_stop_counter >= patience:
            print(f"🛑 Arrêt anticipé (Early Stopping) activé après {epoch+1} époques.")
            break

    # ---- TEST FINAL (Avec le MEILLEUR modèle) ----
    print("\n🏁 Chargement du meilleur modèle pour le Test...")
    model.load_state_dict(torch.load(best_model_path)) # Rechargement critique !
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
    test_loss /= test_total
    test_acc = test_correct / test_total

    print(f"🏆 RÉSULTAT FINAL (TEST) : Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}")

    # On sauvegarde le dernier état comme "modèle final" si besoin, ou on renomme le best
    torch.save(model.state_dict(), save_path)
    print(f"Modèle final (meilleur) sauvegardé sous : {save_path}")

    # ----------------------------
    # Tracé des courbes
    # ----------------------------
    # Création du dossier plot si inexistant
    plot_dir = "CNN/training_plot"
    os.makedirs(plot_dir, exist_ok=True)
    
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.title('Courbes de perte')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.title('Courbes de précision')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "training_curves_best.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graphique sauvegardé : {plot_path}")

    return model, test_loader, dataset

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    csv_file = 'Data/features_30_sec.csv'
    audio_dir = 'Data/genres_original/'
    
    if os.path.exists(csv_file) and os.path.exists(audio_dir):
        # Augmentation de la patience à 10 pour laisser le temps au modèle d'apprendre
        model, test_loader, dataset = train_model(csv_file, audio_dir, epochs=100, patience=10)
    else:
        print("❌ Erreur : Fichiers de données introuvables.")