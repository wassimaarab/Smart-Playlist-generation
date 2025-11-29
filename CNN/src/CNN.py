import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
        
        cat = pd.Categorical(self.original_data['label'])
        self.labels = cat.codes
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}
        self.index_to_label = {i: cat.categories[i] for i in range(len(cat.categories))}

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
                # --- DÉBUT : CODE DE CHARGEMENT ET DE PRÉTRAITEMENT (CRÉATION DE X et y) ---
                
                # 1. Chargement du fichier audio avec librosa
                y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
                if y.size == 0:
                    raise ValueError("Audio vide ou corrompu")
                    
                # 2. Calcul du Mel-Spectrogramme
                S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                # 3. Normalisation (Vérification de la division par zéro)
                denom = S_dB.max() - S_dB.min()
                if denom == 0:
                    raise ValueError("Spectrogramme plat (silence ou erreur de lecture)")
                    
                S_norm = (S_dB - S_dB.min()) / denom
                
                # 4. Conversion en tenseurs PyTorch
                X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
                y_label_idx = self.label_to_index[label_str] # Le label est bien défini ici.
                y = torch.tensor(y_label_idx, dtype=torch.long)
                
                # --- FIN : CODE DE CHARGEMENT ET DE PRÉTRAITEMENT ---
                
                # STOCKAGE dans le cache (l'ordre ici correspondra au nouvel index)
                self.cached_specs.append((X, y))
                
                # Enregistrement de l'indice original pour le filtrage
                valid_indices_original_df.append(idx)
                
            except Exception as e:
                # Échantillon ignoré
                # print(f"Erreur lors du chargement de {path} : {e}") # Décommenter pour débogage
                continue
                
        print(f"Mise en cache terminée. {len(self.cached_specs)} échantillons chargés.")
        
        # Retourne uniquement les lignes du DF original qui correspondent aux échantillons chargés
        return self.original_data.iloc[valid_indices_original_df].reset_index(drop=True)

    def __getitem__(self, idx):
        # Récupération directe du cache
        # Note : Le bloc try-except pour vérifier les indices dans __getitem__ n'est plus nécessaire 
        # car len(self.cached_specs) et len(self.data) sont synchronisés.
        return self.cached_specs[idx]


# ----------------------------
# Stratified split
# ----------------------------
def stratified_split(dataset, test_size=0.15, val_size=0.15, random_state=42):
    """
    test_size = fraction du dataset pour test
    val_size = fraction du train_val pour validation
    """
    df = dataset.data
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=df.loc[train_val_idx, 'label'],
        random_state=random_state
    )

    return train_idx, val_idx, test_idx

# ----------------------------
# DataLoader
# ----------------------------
def get_loaders(dataset, batch_size=16):
    train_idx, val_idx, test_idx = stratified_split(dataset)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


# ----------------------------
# CNN
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
        if return_embedding:
            return embedding
        return out


# ----------------------------
# Entraînement
# ----------------------------
def train_model(csv_file, audio_dir, epochs=50, batch_size=16, save_path="CNN/weights/temporal_cnn_gtzan.pth", val_threshold=0.65):
    dataset = AudioDataset(csv_file, audio_dir, duration=28)
    train_size = int(0.7 * len(dataset))
    test_size = (len(dataset) - train_size)//2
    valid_size = len(dataset) - train_size - test_size
    train_loader, valid_loader, test_loader, train_idx, val_idx, test_idx = get_loaders(dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(pd.read_csv(csv_file)['label'].unique())
    model = TemporalCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Historique
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1:02d}/{epochs}: "
              f"Train Loss {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # --- Early stopping ---
        if val_acc >= val_threshold:
            print(f"✅ Early stopping: validation accuracy {val_acc:.4f} >= {val_threshold}")
            break

    # ---- TEST ----
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

    print(f"🧪 Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

    # ----------------------------
    # Tracé des courbes
    # ----------------------------
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    # --- Courbe des pertes ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Époch')
    plt.ylabel('Loss')
    plt.title('Courbes de perte')
    plt.legend()

    # --- Courbe des précisions ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Époch')
    plt.ylabel('Accuracy')
    plt.title('Courbes de précision')
    plt.legend()

    plt.tight_layout()
    plt.savefig("CNN/training_plot/training_curves_1.png", dpi=300, bbox_inches='tight')
    plt.show()

    return model, test_loader, dataset


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    csv_file = 'Data/features_30_sec.csv'
    audio_dir = 'Data/genres_original/'
    model, test_loader, dataset = train_model(csv_file, audio_dir, epochs=100)
