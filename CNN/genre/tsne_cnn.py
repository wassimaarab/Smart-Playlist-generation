import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CLASSE DU MODÈLE (Avec modification pour extraire l'embedding)
# ============================================================================
class TemporalCNN(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        
        # --- BLOC 1 : EXTRACTION DE FEATURES ---
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        
        self.pool4x = nn.MaxPool2d(4)
        self.pool2x = nn.MaxPool2d(2)
        
        # --- BLOC 2 : CLASSIFICATION ---
        self.fc1 = nn.Linear(512*3, 2048)
        self.fc2 = nn.Linear(2048, 1024) # C'est ici qu'on récupère l'embedding
        self.fc3 = nn.Linear(1024, num_genres)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_embedding=False):
        # 1. Convolutions
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        
        # 2. Stats
        mean = x.mean(dim=2)
        max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2) 
        
        # 3. Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # On capture l'embedding ici (Feature Vector)
        embedding = F.relu(self.fc2(x))
        
        if return_embedding:
            return embedding
            
        x = self.dropout(embedding)
        out = self.fc3(x)
        return out

# ============================================================================
# 2. DATASET (GTZAN)
# ============================================================================
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=28):
        self.original_data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr; self.n_mels = n_mels; self.duration = duration
        
        # Mapping initial
        cat = pd.Categorical(self.original_data['label'])
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}

        self.cached_specs = []
        self.data = self.load_data_to_cache() 
        
        # Mise à jour des labels basés sur les données valides
        cat = pd.Categorical(self.data['label'])
        self.labels = cat.codes
        self.label_to_index = {cat.categories[i]: i for i in range(len(cat.categories))}
        self.index_to_label = {i: cat.categories[i] for i in range(len(cat.categories))}

    def __len__(self): return len(self.cached_specs)

    def load_data_to_cache(self):
        print("⏳ Mise en cache des spectrogrammes...")
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
        print(f"✅ Terminé : {len(self.cached_specs)} fichiers.")
        return self.original_data.iloc[valid_indices].reset_index(drop=True)

    def __getitem__(self, idx): return self.cached_specs[idx]

# ============================================================================
# 3. SPLIT
# ============================================================================
def stratified_split(dataset, test_size=0.15, val_size=0.15, random_state=42):
    df = dataset.data
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=df['label'], random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, stratify=df.loc[train_val_idx, 'label'], random_state=random_state
    )
    return train_idx, val_idx, test_idx

# ============================================================================
# 4. VISUALISATION T-SNE
# ============================================================================
def visualize_tsne(model_path, csv_file, audio_dir, batch_size=16, num_samples=300, random_state=42):
    print("\n🎨 Préparation de la visualisation t-SNE...")
    
    # 1. Charger dataset
    dataset = AudioDataset(csv_file, audio_dir, duration=28)

    # 2. Récupérer les indices de TEST
    _, _, test_idx = stratified_split(dataset, random_state=random_state)
    
    # Si on demande plus d'échantillons qu'il n'y en a dans le test, on prend tout le test
    actual_samples = min(num_samples, len(test_idx))
    print(f"📊 Utilisation de {actual_samples} échantillons du Test Set.")
    
    # DataLoader sur le sous-ensemble de test
    test_subset = Subset(dataset, test_idx[:actual_samples])
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # 3. Charger modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_to_index)
    
    model = TemporalCNN(num_genres=num_classes).to(device)
    
    # Chargement sécurisé des poids
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable à {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Extraire embeddings
    print("🚀 Extraction des embeddings...")
    X_list, y_list = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            # Appel avec return_embedding=True
            embeddings = model(X, return_embedding=True)
            
            X_list.append(embeddings.cpu().numpy())
            y_list.append(y.numpy())

    X_arr = np.concatenate(X_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)

    # 5. Calcul t-SNE
    
    print("🧮 Calcul du t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    X_2d = tsne.fit_transform(X_arr)

    # 6. Affichage
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(y_arr)
    cmap = plt.get_cmap('tab10')
    
    for i, lbl_idx in enumerate(unique_labels):
        idxs = y_arr == lbl_idx
        label_name = dataset.index_to_label[lbl_idx]
        plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], label=label_name, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")
    plt.title("Projection t-SNE des Embeddings CNN (Monotâche - Test Set)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("CNN/tsne_genre", exist_ok=True)
    save_path = "CNN/tsne_genre/tsne_cnn_embeddings_test.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graphique sauvegardé : {save_path}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Assurez-vous que ce chemin correspond bien à celui sauvegardé par votre script d'entraînement précédent
    # Note : Le script d'entraînement précédent utilisait "CNN/weights/temporal_cnn_gtzan_best.pth"
    MODEL_PATH = "CNN/weights/temporal_cnn_gtzan_best.pth"
    CSV_FILE = 'Data/features_30_sec.csv'
    AUDIO_DIR = 'Data/genres_original/'
    
    if os.path.exists(CSV_FILE) and os.path.exists(AUDIO_DIR):
        visualize_tsne(MODEL_PATH, CSV_FILE, AUDIO_DIR, num_samples=300)
    else:
        print("❌ Erreur : Données introuvables.")