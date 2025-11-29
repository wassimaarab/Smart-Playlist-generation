import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 1. Dataset (Identique)
# -----------------------------------------------------------
class VisuDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=128, duration=30, cache_dir="cache_visu"):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.cache_dir = cache_dir
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Dossier {root_dir} introuvable.")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.file_list = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.mp3', '.wav')):
                    path = os.path.join(class_dir, fname)
                    song_id = os.path.splitext(fname)[0]
                    self.file_list.append((path, self.class_to_idx[cls], song_id))
        
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        self.target_len = int(np.ceil(self.duration * self.sr / 512))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path, label_idx, song_id = self.file_list[idx]
        
        safe_id = "".join([c for c in song_id if c.isalnum()])
        cache_path = os.path.join(self.cache_dir, f"{safe_id}_{label_idx}.pt")

        if os.path.exists(cache_path):
            try: return torch.load(cache_path), torch.tensor(label_idx)
            except: pass
        
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
            target_samples = int(self.sr * self.duration)
            if len(y) < target_samples: y = np.pad(y, (0, target_samples - len(y)))
            else: y = y[:target_samples]

            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels), ref=np.max)
            S = (S - S.min()) / (S.max() - S.min() + 1e-6)
            
            if S.shape[1] < self.target_len:
                S = np.pad(S, ((0, 0), (0, self.target_len - S.shape[1])))
            else: S = S[:, :self.target_len]

            X = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
            torch.save(X, cache_path)
            return X, torch.tensor(label_idx)
        except:
            return torch.zeros(1, 128, self.target_len), torch.tensor(label_idx)

# -----------------------------------------------------------
# 2. Modèle (CORRIGÉ POUR MATCHER L'ENTRAINEMENT)
# -----------------------------------------------------------
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
        
    def forward(self, x, return_embedding=False):
        # 1. Feature Extraction (Convolutions)
        x = F.relu(self.conv1(x)); x = self.pool4(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = F.relu(self.conv4(x)) # Output: (Batch, 512, Freq=8, Time=78)
        
        # 2. Aggregation Fréquentielle
        # On écrase la dimension 2 (Freq) pour garder (Batch, 512, Time)
        mean = x.mean(dim=2)
        max_ = x.max(dim=2)[0]
        l2 = torch.sqrt((x**2).sum(dim=2))
        
        # 3. Concaténation
        x = torch.cat([mean, max_, l2], dim=1) # Output: (Batch, 1536, Time)
        
        # 4. Aggregation Temporelle (C'ÉTAIT L'ERREUR !)
        # Il faut écraser la dimension Time avant de passer dans FC1
        x = x.mean(dim=2) # Output: (Batch, 1536)
        
        # 5. Classifier
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        
        if return_embedding:
            return embedding
            
        out = self.fc3(embedding)
        return out

# -----------------------------------------------------------
# 3. Main Visu
# -----------------------------------------------------------
def run_visualization(model_path, dataset_dir):
    print(f"📂 Chargement Dataset depuis {dataset_dir}...")
    dataset = VisuDataset(dataset_dir)
    
    # Split Test
    idx = np.arange(len(dataset))
    labels = [dataset.file_list[i][1] for i in idx]
    _, test_idx = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)
    
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)
    class_names = dataset.classes
    print(f"Classes : {class_names}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCNN(num_classes=len(class_names)).to(device)
    
    print(f"⚖️ Chargement poids : {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    print("🧠 Extraction des embeddings...")
    all_preds, all_labels, all_embeddings = [], [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            emb = model(X, return_embedding=True)
            all_embeddings.extend(emb.cpu().numpy())

    # Matrice de Confusion
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion (Test Set)')
    plt.ylabel('Vrai Label')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.show()

    # t-SNE
    X_tsne = np.array(all_embeddings)
    y_tsne = np.array(all_labels)
    
    perp = 25 if len(X_tsne) > 100 else 10
    print(f"🎨 Calcul t-SNE ({len(X_tsne)} points)...")
    
    tsne = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X_tsne)
    
    plt.figure(figsize=(10, 8))
    # Couleurs distinctes
    colors = ['red', 'orange', 'blue', 'green'] # angry, happy, relax, sad
    
    for i, cls in enumerate(class_names):
        mask = y_tsne == i
        c = colors[i] if i < len(colors) else None
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cls, color=c, alpha=0.7)
    
    plt.legend()
    plt.title("Espace Latent (t-SNE) - Dataset Équilibré")
    plt.savefig("CNN/tsne_plot/tsne_cnn_embeddings_deam.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "DEAM_Balanced" 
    MODEL_FILE = "CNN/weights_balanced/final_model.pth"
    
    if os.path.exists(DATA_DIR) and os.path.exists(MODEL_FILE):
        run_visualization(MODEL_FILE, DATA_DIR)
    else:
        print("❌ Erreur de chemins.")