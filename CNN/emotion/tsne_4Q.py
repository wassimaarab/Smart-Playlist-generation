import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 1. Dataset (Identique à l'entraînement)
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

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path, label, fname = self.file_list[idx]
        clean_id = os.path.splitext(fname)[0]
        cache_path = os.path.join(self.cache_dir, f"{clean_id}_{label}.pt")
        
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), torch.tensor(label)
            except: pass
        
        # Retourne un tenseur vide si le cache est absent (mode visu uniquement)
        return torch.zeros(1, 128, self.target_len), torch.tensor(label)

# ----------------------------
# 2. Modèle (Structure EXACTE de ton entraînement)
# ----------------------------
class TemporalCNN(nn.Module):
    def __init__(self, num_classes):
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
        # FC Shared (512*3 -> 2048)
        self.fc1 = nn.Linear(512*3, 2048)
        
        # Partie qui était dans la "tête genre" (2048 -> 1024 -> num_classes)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_embedding=False):
        # 1. Convolutions
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        
        # 2. Global Statistics Pooling
        mean = x.mean(dim=2)
        max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2)
        
        # 3. Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # On capture l'embedding (sortie de la couche 1024 avant le dernier dropout)
        embedding = F.relu(self.fc2(x))
        
        if return_embedding:
            return embedding
        
        x = self.dropout(embedding)
        out = self.fc3(x)
        return out

# ----------------------------
# 3. Script de Visualisation
# ----------------------------
def run_visu(dataset_dir, model_path):
    print(f"📂 Chargement du dataset : {dataset_dir}")
    dataset = FourQDataset(dataset_dir)
    lbl = [i[1] for i in dataset.file_list]
    
    # Recréation du split Test (Seed 42)
    idx_full = np.arange(len(dataset))
    _, test_idx = train_test_split(idx_full, test_size=0.15, stratify=lbl, random_state=42)
    
    loader = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device : {device}")
    
    # Chargement Modèle
    model = TemporalCNN(num_classes=len(dataset.classes)).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable ({model_path})")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Poids chargés avec succès.")
    except Exception as e:
        print(f"❌ Erreur de chargement des poids :\n{e}")
        return

    # Inférence
    print("🔮 Calcul des prédictions...")
    preds, labels, embs = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            
            # Prédiction standard
            out = model(X)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
            
            # Extraction Embedding (pour t-SNE)
            emb_vec = model(X, return_embedding=True)
            embs.extend(emb_vec.cpu().numpy())
    
    # --- RÉSULTATS ---
    acc = accuracy_score(labels, preds)
    print(f"\n🏆 Accuracy Test Set : {acc:.4f} ({acc*100:.2f}%)")
    
    # 1. Matrice de Confusion
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title(f"Confusion Matrix (Acc: {acc:.2f})")
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.show()
    
    # 2. t-SNE
    print("🎨 Calcul du t-SNE...")
    n_samples = len(embs)
    perp = min(20, n_samples - 1) if n_samples > 1 else 1
    
    # Remplace init='pca' par init='random'
    tsne = TSNE(n_components=2, perplexity=perp, init='random', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(np.array(embs))
    
    plt.figure(figsize=(10, 8))
    # Couleurs fixes pour Q1, Q2, Q3, Q4 (ajuster selon l'ordre alphabétique de vos dossiers)
    colors = ['red', 'orange', 'blue', 'green'] 
    
    for i, class_name in enumerate(dataset.classes):
        mask = np.array(labels) == i
        if np.sum(mask) > 0:
            c = colors[i % len(colors)]
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                        label=class_name, color=c, alpha=0.7, edgecolors='black', linewidth=0.3)
            
    plt.title(f"t-SNE Embeddings (Test Set) - Acc: {acc:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "CNN/tsne_genre/tsne_4Q_final.png"
    if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graphique t-SNE sauvegardé : {save_path}")
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "4Q/MER Audio"
    MODEL_FILE = "CNN/weights_4q/best_model_4q.pth"
    
    run_visu(DATA_DIR, MODEL_FILE)