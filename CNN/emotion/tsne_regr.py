import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 1. CLASSES (Doivent être identiques à l'entraînement)
# ============================================================================
class RegressionDataset(Dataset):
    def __init__(self, root_dir, dataframe, cache_dir="cache_reg"):
        self.root_dir = os.path.join(root_dir, "audio")
        self.df = dataframe.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.target_len = 1292

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        target = torch.tensor([row['valence'] / 10.0, row['arousal'] / 10.0], dtype=torch.float32)
        cache_path = os.path.join(self.cache_dir, f"{fname}.pt")
        if os.path.exists(cache_path):
            try: return torch.load(cache_path), target
            except: pass
        return torch.zeros(1, 128, self.target_len), target

class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (1,4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (1,4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (1,4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (1,4), padding=1)
        self.bn1 = nn.BatchNorm2d(128); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(512)
        self.pool4x = nn.MaxPool2d(4); self.pool2x = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_embedding=False):
        x = self.pool4x(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2x(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2x(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) 
        mean = x.mean(dim=2); max_ = x.max(dim=2).values
        l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2)
        x = F.relu(self.fc1(x))
        emb = F.relu(self.fc2(x))
        if return_embedding: return emb
        return self.fc3(emb)

# ============================================================================
# 2. EVALUATION COMPLÈTE (Métriques + t-SNE multiples)
# ============================================================================
def evaluate_with_tsne():
    ROOT_DIR = "DEAM_Regression"
    # Utilise le bon fichier selon ton entraînement précédent
    MODEL_PATH = "CNN/weights_balanced/best_reg_model_da.pth" 
    CSV_PATH = os.path.join(ROOT_DIR, "metadata.csv")
    
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "CNN/weights_balanced/best_reg_model.pth"
        if not os.path.exists(MODEL_PATH):
            print("❌ Aucun modèle trouvé."); return

    print(f"⚖️ Chargement du modèle : {os.path.basename(MODEL_PATH)}")

    # 1. Charger Test Set
    df = pd.read_csv(CSV_PATH)
    idx = np.arange(len(df))
    _, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    test_ds = RegressionDataset(ROOT_DIR, df.iloc[test_idx])
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 2. Charger Modèle
    model = RegressionCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Prédictions & Embeddings
    print("🚀 Calcul des prédictions et extraction des embeddings...")
    preds_val, preds_aro = [], []
    targs_val, targs_aro = [], []
    all_embs = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            
            # A. On récupère l'embedding (l'avant-dernière couche)
            emb = model(X, return_embedding=True)
            all_embs.extend(emb.cpu().numpy())
            
            # B. On passe l'embedding dans la dernière couche pour avoir la prédiction
            out = model.fc3(emb).cpu().numpy()
            y = y.numpy()
            
            preds_val.extend(out[:, 0]); preds_aro.extend(out[:, 1])
            targs_val.extend(y[:, 0]); targs_aro.extend(y[:, 1])

    # 4. Métriques
    r2_v = r2_score(targs_val, preds_val); r2_a = r2_score(targs_aro, preds_aro)
    print(f"\n📊 RÉSULTATS : Valence R2={r2_v:.3f} | Arousal R2={r2_a:.3f}")

    # 5. Plot Scatter (R2)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(targs_val, preds_val, alpha=0.5, color='blue', s=20)
    plt.plot([0, 1], [0, 1], 'k--'); plt.title(f"VALENCE (R2={r2_v:.2f})"); plt.xlim(0,1); plt.ylim(0,1)
    plt.subplot(1, 2, 2)
    plt.scatter(targs_aro, preds_aro, alpha=0.5, color='red', s=20)
    plt.plot([0, 1], [0, 1], 'k--'); plt.title(f"AROUSAL (R2={r2_a:.2f})"); plt.xlim(0,1); plt.ylim(0,1)
    plt.savefig("CNN/tsne_reg/scatter_separated.png")
    plt.show()

    # 6. Calcul t-SNE
    print("🎨 Calcul du t-SNE (Cela peut prendre quelques secondes)...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    X_2d = tsne.fit_transform(np.array(all_embs))

    # 7. Affichage des 3 t-SNE
    plt.figure(figsize=(18, 5))
    
    # A. t-SNE coloré par VALENCE (Continu)
    plt.subplot(1, 3, 1)
    # cmap='plasma' : bleu foncé (bas) -> jaune brillant (haut)
    sc1 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=targs_val, cmap='plasma', alpha=0.7, s=20)
    plt.colorbar(sc1, label='Négatif <-> Positif'); plt.title("t-SNE : Valence (Positivité)")
    
    # B. t-SNE coloré par AROUSAL (Continu)
    plt.subplot(1, 3, 2)
    # cmap='viridis' : violet (bas) -> jaune/vert (haut)
    sc2 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=targs_aro, cmap='viridis', alpha=0.7, s=20)
    plt.colorbar(sc2, label='Calme <-> Énergique'); plt.title("t-SNE : Arousal (Énergie)")
    
    # C. t-SNE coloré par QUADRANTS (Discret)
    plt.subplot(1, 3, 3)
    med_v = np.median(targs_val); med_a = np.median(targs_aro)
    labels, colors = [], []
    for i in range(len(targs_val)):
        v, a = targs_val[i], targs_aro[i]
        if v >= med_v and a >= med_a: l="Q1 Happy"; c="red"
        elif v < med_v and a >= med_a: l="Q2 Angry"; c="orange"
        elif v < med_v and a < med_a: l="Q3 Sad"; c="blue"
        else: l="Q4 Relax"; c="green"
        labels.append(l); colors.append(c)
        
    unique_labels = sorted(list(set(labels)))
    for ul in unique_labels:
        mask = np.array(labels) == ul
        c = colors[np.where(mask)[0][0]]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=c, label=ul, alpha=0.6, s=20)
    plt.legend(); plt.title("t-SNE : Quadrants Emotionnels")

    plt.tight_layout()
    plt.savefig("CNN/tsne_reg/tsne_triple_view.png")
    print("✅ Terminé ! Graphes sauvegardés dans CNN/tsne_reg/")
    plt.show()

if __name__ == "__main__":
    evaluate_with_tsne()