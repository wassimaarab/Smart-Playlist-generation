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

# 1. Dataset (Copie conforme)
class FourQDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=128, duration=30, cache_dir="cache_4q"):
        self.root_dir = root_dir; self.sr = sr; self.n_mels = n_mels; self.duration = duration; self.cache_dir = cache_dir
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.file_list = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.mp3'): self.file_list.append((os.path.join(cls_dir, f), self.class_to_idx[cls], f))
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
        # Fallback empty for viz if cache missing (should not happen if trained)
        return torch.zeros(1, 128, self.target_len), torch.tensor(label)

# 2. Modèle STANDARD (Copie conforme)
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
        x = F.relu(self.conv1(x)); x = self.pool4x(x)
        x = F.relu(self.conv2(x)); x = self.pool2x(x)
        x = F.relu(self.conv3(x)); x = self.pool2x(x)
        x = F.relu(self.conv4(x))
        mean = x.mean(dim=2); max_ = x.max(dim=2).values; l2 = torch.sqrt((x**2).sum(dim=2))
        x = torch.cat([mean, max_, l2], dim=1).mean(dim=2)
        x = F.relu(self.fc1(x))
        emb = F.relu(self.fc2(x))
        if return_embedding: return emb
        return self.fc3(emb)

# 3. Visu Logic
def run_visu(dataset_dir, model_path):
    print(f"Chargement {dataset_dir}...")
    dataset = FourQDataset(dataset_dir)
    lbl = [i[1] for i in dataset.file_list]
    
    # Split Test
    _, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.15, stratify=lbl, random_state=42)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCNN(len(dataset.classes)).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return

    print("Calcul Prédictions...")
    preds, labels, embs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds.extend(model(X).argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            embs.extend(model(X, True).cpu().numpy())
            
    # Matrix
    cm = confusion_matrix(labels, preds)
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
    plt.title("Confusion Matrix (4Q)")
    plt.show()
    
    # t-SNE
    print("Calcul t-SNE...")
    perp = 15#30 if len(embs) > 100 else 10
    X2 = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42).fit_transform(np.array(embs))
    
    plt.figure(figsize=(8,6))
    colors = ['red', 'orange', 'blue', 'green'] # Q1, Q2, Q3, Q4 approx
    for i, c in enumerate(dataset.classes):
        m = np.array(labels) == i
        plt.scatter(X2[m,0], X2[m,1], label=c, color=colors[i], alpha=0.7)
    plt.legend(); plt.title("t-SNE 4Q"); plt.savefig("CNN/tsne_genre/tsne_4Q.png", dpi=300);plt.show()

if __name__ == "__main__":
    DATA_DIR = "4Q/MER Audio"
    MODEL_FILE = "CNN/weights_4q/model_4q.pth"
    
    if os.path.exists(DATA_DIR) and os.path.exists(MODEL_FILE):
        run_visu(DATA_DIR, MODEL_FILE)
    else:
        print("❌ Erreur chemins.")