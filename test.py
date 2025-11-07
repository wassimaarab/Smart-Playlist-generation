from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
from torchsummary import summary

# Supprimer les warnings de librosa sur PySoundFile / audioread
warnings.filterwarnings("ignore")

# --- Dataset avec spectrogrammes ---
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sr=22050, n_mels=128, duration=3):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.labels = pd.Categorical(self.data['label']).codes
        self.label_to_index = dict(zip(self.data['label'], self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_str = self.data.loc[idx, 'label']
        filename = self.data.loc[idx, 'filename']
        path = os.path.join(self.audio_dir, label_str, filename)

        # Charger audio avec gestion des erreurs
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
            if y.size == 0:
                raise ValueError("Audio vide")
        except Exception:
            # Choisir un autre fichier au hasard
            new_idx = np.random.randint(0, len(self.data))
            return self.__getitem__(new_idx)

        # Spectrogramme Mel
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        #dim 128,130
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Normalisation robuste
        denom = S_dB.max() - S_dB.min()
        if denom == 0:
            new_idx = np.random.randint(0, len(self.data))
            return self.__getitem__(new_idx)
        S_norm = (S_dB - S_dB.min()) / denom

        X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.label_to_index[label_str], dtype=torch.long)
        return X, y


# --- Paramètres ---
csv_file = './Data/features_30_sec.csv'
audio_dir = './Data/genres_original/'
dataset = AudioDataset(csv_file, audio_dir)

batch_size = 16
lr = 0.001
epochs = 10
# Split train/val 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)



#CNN part

model = nn.Sequential(
    nn.Conv2d(1,128,kernel_size=(4,1),padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(4)),
    nn.Conv2d(128,256,kernel_size=(4,1),padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2)),
    nn.Conv2d(256,256,kernel_size=(4,1),padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2)),
    nn.Conv2d(256,512,kernel_size=(4,1),padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((3,1)),
    nn.Flatten(),
    nn.Linear(512*3,2048),
    nn.ReLU(),
    nn.Linear(2048,2048),
    nn.ReLU(),
    nn.Linear(2048,10)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = model.to(device)  # <- déplace le modèle sur GPU

#summary(model, (1,128,130))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)  # Forward
        loss = criterion(outputs, y)  # Loss
        loss.backward()  # Backprop
        optimizer.step()  # Update weights

        # Statistics
        running_loss += loss.item() * X.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    # Validation
    model.eval()
    val_corrects = 0
    val_total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs_val = model(X_val)
            preds_val = outputs_val.argmax(dim=1)
            val_corrects += (preds_val == y_val).sum().item()
            val_total += y_val.size(0)
    val_acc = val_corrects / val_total

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

#TSNE Visualization

def visualize_tsne(dataset, num_samples=900):
    X_list, y_list = [] , []
    count = 0
    for batch_X, batch_y in dataset:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #print(batch_X.shape)
        #outputs = model(batch_X)
        with torch.no_grad():   
            features = model[:-3](batch_X)  # Exclude last 3 layers for feature extraction

        for i in range(batch_X.size(0)):
            X_list.append(features[i].flatten().cpu().numpy())
            y_list.append(batch_y[i].cpu().numpy())
            count+=1
            if count >= num_samples:
                break
        if count >= num_samples:
            break
    
    X = np.array(X_list)
    y = np.array(y_list)
    

    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    X_2d = tsne.fit_transform(X)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10')
    plt.title("Projection t-SNE en 2D")
    plt.show()

visualize_tsne(val_loader)