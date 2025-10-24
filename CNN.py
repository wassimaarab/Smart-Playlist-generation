import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

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

# Split train/test 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- CNN adapté à spectrogramme ---
class TemporalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(4), padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(4), padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(4), padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4), padding=1)

        self.pool4x = nn.MaxPool2d(kernel_size=(4))
        self.pool2x = nn.MaxPool2d(kernel_size=(2))

        self.fc1 = nn.Linear(512 * 3, 2048) 
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool4x(x)  
        
        x = F.relu(self.conv2(x))
        x = self.pool2x(x)  
        
        x = F.relu(self.conv3(x))
        x = self.pool2x(x)  
        
        x = F.relu(self.conv4(x)) 
        

        mean = x.mean(dim=2) 
        max_ = x.max(dim=2).values  
        l2 = torch.sqrt((x ** 2).sum(dim=2)) 
        
        x = torch.cat([mean, max_, l2], dim=1) 
        x = x.mean(dim=2)  
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# --- Entraînement ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(pd.read_csv(csv_file)['label'].unique())
model = TemporalCNN(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_correct += (outputs.argmax(1) == y).sum().item()
        total_samples += y.size(0)
    print(f"Epoch {epoch+1}: Train Loss {total_loss/total_samples:.4f}, Train Acc {total_correct/total_samples:.4f}")

# --- Évaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
print(f"Test Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
