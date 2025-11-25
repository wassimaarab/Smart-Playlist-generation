
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. DATASET CUSTOM
# ============================================
class MuQEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ============================================
# 2. MODÈLE DE CLASSIFICATION
# ============================================
class MuQClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256]):
        super(MuQClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Couches cachées avec Dropout et BatchNorm
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

# ============================================
# 3. CHARGEMENT DES DONNÉES
# ============================================
def load_data(csv_path="embeddings/muq_embeddings.csv",label_column='genre'):
    """Charge les embeddings depuis le CSV"""
    df = pd.read_csv(csv_path)
    
    # Séparer embeddings et labels
    embeddings = df.drop([label_column, 'filename'], axis=1).values
    genres = df[label_column].values

    # Encoder les labels
    le = LabelEncoder()
    labels = le.fit_transform(genres)
    
    print(f"📊 Données chargées : {len(embeddings)} échantillons")
    print(f"🎵 Classes : {le.classes_}")
    print(f"📐 Dimension embeddings : {embeddings.shape[1]}")
    
    return embeddings, labels, le

# ============================================
# 4. ENTRAÎNEMENT
# ============================================
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001,model_path='models/best_muq_classifier.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=5)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # ---- VALIDATION ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        # Scheduler
        scheduler.step(val_acc)
        
        # History
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
        
        # Affichage
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
    
    return history

# ============================================
# 5. ÉVALUATION
# ============================================
def evaluate_model(model, test_loader, label_encoder, device, confusion_matrix_path='Viz/confusion_matrix.png'):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Rapport de classification
    print("\n RAPPORT DE CLASSIFICATION :")
    target_names = [str(c) for c in label_encoder.classes_]
    print(classification_report(all_labels, all_preds, 
                                target_names=target_names))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    print(f"Matrice de confusion sauvegardée : {confusion_matrix_path}")

# ============================================
# 6. PRÉDICTION SUR UN NOUVEAU FICHIER
# ============================================
def predict_genre(model, embedding, label_encoder, device):
    """Prédit le genre d'un embedding"""
    model.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(emb_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        pred_genre = label_encoder.inverse_transform([pred_idx])[0]
        
        return pred_genre, confidence


