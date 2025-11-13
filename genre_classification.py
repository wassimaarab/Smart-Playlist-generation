from tete_classif import MuQEmbeddingDataset, MuQClassifier, train_model, evaluate_model,load_data
import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split

# ========================= GTZAN  :  GENRE CLASSIFICATION ========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device : {device}")

# Charger les données
embeddings, labels, label_encoder = load_data("muq_embedding_GTZAN.csv")

# Split train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    embeddings, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n Split des données :")
print(f"   Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# DataLoaders
train_dataset = MuQEmbeddingDataset(X_train, y_train)
val_dataset = MuQEmbeddingDataset(X_val, y_val)
test_dataset = MuQEmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Créer le modèle
input_dim = embeddings.shape[1]
num_classes = len(label_encoder.classes_)
model = MuQClassifier(input_dim, num_classes, hidden_dims=[512, 256, 128]).to(device)

print(f"\n Modèle créé : {sum(p.numel() for p in model.parameters())} paramètres")

# Entraînement
print("\n Début de l'entraînement...\n")
history = train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001)

# Charger le meilleur modèle
model.load_state_dict(torch.load('best_muq_classifier.pt'))

# Évaluation finale
print("\n Évaluation sur le test set :")
evaluate_model(model, test_loader, label_encoder, device)


print("\n Entraînement terminé !")
print(" Fichiers sauvegardés :")
print("   - best_muq_classifier.pt (modèle)")
print("   - confusion_matrix.png (visualisation)")