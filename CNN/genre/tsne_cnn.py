import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from CNN import TemporalCNN, AudioDataset, stratified_split
import pandas as pd
from torch.utils.data import DataLoader, Subset

def visualize_tsne(model_path, csv_file, audio_dir, batch_size=16, num_samples=150, random_state=42):
    # Charger dataset
    dataset = AudioDataset(csv_file, audio_dir, duration=28)

    # Recréer le split pour récupérer les indices de test
    _, _, test_idx = stratified_split(dataset, random_state=random_state)

    # DataLoader uniquement sur le test
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    # Charger modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(pd.read_csv(csv_file)['label'].unique())
    model = TemporalCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extraire embeddings
    X_list, y_list = [], []
    count = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            embeddings = model(X, return_embedding=True)
        for i in range(X.size(0)):
            X_list.append(embeddings[i].cpu().numpy())
            y_list.append(y[i].cpu().numpy())
            count += 1
            if count >= num_samples:
                break
        if count >= num_samples:
            break

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_arr)

    plt.figure(figsize=(10,8))
    for lbl in np.unique(y_arr):
        idxs = y_arr == lbl
        plt.scatter(X_2d[idxs,0], X_2d[idxs,1], label=dataset.index_to_label[lbl], alpha=0.7)
    plt.legend()
    plt.title("Projection t-SNE en 2D des embeddings CNN (Test set)")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.savefig("CNN/tsne_genre/tsne_cnn_embeddings_test.png", dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model_path = "CNN/weights/temporal_cnn_gtzan.pth"
    csv_file = 'Data/features_30_sec.csv'
    audio_dir = 'Data/genres_original/'
    visualize_tsne(model_path, csv_file, audio_dir, num_samples=150)
