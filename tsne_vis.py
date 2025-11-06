import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE
def load_embeddings(csv_path="muq_embedding_GTZAN.csv"):
    """
    Load embedding vectors from a CSV file into a NumPy array.
    """
    
    data = pd.read_csv(csv_path)
    genres = data['genre']  #  the second last column is 'genre'
    filenames = data['filename']  #  the last column is 'filename'
    data.drop(columns=['genre', 'filename'], inplace=True)
    return data.to_numpy(), genres.to_numpy(), filenames.to_numpy()

if __name__ == "__main__":
    embeddings , genres , filenames = load_embeddings()
    X_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    print("Loaded embeddings shape:", embeddings.shape)
    print("t-SNE result shape:", X_2d.shape)

    plt.figure(figsize=(10, 8))

    # Obtenir la liste unique des genres
    unique_genres = np.unique(genres)

    # Couleurs : utiliser un colormap qui a assez de couleurs pour tous les genres
    cmap = plt.get_cmap('tab10')  # tab10 a 10 couleurs
    colors = {genre: cmap(i) for i, genre in enumerate(unique_genres)}

    # Plot genre par genre
    for genre in unique_genres:
        idx = genres == genre
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=genre, color=colors[genre], alpha=0.7, s=20)

    plt.title("t-SNE Visualization of MuQ Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Genre")
    plt.grid(True)

    # Sauvegarde du graphique
    plt.savefig("tsne_muQ_embeddings_GTZAN.png", dpi=300)
    print("✅ t-SNE plot saved as tsne_muQ_embeddings.png")

