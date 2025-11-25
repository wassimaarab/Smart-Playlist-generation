import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

def load_embeddings(csv_path="fma_muq_embeddings.csv"):
    """
    Load embedding vectors from a CSV file into a NumPy array.
    """
    data = pd.read_csv(csv_path)
    genres = data['genre']
    filenames = data['track_id']
    data.drop(columns=['genre', 'track_id'], inplace=True)
    return data.to_numpy(), genres.to_numpy(), filenames.to_numpy()

if __name__ == "__main__":
    embeddings, genres, filenames = load_embeddings()
    X_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings)
    print("Loaded embeddings shape:", embeddings.shape)
    print("t-SNE 3D result shape:", X_3d.shape)

    # Create a DataFrame for Plotly
    df_plot = pd.DataFrame({
        'x': X_3d[:, 0],
        'y': X_3d[:, 1],
        'z': X_3d[:, 2],
        'genre': genres,
        'track_id': filenames
    })

    # Interactive 3D scatter plot
    fig = px.scatter_3d(
        df_plot,
        x='x',
        y='y',
        z='z',
        color='genre',
        hover_data=['track_id'],
        opacity=0.7
    )

    fig.update_layout(
        title="Interactive t-SNE 3D Visualization of MuQ Embeddings",
        legend_title="Genre"
    )

    # Show the interactive plot in your browser
    fig.show()
