import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def load_embeddings(csv_path="fma_muq_embeddings.csv"):
    data = pd.read_csv(csv_path)
    genres = data['genre']
    filenames = data['track_id']
    data.drop(columns=['genre', 'track_id'], inplace=True)
    return data.to_numpy(), genres.to_numpy(), filenames.to_numpy()

def create_sphere(center, radius, resolution=20):
    """
    Create x, y, z coordinates for a sphere given a center and radius
    """
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return x, y, z

if __name__ == "__main__":
    embeddings, genres, filenames = load_embeddings()
    X_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings)
    
    df_plot = pd.DataFrame({
        'x': X_3d[:, 0],
        'y': X_3d[:, 1],
        'z': X_3d[:, 2],
        'genre': genres,
        'track_id': filenames
    })

    fig = go.Figure()

    unique_genres = np.unique(genres)
    colors = px.colors.qualitative.Plotly  # some colors for clusters

    # Scatter points per genre
    for i, genre in enumerate(unique_genres):
        genre_points = df_plot[df_plot['genre'] == genre]
        fig.add_trace(go.Scatter3d(
            x=genre_points['x'],
            y=genre_points['y'],
            z=genre_points['z'],
            mode='markers',
            marker=dict(size=3, color=colors[i % len(colors)]),
            name=genre,
            hovertext=genre_points['track_id']
        ))

        # Sphere around cluster
        centroid = genre_points[['x', 'y', 'z']].mean().to_numpy()
        distances = np.linalg.norm(genre_points[['x', 'y', 'z']].to_numpy() - centroid, axis=1)
        radius = distances.max()  # can also use mean or std for smaller sphere
        x_s, y_s, z_s = create_sphere(centroid, radius)
        fig.add_trace(go.Surface(
            x=x_s, y=y_s, z=z_s,
            opacity=0.2,
            showscale=False,
            surfacecolor=np.full_like(x_s, i),
            name=f"{genre} sphere",
            hoverinfo='skip',
            colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]]
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Dim 1',
            yaxis_title='t-SNE Dim 2',
            zaxis_title='t-SNE Dim 3'
        ),
        title="Interactive 3D t-SNE with Spheres Around Genre Clusters",
        legend_title="Genre"
    )

    fig.show()
