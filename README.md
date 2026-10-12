# Music Playlist Generation with MuQ & CNN Embeddings

This project demonstrates automatic playlist generation based on **music similarity** using embeddings extracted from audio data. It combines **classification models**, **CNN-based embeddings**, and **multi-task learning** to recommend tracks according to emotion, genre, and instrumentation.

---


## Features

- **Audio Embeddings**: Extract MuQ embeddings from multiple datasets (Jamendo, NSynth, MIREX, 4Q).
- **Music Classification**: Predict emotion, genre, and instruments using CNN and multi-task models.
- **Playlist Generation**: Create playlists based on similarity in embedding space.
- **Visualization**: Explore embeddings with 2D/3D t-SNE plots.
- **Energy Monitoring**: Track GPU/CPU consumption during model training.

---

## USAGE

1. Run the main application:
   python app.py

2. Extract embeddings:
   embeddings/embedding_extraction_{dataset_name}.py

3. Train classification models:
   CNN approach : python CNN/{classification_type}/CNN.py
   MuQ approach : python Classification/{classificationType}_classification.py

4. Generate playlists:
   CNN approach : python CNN/multitache/generate_playlist.py
   MuQ approach : Classification/multi_classif_jamendo.ipynb

5. Visualize embeddings:
   python Viz/tsne_vis.py
   python Viz/tsne_vis_3d.py

6. Compute energy consumption:
   python consommation/compute_energy.py


## NOTES
- All scripts can be run individually.
- Playlists are exported in Exported_Playlists folder with a corresponding .m3u file.
- Embedding extraction and visualization scripts support multiple datasets.
- Energy tracking is optional and requires GPU access for accurate logs.

## Repport 
The repport contains all the details and the results.
