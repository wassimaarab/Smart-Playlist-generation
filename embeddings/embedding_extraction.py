import torch
import librosa
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from muq import MuQ

# --- 1. Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 24000


# --- 2. Charger le modèle MuQ ---
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

# --- 3. Fonction d’extraction d’embedding ---
def extract_embedding(file_path):
    try:
        wav, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"⚠️ Erreur lecture {file_path}: {e}")
        return None  # On retourne rien pour ce fichier

    wavs = torch.tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        output = muq(wavs)
    emb = output.last_hidden_state.squeeze(0).mean(dim=0)
    return emb.cpu().numpy()
 
if __name__ == "__main__":
    data_dir = "Data/genres_original"
    # --- 4. boucle sur tous les fichiers ---
    embeddings = []
    genres = []
    filenames = []

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        for file in tqdm(files, desc=f"Traitement {genre}"):
            file_path = os.path.join(genre_path, file)
            emb = extract_embedding(file_path)
            if emb is None:
                continue
            embeddings.append(emb)
            genres.append(genre)
            filenames.append(file)

            # 🔹 LOG après chaque audio traité
            print(f"✅ Fichier traité : {file} | Genre : {genre} | Embedding : {emb.shape}")


    # --- 5. Sauvegarde ---
    embeddings = np.array(embeddings)
    df = pd.DataFrame(embeddings)
    df['genre'] = genres
    df['filename'] = filenames

    # En CSV (lisible avec pandas) :
    df.to_csv("muq_embedding_GTZAN.csv", index=False)

    # Ou en format PyTorch pour entraînement direct :
    torch.save({'embeddings': torch.tensor(embeddings), 'genres': genres, 'filenames': filenames}, "muq_embeddings.pt")

    print("✅ Extraction terminée !")
    print("Dimensions embeddings :", embeddings.shape)  # (N, D)

