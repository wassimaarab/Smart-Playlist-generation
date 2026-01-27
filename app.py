import streamlit as st
import pandas as pd
import os
from glob import glob

st.set_page_config(page_title="Dashboard entraînement", page_icon="📊", layout="wide")

# -----------------------
#       NAVIGATION
# -----------------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Aller à :",
    ["🏠 Accueil", "📄 Tableau CSV", "🖼️ Images d'entraînement","🖼️ TSNE plot","🎵 Playlist Audio"]
)

# -----------------------
#       ACCUEIL
# -----------------------
if choice == "🏠 Accueil":
    st.title("Bienvenue sur le Dashboard")
    st.write("Sélectionnez une section dans le menu de gauche :")
    st.markdown("""
    ### 📄 *Tableau CSV*
    Visualisation filtrée de vos données CodeCarbon.

    ### 🖼️ *Images d'entraînement*
    Parcourir toutes les courbes / images avec un slider.

    ### 🖼️ *TSNE plot*
    Parcourir toutes les tsne / images avec un slider.
    """)
    st.image("training_plot/training_curves_1.png", caption="Exemple", use_column_width=True)


# -----------------------
#       TABLEAU CSV
# -----------------------
elif choice == "📄 Tableau CSV":
    st.title("📄 Analyse des Émissions & Performance")

    # Charger CSV
    df = pd.read_csv("emissions.csv")
    
    # Nettoyage / Préparation rapide
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    columns_to_keep = ["timestamp", "project_name", "duration", "emissions", "gpu_model", "energy_consumed"]
    display_df = df[columns_to_keep].copy()

    # --- 1. INDICATEURS CLÉS (METRICS) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Émissions (kgCO2)", round(df["emissions"].sum(), 4))
    with col2:
        st.metric("Consommation Totale (kWh)", round(df["energy_consumed"].sum(), 2))
    with col3:
        st.metric("Durée Totale (s)", int(df["duration"].sum()))
    with col4:
        st.metric("Projets lancés", df["project_name"].nunique())

    st.divider()

    # --- 2. FILTRES INTERACTIFS ---
    st.subheader("🔍 Filtres et Données")
    
    c1, c2 = st.columns(2)
    with c1:
        projets = st.multiselect("Filtrer par projet", options=df["project_name"].unique())
    with c2:
        gpu = st.multiselect("Filtrer par GPU", options=df["gpu_model"].unique())

    # Application des filtres
    if projets:
        display_df = display_df[display_df["project_name"].isin(projets)]
    if gpu:
        display_df = display_df[display_df["gpu_model"].isin(gpu)]

    # --- 3. TABLEAU STYLISÉ ---
    # On colore les émissions élevées en rouge pour alerter
    st.dataframe(
        display_df.style.highlight_max(axis=0, subset=["emissions"], color="#ff4b4b66")
        .format({"emissions": "{:.4f}", "energy_consumed": "{:.2f}"}),
        use_container_width=True
    )

    # --- 4. GRAPHIQUE DE TENDANCE ---
    st.subheader("📈 Évolution de la consommation énergétique")
    st.line_chart(display_df, x="timestamp", y="energy_consumed")


# -----------------------
#       IMAGES AVEC SLIDER
# -----------------------
elif choice == "🖼️ Images d'entraînement":
    st.title("🖼️ Training Plots")

    IMAGE_DIR = "training_plot"
    extensions = ("*.png", "*.jpg", "*.jpeg")

    # Récupérer images
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(IMAGE_DIR, ext)))

    files = sorted(files)  # tri

    if len(files) == 0:
        st.error("⚠️ Aucune image trouvée dans training_plot/")
    else:
        st.subheader("Slider d'images")
        index = st.slider(
            "Sélectionner l'image",
            0,
            len(files) -1 ,
            0
        )

        img_path = files[index]
        filename = os.path.basename(img_path)

        st.image(img_path, caption=filename, use_container_width=True)

# -----------------------
#       IMAGES AVEC SLIDER
# -----------------------
elif choice == "🖼️ TSNE plot":
    st.title("🖼️ TSNE Plots")

    TSNE_DIR = "CNN/tsne_plot"
    extensions = ("*.png", "*.jpg", "*.jpeg")

    # Récupérer images
    files2 = []
    for ext in extensions:
        files2.extend(glob(os.path.join(TSNE_DIR, ext)))

    files2 = sorted(files2)  # tri

    if len(files2) == 0:
        st.error("⚠️ Aucune image trouvée dans training_plot/")
    else:
        st.subheader("Slider d'images")
        index = st.slider(
            "Sélectionner l'image",
            0,
            len(files2) -1 ,
            0
        )

        img_path = files2[index]
        filename = os.path.basename(img_path)

        st.image(img_path, caption=filename, use_container_width=True)

# -----------------------
#       PLAYLIST AUDIO
# -----------------------
elif choice == "🎵 Playlist Audio":
    st.markdown("""
        <style>
        /* Style global inspiré de l'image */
        .main { background-color: #121212; }
        .stApp { background: linear-gradient(180deg, #222222 0%, #121212 100%); }
        
        /* Conteneur de la playlist */
        .spotify-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
        }

        /* Ligne de chanson */
        .track-row {
            display: flex;
            align-items: center;
            padding: 10px;
            border-radius: 8px;
            transition: background 0.2s;
            margin-bottom: 5px;
        }
        .track-row:hover { background: rgba(255, 255, 255, 0.1); }
        
        .track-img {
            width: 50px;
            height: 50px;
            border-radius: 4px;
            margin-right: 15px;
            background-color: #282828;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .track-info { flex-grow: 1; }
        .track-name { color: white; font-weight: 600; font-size: 16px; margin: 0; }
        .track-artist { color: #b3b3b3; font-size: 13px; margin: 0; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Today's Top Hits")
    
    AUDIO_DIR = "Exported_Playlists/Metal_Journey/"
    # On cherche les WAV et les MP3, on les additionne (+), puis on trie le tout
    audio_files = sorted(glob(os.path.join(AUDIO_DIR, "*.wav")) + glob(os.path.join(AUDIO_DIR, "*.mp3")))

    if not audio_files:
        st.error("Aucun fichier audio trouvé.")
    else:
        # Initialisation du morceau courant
        if 'current_track' not in st.session_state:
            st.session_state.current_track = audio_files[0]

        # --- HEADER (Comme le haut du téléphone) ---
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            # On simule une pochette d'album
            st.markdown('<div style="background: linear-gradient(45deg, #1DB954, #191414); height: 200px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 80px;">🎵</div>', unsafe_allow_html=True)
        with col_h2:
            st.caption("PLAYLIST")
            st.header("Mes Entraînements")
            st.write(f"**{len(audio_files)} titres** • Échantillons générés")

        st.write("##") # Espace

        # --- LISTE DES TITRES ---
        for file_path in audio_files:
            filename = os.path.basename(file_path)
            
            # Création de la ligne "Spotify"
            col_info, col_btn = st.columns([5, 1])
            
            with col_info:
                st.markdown(f"""
                    <div class="track-row">
                        <div class="track-img">🔊</div>
                        <div class="track-info">
                            <p class="track-name">{filename}</p>
                            <p class="track-artist">Système Expert • Projet ENSEIRB</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_btn:
                st.write("") # Alignement vertical
                if st.button("▶️", key=file_path, help="Lire ce titre"):
                    st.session_state.current_track = file_path
                    st.rerun()

        # --- LECTEUR FIXE (BARRE DE LECTURE EN BAS) ---
        st.markdown(f"""
        <style>
        .player-bar {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(24, 24, 24, 0.98);
            border-top: 1px solid rgba(255,255,255,0.1);
            padding: 10px 20px;
            z-index: 9999;
        }}

        .player-content {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .player-track {{
            color: white;
            font-weight: 600;
        }}

        .player-artist {{
            color: #b3b3b3;
            font-size: 12px;
        }}
        </style>

        <div class="player-bar">
            <div class="player-content">
                <div>
                    <div class="player-track">{os.path.basename(st.session_state.current_track)}</div>
                    <div class="player-artist">Système Expert • Projet ENSEIRB</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.audio(st.session_state.current_track)


    st.markdown(
    "<div style='height: 120px;'></div>",
    unsafe_allow_html=True
    )