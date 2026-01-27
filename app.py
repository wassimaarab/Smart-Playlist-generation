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
# -----------------------
#       PLAYLIST AUDIO
# -----------------------
elif choice == "🎵 Playlist Audio":
    # --- CSS PERSONNALISÉ ---
    st.markdown("""
        <style>
        /* Supprimer les marges par défaut pour un look app */
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        
        /* Style de la carte 'Now Playing' */
        .now-playing-card {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
            border: 1px solid #333;
        }
        
        /* Style de la fausse pochette d'album */
        .album-art {
            width: 100%;
            aspect-ratio: 1/1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 4rem;
            margin-bottom: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        /* Style de la liste des chansons */
        .song-row {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: background 0.2s;
            border: 1px solid transparent;
        }
        .song-row:hover {
            background-color: #2C2C2C;
            border: 1px solid #444;
        }
        .active-song {
            background-color: #262626;
            border-left: 4px solid #764ba2;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎧 Studio Session")
    st.markdown("Parcourez les échantillons audio générés par le modèle.")

    AUDIO_DIR = "Exported_Playlists/Metal_Journey/"
    # On cherche les WAV et les MP3, on les additionne (+), puis on trie le tout
    
    # Sécurité si le dossier n'existe pas
    if not os.path.isdir(AUDIO_DIR):
        st.error(f"Le dossier `{AUDIO_DIR}` est introuvable.")
    else:
        audio_files = sorted(glob(os.path.join(AUDIO_DIR, "*.wav")) + glob(os.path.join(AUDIO_DIR, "*.mp3")))

        if not audio_files:
            st.warning("Aucun fichier .wav trouvé dans le dossier.")
        else:
            # Gestion de l'état (Session State)
            if 'current_track' not in st.session_state:
                st.session_state.current_track = audio_files[0]

            # --- MISE EN PAGE : 2 COLONNES (Liste | Lecteur) ---
            col_list, col_player = st.columns([2, 1], gap="large")

            # --- COLONNE DE DROITE : LE LECTEUR (NOW PLAYING) ---
            with col_player:
                st.markdown("### En lecture")
                
                current_filename = os.path.basename(st.session_state.current_track)
                
                # Conteneur visuel (Card)
                with st.container():
                    # Simulation pochette
                    st.markdown(f"""
                        <div class="now-playing-card">
                            <div class="album-art">🎵</div>
                            <h3 style="margin: 0; color: white;">{current_filename}</h3>
                            <p style="color: #888; font-size: 0.9em;">Génération AI • Blues Model</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Spacer
                    
                    # Le lecteur audio natif Streamlit
                    # Note : on lit le fichier binaire pour éviter les problèmes de cache parfois
                    try:
                        st.audio(st.session_state.current_track, format="audio/wav")
                    except Exception as e:
                        st.error(f"Erreur de lecture : {e}")

                    # Bouton de téléchargement
                    with open(st.session_state.current_track, "rb") as file:
                        st.download_button(
                            label="⬇️ Télécharger le titre",
                            data=file,
                            file_name=current_filename,
                            mime="audio/wav",
                            use_container_width=True
                        )

            # --- COLONNE DE GAUCHE : LA PLAYLIST ---
            with col_list:
                st.markdown("### Bibliothèque")
                
                # Création d'un conteneur avec scroll si la liste est longue
                with st.container():
                    for i, file_path in enumerate(audio_files):
                        filename = os.path.basename(file_path)
                        is_active = (file_path == st.session_state.current_track)
                        
                        # Création d'une ligne pour chaque chanson
                        # On utilise des colonnes pour aligner le texte et le bouton
                        c1, c2 = st.columns([0.85, 0.15])
                        
                        with c1:
                            # Visuel conditionnel (Gras si actif)
                            if is_active:
                                st.markdown(f"**🟢 {i+1}. {filename}**")
                            else:
                                st.markdown(f"{i+1}. {filename}")
                                
                        with c2:
                            # Le bouton Play
                            # On utilise une clé unique pour chaque bouton
                            if st.button("▶", key=f"play_{i}", use_container_width=True):
                                st.session_state.current_track = file_path
                                st.rerun()
                        
                        # Petit séparateur visuel discret
                        st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)