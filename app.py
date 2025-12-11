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
    ["🏠 Accueil", "📄 Tableau CSV", "🖼️ Images d'entraînement","🖼️ TSNE plot"]
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
    st.title("📄 Visualisation du CSV")

    # Charger CSV
    df = pd.read_csv("emissions.csv")

    # Colonnes utiles (modifie ici si tu veux)
    columns_to_keep = [
        "timestamp", "project_name", "duration",
        "emissions", "gpu_model", "energy_consumed"
    ]

    st.subheader("Tableau filtré")
    st.dataframe(df[columns_to_keep], use_container_width=True)


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
