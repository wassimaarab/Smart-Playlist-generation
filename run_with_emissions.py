import subprocess
import argparse
from codecarbon import EmissionsTracker
import os

def main():
    parser = argparse.ArgumentParser(description="Exécuter un script avec suivi des émissions.")
    parser.add_argument("train_script", help="Le script d'entraînement à exécuter")
    parser.add_argument("--project_name", default="default_project",
                        help="Nom du projet CodeCarbon (optionnel)")
    args = parser.parse_args()

    # Vérification
    if not os.path.isfile(args.train_script):
        print(f"❌ Erreur : fichier '{args.train_script}' introuvable.")
        return

    print(f"🌱 Lancement du suivi d'émissions pour : {args.project_name}")
    
    tracker = EmissionsTracker(project_name=args.project_name)
    tracker.start()

    # Exécution du script d'entraînement
    subprocess.run(["python", args.train_script])

    # Fin du tracking
    emissions = tracker.stop()

    print("✅ Entraînement terminé !")
    print(f"🌍 Émissions totales : {emissions:.6f} kg CO₂eq")
    print("📁 Le CSV des émissions a été généré automatiquement.")

if __name__ == "__main__":
    main()
