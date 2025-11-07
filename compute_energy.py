import pandas as pd

# Charger le log
df = pd.read_csv("gpu_power_log.csv", skiprows=1, names=["timestamp", "power"])
# Nettoyage
df["power"] = df["power"].str.replace(" W", "").astype(float)

# Durée totale (en secondes)
duration_s = len(df)  # si tu logues 1 mesure par seconde

# Énergie totale (en Wh)
energy_Wh = df["power"].mean() * (duration_s / 3600)

print(f"Durée totale : {duration_s/60:.1f} minutes")
print(f"Puissance moyenne : {df['power'].mean():.2f} W")
print(f"Énergie totale : {energy_Wh:.3f} Wh  ({energy_Wh/1000:.4f} kWh)")

#utiliser la commande suivante pour logger la puissance GPU pendant l'entraînement
#nvidia-smi --query-gpu=power.draw --format=csv -l 1 > gpu_power_log.csv