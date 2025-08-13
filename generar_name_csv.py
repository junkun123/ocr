import pandas as pd

# Ruta al CSV original con anotaciones
annot_csv = "annot.csv"

# Cargar el CSV original
df = pd.read_csv(annot_csv)

# Crear el nuevo DataFrame con formato filename, text
df_out = pd.DataFrame()
df_out["filename"] = df["id"] + ".jpg"  # El nombre del archivo final
df_out["text"] = df["utf8_string"]

# Guardar el nuevo CSV
df_out.to_csv("train_labels.csv", index=False)

print("✅ train_labels.csv generado correctamente.")
