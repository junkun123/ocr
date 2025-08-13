import os
import csv

# Ruta de la carpeta con imágenes
train_dir = "dataset/train"

# Ruta de salida del CSV
csv_path = "train_labels.csv"

# Abrir el archivo CSV para escribir
with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "text"])  # Encabezados

    # Recorrer todos los archivos en la carpeta train
    for filename in sorted(os.listdir(train_dir)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            writer.writerow([filename, ""])  # Nombre de archivo, texto vacío

print(f"✅ CSV generado: {csv_path}")
