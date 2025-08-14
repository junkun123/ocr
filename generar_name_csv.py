import os
import csv

# Ruta a las imágenes de entrenamiento
train_dir = "dataset/train"

# CSV de salida
csv_file = "dataset/train_labels.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "text"])  # Cambié filename por image_path

    for img_name in os.listdir(train_dir):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(train_dir, img_name)  # Ruta completa
            # Aquí iría el texto real asociado a esa imagen
            texto = "texto ejemplo"
            writer.writerow([image_path, texto])

print(f"CSV generado: {csv_file}")
