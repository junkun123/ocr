import os
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# ========================
# 1. Cargar CSVs
# ========================
train_df = pd.read_csv("dataset/train_labels.csv")
val_df = pd.read_csv("dataset/val_labels.csv")

# Asegurar que las rutas existen
for path in train_df["image_path"].tolist() + val_df["image_path"].tolist():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró: {path}")

# ========================
# 2. Cargar modelo y procesador
# ========================
model_name = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# ========================
# 3. Función para convertir DataFrame → Dataset
# ========================
def df_to_dataset(df):
    dataset_dict = {
        "image": [],
        "text": []
    }
    for _, row in df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        dataset_dict["image"].append(image)
        dataset_dict["text"].append(str(row["text"]))
    return Dataset.from_dict(dataset_dict)

train_dataset = df_to_dataset(train_df)
val_dataset = df_to_dataset(val_df)

# ========================
# 4. Tokenización / Procesamiento
# ========================
def preprocess_batch(batch):
    pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values
    labels = processor.tokenizer(batch["text"], padding="max_length", max_length=128, truncation=True).input_ids
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

train_dataset = train_dataset.map(preprocess_batch, batched=True, batch_size=4)
val_dataset = val_dataset.map(preprocess_batch, batched=True, batch_size=4)

# ========================
# 5. Configurar entrenamiento
# ========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    save_total_limit=2,
    logging_steps=10
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ========================
# 6. Entrenar
# ========================
trainer.train()

# ========================
# 7. Guardar modelo final
# ========================
model.save_pretrained("./trocr_model_final")
processor.save_pretrained("./trocr_model_final")
