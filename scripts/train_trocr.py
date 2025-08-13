# train_trocr.py
import os
import math
import torch
from datasets import load_from_disk
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator)
from PIL import Image

BASE_MODEL = 'microsoft/trocr-base-stage1'
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
HF_DATASET_DIR = os.path.join(PROJECT_DIR, 'hf_dataset')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'trocr_finetuned')

# Cargar dataset preparado
ds = load_from_disk(HF_DATASET_DIR)

# Cargar processor y modelo
processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

# Ajustes del tokenizer/decoder
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Preprocess function: transforma imagen -> pixel_values; tokeniza labels

def preprocess(batch):
    images = [Image.open(p).convert('RGB') for p in batch['image_path']]
    pixel_values = processor(images=images, return_tensors='pt').pixel_values
    # tokenizar textos
    with processor.as_target_processor():
        labels = processor(text=batch['text']).input_ids
    batch['pixel_values'] = list(pixel_values)
    batch['labels'] = labels
    return batch

# Mapear (usar batched=False para manejar PIL Image conversion)
ds_train = ds['train'].map(preprocess)
ds_val = ds['validation'].map(preprocess)

# data_collator para Seq2SeqTrainer

def collate_fn(features):
    pixel_values = torch.stack([f['pixel_values'] for f in features])
    labels = [f['labels'] for f in features]
    labels = processor.tokenizer.pad({'input_ids': labels}, return_tensors='pt', padding=True)
    labels_input_ids = labels['input_ids']
    labels_input_ids[labels_input_ids == processor.tokenizer.pad_token_id] = -100
    return {'pixel_values': pixel_values, 'labels': labels_input_ids}

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    num_train_epochs=5,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=collate_fn,
    tokenizer=processor.feature_extractor,  # no es tokenizer real, pero evita errores; usamos processor directamente
)

trainer.train()

# guardar
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print('Modelo guardado en', OUTPUT_DIR)