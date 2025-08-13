# inference.py
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

MODEL_DIR = '../trocr_finetuned'

if len(sys.argv) < 2:
    print('Uso: python inference.py ruta/imagen.jpg')
    sys.exit(1)

img_path = sys.argv[1]

processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

image = Image.open(img_path).convert('RGB')
inputs = processor(images=image, return_tensors='pt').pixel_values
generated = model.generate(inputs)
text = processor.batch_decode(generated, skip_special_tokens=True)[0]
print('Texto:', text)