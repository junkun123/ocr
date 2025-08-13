import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

def read_csv(csv_path, subfolder):
    df = pd.read_csv(csv_path)  # lee encabezado de archivo CSV
    df['image_path'] = df['filename'].apply(lambda p: os.path.join(DATA_DIR, subfolder, p))
    return df[['image_path', 'text']]

if __name__ == '__main__':
    train_df = read_csv(os.path.join(DATA_DIR, 'train_labels.csv'), 'train')
    val_df = read_csv(os.path.join(DATA_DIR, 'val_labels.csv'), 'val')

    def check_images(df):
        ok = []
        for idx, row in df.iterrows():
            p = row['image_path']
            try:
                Image.open(p).convert('RGB')
                ok.append(True)
            except Exception as e:
                print('Error abriendo', p, '->', e)
                ok.append(False)
        return df[ok]

    train_df = check_images(train_df)
    val_df = check_images(val_df)

    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))
    ds = DatasetDict({'train': ds_train, 'validation': ds_val})

    ds.save_to_disk(os.path.join(os.path.dirname(__file__), '..', 'hf_dataset'))
    print('Dataset saved to hf_dataset')
