import os
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

import config


def export_output(output_pred: list):
    np.savetxt(
        f'./output/{config.ALGORITHM}.csv',
        np.column_stack((np.arange(1, len(output_pred) + 1), output_pred)),
        delimiter=',',
        header='ID,label',
        comments='',
        fmt='%d'
    )

    print("Prediction saved.")


def extract_data_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed --> {os.path.abspath(extract_to)}")


def load_data(percent: int = 100, path = "./"):
    # Unzip
    extract_data_zip(f"{path}data/data.zip", f"{path}data")

    # Data loading
    data_train = np.load(f'{path}data/train_data.pkl', allow_pickle=True)
    data_test = np.load(f'{path}data/test_data.pkl', allow_pickle=True)

    # Extract data
    images_train = np.array(data_train['images'])
    labels_train = np.array(data_train['labels'])
    images_test = np.array(data_test['images'])

    total_samples = len(images_train)
    num_samples = int(total_samples * (percent / 100))

    # Stratified sampling
    if percent < 100:
        images_train_subset, _, labels_train_subset, _ = train_test_split(
            images_train, labels_train,
            train_size=num_samples,
            stratify=labels_train,  # Same class distribution
            random_state=1
        )
    else:
        images_train_subset = images_train
        labels_train_subset = labels_train

    print(
        f"Loaded {len(images_train_subset)} train images - Shape {images_train_subset.shape[1]} x {images_train_subset.shape[2]}")

    return images_train_subset, labels_train_subset, images_test
