import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

from src.Vision_transformer.utils import OCTDataset, convert_to_rgb

import config
from src.extract_data import export_output


def load_model_and_processor():
    model_path = f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}"
    processor_path = f"./hyperparameters/{config.ALGORITHM}/processor_{config.ALGORITHM}"

    # Charger le modèle
    model = ViTForImageClassification.from_pretrained(model_path)

    # Charger le processeur
    processor = ViTImageProcessor.from_pretrained(processor_path)

    return model, processor


def predict(predict_images: np.ndarray, hp: dict):
    """

    :param hp:
    :param processor:
    :param model:
    :param predict_images:
    :return:
    """
    # Initialisation
    output_pred = []
    model, processor = load_model_and_processor()

    model.eval()
    model.to(config.DEVICE)

    predict_images = [convert_to_rgb(img) for img in predict_images]

    test_dataset = OCTDataset(predict_images, labels=None, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    with torch.no_grad():
        for images in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images).logits
            _, y_pred = torch.max(outputs, 1)
            output_pred.extend(y_pred.cpu().numpy())

    export_output(output_pred)
