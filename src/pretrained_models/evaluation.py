import numpy as np
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

from src.pretrained_models.training import get_default_transform
from src.extract_data import export_output
import torch
from torchvision import models
import torch.nn as nn
from src.pretrained_models.utils import OCTDataset, convert_to_rgb
import config


def load_model_and_processor():
    """
    Load the model and processor/transformations for the specified algorithm.
    """
    model_path = f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}"
    processor_path = f"./hyperparameters/{config.ALGORITHM}/processor_{config.ALGORITHM}"

    if config.ALGORITHM == "ViT":
        # Charger le modèle et le processeur pour model_ViT
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(processor_path)
        transform = None
    else:
        # Charger un modèle Torchvision
        if config.ALGORITHM == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "EfficientNet-B0":
            model =  models.efficientnet_b0(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "DenseNet121":
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, 4)
        elif config.ALGORITHM == "ResNet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 4)
        else:
            raise ValueError(f"Unsupported algorithm: {config.ALGORITHM}")

        model.load_state_dict(torch.load(f"{model_path}.pth"))
        processor = None

    return model, processor


def predict(predict_images: np.ndarray, hp: dict):
    """

    :param hp:
    :param predict_images:
    :return:
    """
    # Initialisation
    config.ALGORITHM = hp["algo"]
    output_pred = []
    model, processor = load_model_and_processor()
    print(f"{model} \n Processor : {processor}")

    model.eval()
    model.to(config.DEVICE)

    if config.ALGORITHM == "ViT":
        predict_images = [convert_to_rgb(img) for img in predict_images]

    test_dataset = OCTDataset(predict_images, labels=None,
                              processor=processor,
                              transform=get_default_transform(transform=False))

    test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    with torch.no_grad():
        for images in test_loader:
            images = images.to(config.DEVICE)

            outputs = model(images)
            if isinstance(outputs, dict):  # Transformers
                outputs = outputs['logits']

            _, y_pred = torch.max(outputs, 1)
            output_pred.extend(y_pred.cpu().numpy())

    export_output(output_pred)
