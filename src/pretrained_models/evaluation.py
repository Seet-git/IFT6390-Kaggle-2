import numpy as np
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

from src.pretrained_models.training import get_default_transform, HybridResNetEfficientNet
from src.extract_data import export_output
import torch
from torchvision import models
import torch.nn as nn
from src.pretrained_models.utils import OCTDataset, convert_to_rgb
import config


def load_model_and_processor():
    # Path for hyperparameters
    model_path = f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}"
    processor_path = f"./hyperparameters/{config.ALGORITHM}/processor_{config.ALGORITHM}"

    if config.ALGORITHM == "ViT":  # Transformers
        # Load model and processor for ViT models
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(processor_path)
    else:  # Pre-trained torchvision models
        if config.ALGORITHM == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "EfficientNet-B0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "DenseNet121":
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, 4)
        elif config.ALGORITHM == "ResNet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 4)
        elif config.ALGORITHM == "HybridResNetEfficientNet":
            model = HybridResNetEfficientNet(num_classes=4, is_weight=False)
        else:
            raise ValueError(f"Bad ALGORITHM value.")

        # Load model
        model.load_state_dict(torch.load(f"{model_path}.pth"), strict=False)
        processor = None

    return model, processor


def predict(predict_images: np.ndarray, hp: dict):
    # Initialisation
    config.ALGORITHM = hp["algo"]
    model, processor = load_model_and_processor()  # Get model
    model.eval()
    model.to(config.DEVICE)
    output_pred = []

    # Prepare data test
    predict_images = [convert_to_rgb(img) for img in predict_images]
    test_dataset = OCTDataset(predict_images, None, processor, get_default_transform(transform=False))
    test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    # Prediction
    with torch.no_grad():
        for images in test_loader:
            images = images.to(config.DEVICE)

            outputs = model(images)
            if isinstance(outputs, dict):  # Transformers
                outputs = outputs['logits']

            _, y_pred = torch.max(outputs, 1)
            output_pred.extend(y_pred.cpu().numpy())

    # Save prediction
    export_output(output_pred)
