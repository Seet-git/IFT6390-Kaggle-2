import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from timm import create_model
from src.extract_data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    from timm import list_models
    train_images, train_labels, _ = load_data()
    train_images, test_images, train_labels, test_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=1
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),  # Convertir en 3 canaux
        transforms.Resize((224, 224)),  # Taille suffisante pour tous les modèles
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RetinaDataset(train_images, train_labels, transform=transform)
    test_dataset = RetinaDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader


class RetinaDataset(Dataset):
    """Dataset personnalisé pour les images de rétine."""

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


def build_model_list():
    """Renvoie une liste de modèles préentraînés disponibles."""
    return {
        "ResNet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "EfficientNet-B0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "DenseNet121": lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "MobileNetV3": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
        "AlexNet": lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        "VGG16": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "NASNetMobile": lambda: create_model('nasnetalarge', pretrained=True),
        "Xception": lambda: create_model('legacy_xception', pretrained=True),
        "VisionTransformer": lambda: create_model('vit_base_patch16_224', pretrained=True),
    }


def evaluate_model(model, dataloader):
    """Évalue le modèle sur le dataloader donné."""
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    return accuracy, macro_f1, weighted_f1, report


def train_and_evaluate_models(train_loader, test_loader, model_list):
    """Entraîne et évalue chaque modèle dans la liste."""
    results = {}
    criterion = torch.nn.CrossEntropyLoss()

    for model_name, model_func in model_list.items():
        print(f"\nTesting model: {model_name}")

        # Charger le modèle préentraîné
        model = model_func()
        if hasattr(model, 'fc'):  # Pour ResNet et similaires
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
        elif hasattr(model, 'classifier'):  # Pour MobileNet, EfficientNet, etc.
            if isinstance(model.classifier, torch.nn.Sequential):
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 4)
            else:
                model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
        elif hasattr(model, 'head'):  # Pour Vision Transformers
            model.head = torch.nn.Linear(model.head.in_features, 4)
        model = model.to(device)

        # Optimiseur
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Entraînement rapide pour une seule époque
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Évaluation
        accuracy, macro_f1, weighted_f1, report = evaluate_model(model, test_loader)
        results[model_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report,
        }

        print(f"{model_name} Accuracy: {accuracy}")
        print(f"{model_name} Macro F1 Score: {macro_f1}")
        print(f"{model_name} Weighted F1 Score: {weighted_f1}")

    return results


def models_test():
    """Point d'entrée principal."""
    train_loader, test_loader = prepare_data()
    model_list = build_model_list()
    results = train_and_evaluate_models(train_loader, test_loader, model_list)

    print("\nComparaison des modèles :")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: Accuracy = {metrics['accuracy']}, Macro F1 = {metrics['macro_f1']}, Weighted F1 = {metrics['weighted_f1']}")
