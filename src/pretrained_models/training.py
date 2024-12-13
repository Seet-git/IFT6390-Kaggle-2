import wandb
from datetime import datetime
import pytz
from pytorch_grad_cam import GradCAM
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from torchvision import models, transforms
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from torchsummary import summary

import config
from src.pretrained_models.utils import convert_to_rgb, OCTDataset, apply_smote_to_images, \
    balance_validation_set, generate_bottom_mask
from src.pretrained_models.visualisation import plot_visualisations, plot_grad_cam

from src.pretrained_models.save_models import save_models_base

logging.set_verbosity_error()

# Global variables
best_macro = -np.inf
patience = 10


class HybridResNetEfficientNet(nn.Module):
    def __init__(self, num_classes, is_weight=True):
        res_weight = None
        eff_weight = None
        super(HybridResNetEfficientNet, self).__init__()
        if is_weight:
            res_weight = models.ResNet50_Weights.IMAGENET1K_V1
            eff_weight = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.resnet50 = models.resnet50(weights=res_weight)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.to(device=config.DEVICE)
        # summary(self.resnet50, input_size=(3, 224, 224))

        self.efficientnet_b0 = models.efficientnet_b0(weights=eff_weight)
        self.efficientnet_b0 = nn.Sequential(*list(self.efficientnet_b0.children())[:-1])
        self.to(device=config.DEVICE)
        # summary(self.efficientnet_b0, input_size=(3, 224, 224))
        self.fc = nn.Sequential(
            nn.Linear(2048 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_features = self.resnet50(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        efficientnet_features = self.efficientnet_b0(x)
        efficientnet_features = efficientnet_features.view(efficientnet_features.size(0), -1)
        combined_features = torch.cat((resnet_features, efficientnet_features), dim=1)
        output = self.fc(combined_features)
        return output


def get_default_transform(transform=False, image_size=(224, 224)):
    scale = (0.9, 0.9)
    if transform:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def preprocessing():
    if config.ALGORITHM == "ViT":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=4
        )
    else:
        if config.ALGORITHM == "MobileNetV3":
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "EfficientNet-B0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "EfficientNet-B1":
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        elif config.ALGORITHM == "DenseNet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model.classifier = nn.Linear(model.classifier.in_features, 4)

        elif config.ALGORITHM == "ResNet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 4)

        elif config.ALGORITHM == "vgg16":
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            mod = list(model.classifier.children())
            mod.pop()
            mod.append(torch.nn.Linear(4096, 4))
            new_classifier = torch.nn.Sequential(*mod)
            model.classifier = new_classifier

        elif config.ALGORITHM == "HybridResNetEfficientNet":
            model = HybridResNetEfficientNet(num_classes=4)

        else:
            raise ValueError(f"Unsupported model name: {config.ALGORITHM}")

        # Freeze layers
        if config.ALGORITHM != "HybridResNetEfficientNet":
            for param in model.features.parameters():
                param.requires_grad = True

        processor = None

    return model, processor


def infer(model, infer_loader, is_cam_grad=False):
    model.eval()
    labels_tab = []
    pred_tab = []
    efficientnet_cam = None
    resnet_cam = None

    # Target layer for Grad-CAM
    if config.ALGORITHM == "HybridResNetEfficientNet":
        efficientnet_target_layer = model.efficientnet_b0[-2]
        resnet_target_layer = model.resnet50[6][-1].conv3
        efficientnet_cam = GradCAM(model=model, target_layers=[efficientnet_target_layer])
        resnet_cam = GradCAM(model=model, target_layers=[resnet_target_layer])

    with torch.set_grad_enabled(True):
        for idx, (images, labels) in enumerate(infer_loader):

            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # AMC
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']

            _, y_pred = torch.max(outputs, 1)

            labels_tab.extend(labels.cpu().numpy())
            pred_tab.extend(y_pred.cpu().numpy())

            # Plot grad cam for misclassified predictions
            if is_cam_grad and config.ALGORITHM == "HybridResNetEfficientNet":
                counter = 0
                for i in range(len(images)):
                    if y_pred[i] != labels[i] and labels[i] == 2 and y_pred[i] == 3:  # Misclassified
                        counter += 1
                        plot_grad_cam(images, i, labels, y_pred, efficientnet_cam, resnet_cam)

                print(f"Cam grad batch {idx} - Misclassified {counter}")

    # Compute metrics
    accuracy = accuracy_score(labels_tab, pred_tab)
    f1_macro = f1_score(labels_tab, pred_tab, average='macro')
    f1_weighted = f1_score(labels_tab, pred_tab, average='weighted')
    print(classification_report(labels_tab, pred_tab, zero_division=0))

    # Plot visualisations
    if is_cam_grad:
        plot_visualisations(labels_tab, pred_tab)

    return accuracy, f1_macro, f1_weighted


def fit(model, processor, train_loader, val_loader, criterion, optimizer, fold, scheduler, hp):
    # Initialisation
    global best_macro
    best_macro = -np.inf
    macro_score = []
    no_improvement = 0

    # AMP
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

    # Loop epoch
    for epoch in range(hp["epochs"]):

        # Unfreeze
        if epoch + 1 == (0.1 * hp['epochs']) and config.ALGORITHM != "ViT" and config.ALGORITHM != "HybridResNetEfficientNet":
            print("Unfreeze")
            for param in model.features.parameters():
                param.requires_grad = True

        # Train mode
        model.train()
        train_loss = 0

        # Get current time
        montreal_timezone = pytz.timezone('America/Montreal')
        current_time = datetime.now(montreal_timezone).strftime("%H:%M")

        # Loop on all epochs
        for images, labels in tqdm(train_loader, desc=f"Starting epoch {epoch + 1}/{hp['epochs']} at {current_time}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # Compute gradient
            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                if isinstance(outputs, dict):  # Transformers
                    outputs = outputs['logits']

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            scheduler.step()  # Update scheduler

        # Compute metrics
        train_loss /= len(train_loader)
        acc_train, macro_train, weighted_train = infer(model, train_loader)

        if (epoch + 1) % 10 == 0:
            acc_val, macro_val, weighted_val = infer(model, val_loader, is_cam_grad=True)
        else:
            acc_val, macro_val, weighted_val = infer(model, val_loader)
        macro_score.append(macro_val)

        # Logs
        if config.WANDB_ACTIVATE:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "epoch": epoch + 1,
                "train_f1_macro": macro_train,
                "val_f1_macro": macro_val,
                "fold": fold,
            })

        # Debug
        print(
            f"\tEpoch {epoch + 1} - Loss {train_loss:.4f} - Accuracy (train) {acc_train:.4f} - Accuracy (val) "
            f"{acc_val:.4f} - F1 macro (val) {macro_val:.4f}")

        # Save the best model
        if best_macro < macro_val:
            print("New best model !")
            best_macro = macro_val
            no_improvement = 0
            save_models_base(model, processor, fold, epoch)

        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Stop run
        elif acc_val < 0.2:
            print("Bad score ! Stopping run...\n\n")
            return -1

    print("")
    return macro_score


def k_cross_validation(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict,
                       n_split: int = 5):
    # Initialization
    global best_macro
    best_macro = -np.inf
    all_scores = []

    # Convert PIL gray to PIL RGB
    inputs_images = [convert_to_rgb(img) for img in inputs_images]

    # Stratified k-fold
    k_fold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(inputs_images, labels_images)):
        print(f"Fold {fold + 1}/{n_split}")

        # Load models and processor
        model, processor = preprocessing()
        model.to(config.DEVICE)
        print("Run config:")
        print(f"\tDevice {config.DEVICE}")

        # Set inputs and labels
        inputs_train, inputs_val = [inputs_images[i] for i in train_idx], [inputs_images[i] for i in val_idx]
        labels_train, labels_val = [labels_images[i] for i in train_idx], [labels_images[i] for i in val_idx]

        # Data sampling
        # inputs_train, labels_train = apply_smote_to_images(inputs_train, labels_train)
        # inputs_val, labels_val = balance_validation_set(inputs_val, labels_val)

        # Create datasets
        train_dataset = OCTDataset(inputs_train, labels_train, processor,
                                   get_default_transform(transform=True))
        val_dataset = OCTDataset(inputs_val, labels_val, processor, get_default_transform(transform=False))

        # Debug
        print(f"\tFold {fold + 1}: Train class distribution: {np.bincount(labels_train)}")
        print(f"\tFold {fold + 1}: Val class distribution: {np.bincount(labels_val)}")

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, pin_memory=True,
                                num_workers=16)

        # Compute weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_train),
            y=labels_train
        )

        # Set Loss function
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
        print(f"\tClass weights: {class_weights}\n")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Set Optimizer
        if hp["optimizer"] == "adamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        elif hp["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        elif hp["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        elif hp["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), momentum=hp['momentum'], lr=hp["lr"],
                                        weight_decay=hp["weight_decay"])
        else:
            raise ValueError("Bad OPTIMIZER value")

        # Set Scheduler
        if hp["scheduler"] == 'cosine':
            num_training_steps = len(train_loader) * hp["epochs"]
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps),
                                                        num_training_steps=num_training_steps)
        elif hp["scheduler"] == 'lr':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            raise ValueError("Bad SCHEDULER value")

        # Set model
        acc = fit(model=model, processor=processor, train_loader=train_loader, val_loader=val_loader,
                  criterion=criterion, optimizer=optimizer, scheduler=scheduler, fold=fold,
                  hp=hp)

        # Stop run
        if acc == -1:
            break

        # Logs
        if config.WANDB_ACTIVATE:
            wandb.log({"fold_f1": np.mean(acc), "fold": fold + 1})
        print(f"Fold {fold + 1} -  Mean F1 score (val) {np.mean(acc)}\n")

        all_scores.extend(acc)

    # Prevent too bad score
    if len(all_scores) == 0:
        return [0]

    # Return mean f1 score
    print(f"\nTrain mean F1_score {np.mean(all_scores)}\n")
    return all_scores


def train(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict, trial=''):
    print("Hyperparameters: ", hp)

    # Initialization
    global best_macro
    best_macro = -np.inf
    np.random.seed(4)

    # Get current time
    montreal_timezone = pytz.timezone('America/Montreal')
    current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

    # Setup config file
    config.ALGORITHM = hp["algo"]
    config.OUTPUT_HP_PATH = f"{config.ALGORITHM}-{hp['optimizer']} - {current_time}".replace(":", "-")

    # Wandb start
    if config.WANDB_ACTIVATE:
        if trial != '':
            trial = f" - Trial {trial}"

        wandb.init(
            project=f"Kaggle-2",
            config=hp,
            name=f"{config.ALGORITHM}-{hp['optimizer']} - {current_time}{trial}",
        )
        print("")

    # Start training
    all_score = k_cross_validation(inputs_images, labels_images, hp)

    # Wandb stop
    if config.WANDB_ACTIVATE:
        wandb.finish()
        print("\n")

    return all_score
