import wandb
from datetime import datetime
import pytz
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import torch
from torchvision import models
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

import config
from src.Vision_transformer.utils import convert_to_rgb, OCTDataset, balance_validation_set
from src.Vision_transformer.save_transformer import save_transformer

logging.set_verbosity_error()
best_macro = -np.inf
patience = 3


def preprocessing():
    """
    Load preprocessing and model based on the specified architecture.
    """
    processor = None
    model = None
    if config.ALGORITHM == "ViT":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=4
        )
    elif config.ALGORITHM == "MobileNetV3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        processor = None
    elif config.ALGORITHM == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        processor = None
    elif config.ALGORITHM == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        processor = None
    elif config.ALGORITHM == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        processor = None
    else:
        raise ValueError(f"Unsupported model name: {config.ALGORITHM}")

    return model, processor


def infer(model, infer_loader):
    model.eval()
    labels_tab = []
    pred_tab = []

    with torch.no_grad():
        for images, labels in infer_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images).logits if hasattr(model, 'logits') else model(images)
            _, y_pred = torch.max(outputs, 1)

            labels_tab.extend(labels.cpu().numpy())
            pred_tab.extend(y_pred.cpu().numpy())

    accuracy = accuracy_score(labels_tab, pred_tab)
    f1_macro = f1_score(labels_tab, pred_tab, average='macro')
    f1_weighted = f1_score(labels_tab, pred_tab, average='weighted')
    return accuracy, f1_macro, f1_weighted


def fit(model, processor, train_loader, val_loader, criterion, optimizer, fold, scheduler, hp):
    global best_macro
    best_macro = -np.inf
    no_improvement = 0
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")
    acc_val = None
    for epoch in range(hp["epochs"]):
        model.train()
        train_loss = 0
        montreal_timezone = pytz.timezone('America/Montreal')
        current_time = datetime.now(montreal_timezone).strftime("%H:%M")
        for images, labels in tqdm(train_loader, desc=f"Starting epoch {epoch + 1}/{hp['epochs']} at {current_time}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                if config.ALGORITHM == 'ViT':
                    outputs = model(images).logits
                else :
                    outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            scheduler.step()

        train_loss /= len(train_loader)
        acc_train, macro_train, weighted_train = infer(model, train_loader)
        acc_val, macro_val, weighted_val = infer(model, val_loader)

        if config.WANDB_ACTIVATE:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "epoch": epoch + 1,
                "val_f1_macro": macro_val,
                "fold": fold,
            })

        print(
            f"\tEpoch {epoch + 1} - Loss {train_loss:.4f} - Accuracy (train) {acc_train:.4f} - Accuracy (val) "
            f"{acc_val:.4f} - F1 macro (val) {macro_val:.4f}")

        if best_macro < macro_val:
            print("New best model !")
            best_macro = macro_val
            save_transformer(model, processor, fold, epoch)
            no_improvement = 0
        else:
            no_improvement += 1

        if acc_val < 0.6:
            print("Bad score ! Stopping run...")
            return None

        elif no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("")
    return acc_val


def k_cross_validation(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict,
                       n_split: int = 5):
    global best_macro
    best_macro = -np.inf
    inputs_images = [convert_to_rgb(img) for img in inputs_images]
    all_scores = []
    model = None
    processor = None

    k_fold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(inputs_images, labels_images)):
        print(f"Fold {fold + 1}/{n_split}")

        model, processor = preprocessing()
        model.to(config.DEVICE)
        print("Run config:")
        print(f"\tDevice {config.DEVICE}")

        # Set inputs and labels
        inputs_train, inputs_val = [inputs_images[i] for i in train_idx], [inputs_images[i] for i in val_idx]
        labels_train, labels_val = [labels_images[i] for i in train_idx], [labels_images[i] for i in val_idx]
        print(f"\tFold {fold + 1}: Train class distribution: {np.bincount(labels_train)}")
        print(f"\tFold {fold + 1}: Val class distribution: {np.bincount(labels_val)}")

        # inputs_val, labels_val = balance_validation_set(inputs_val, labels_val)
        print(f"\tFold {fold + 1}: New Val class distribution: {np.bincount(labels_val)}")

        # Create datasets
        train_dataset = OCTDataset(inputs_train, labels_train, processor)
        val_dataset = OCTDataset(inputs_val, labels_val, processor)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False)

        # Compute weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_train),
            y=labels_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
        print(f"\tClass weights: {class_weights}\n")

        # Set model
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if hp["optimizer"] == "adamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        elif hp["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), momentum=hp['momentum'], lr=hp["lr"])
        else:
            raise ValueError("Bad OPTIMIZER value")

        num_training_steps = len(train_loader) * hp["epochs"]
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(0.1 * num_training_steps),
                                                    num_training_steps=num_training_steps)

        # Set model
        acc = fit(model=model, processor=processor, train_loader=train_loader, val_loader=val_loader,
                  criterion=criterion,
                  optimizer=optimizer, scheduler=scheduler, fold=fold, hp=hp)

        if acc == None:
            all_scores.append(0)
            break

        all_scores.append(acc)

        if config.WANDB_ACTIVATE:
            wandb.log({"fold_accuracy": acc, "fold": fold + 1})
        print(f"Fold {fold + 1} -  Accuracy (val) {acc}\n")

    print(f"\nMean Accuracy {np.mean(all_scores)}\n")


def train(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict, trial=''):
    """

        :param inputs_images:
        :param labels_images:
        :param hp:
        :param trial:
        :return:
        """
    print("Hyper-parameters: ", hp)

    config.ALGORITHM = hp["algo"]

    global best_macro
    best_macro = -np.inf
    np.random.seed(4)
    montreal_timezone = pytz.timezone('America/Montreal')
    current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

    if config.WANDB_ACTIVATE:
        if trial != '':
            trial = f" - Trial {trial}"

        wandb.init(
            project=f"{config.ALGORITHM}-Training",
            config=hp,
            name=f"{config.ALGORITHM}-{hp['optimizer']} - {current_time}{trial}",
        )
        print("")

    config.OUTPUT_HP_PATH = f"{config.ALGORITHM}-{hp['optimizer']} - {current_time}".replace(":", "-")
    k_cross_validation(inputs_images, labels_images, hp)

    if config.WANDB_ACTIVATE:
        wandb.finish()
        print("\n")
