from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import logging
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.amp import GradScaler, autocast
import config

import numpy as np
from tqdm import tqdm

from src.Vision_transformer.save_transformer import save_model
from src.Vision_transformer.utils import convert_to_rgb, OCTDataset, balance_validation_set

logging.set_verbosity_error()
best_acc = -np.inf


def preprocessing():
    """
    # Load preprocessing ViT
    :return:
    """
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=4
    )

    return processor, model


def infer(model, infer_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in infer_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images).logits

            _, y_pred = torch.max(outputs, 1)
            correct += (y_pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def fit(model, processor, train_loader, val_loader, criterion, optimizer, fold, epochs=5):
    global best_acc
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # Normal
            # optimizer.zero_grad()
            # outputs = model(images).logits
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # train_loss += loss.item()

            # AMP
            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        acc_train = infer(model, train_loader)
        acc_val = infer(model, val_loader)

        print(
            f"\tEpoch {epoch + 1} - Loss {train_loss} - Accuracy (train) {acc_train} - Accuracy (val) {acc_val}")

        if best_acc < acc_val:
            print("New best model !")
            best_acc = acc_val
            save_model(model, processor, fold, epoch)

    print("")
    return model

def k_cross_validation(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict, n_split: int = 5):
    global best_acc
    best_acc = -np.inf
    inputs_images = [convert_to_rgb(img) for img in inputs_images]
    all_scores = []
    model = None
    processor = None

    k_fold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(inputs_images, labels_images)):
        print(f"Fold {fold + 1}/{n_split}")

        processor, model = preprocessing()
        model.to(config.DEVICE)
        print(f"Device {config.DEVICE}")

        # Set inputs and labels
        inputs_train, inputs_val = [inputs_images[i] for i in train_idx], [inputs_images[i] for i in val_idx]
        labels_train, labels_val = [labels_images[i] for i in train_idx], [labels_images[i] for i in val_idx]
        print(f"Fold {fold + 1}: Train class distribution: {np.bincount(labels_train)}")
        print(f"Fold {fold + 1}: Val class distribution: {np.bincount(labels_val)}")

        inputs_val, labels_val = balance_validation_set(inputs_val, labels_val)

        print(f"Fold {fold + 1}: New Val class distribution: {np.bincount(labels_val)}")

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
        print(f"Class weights: {class_weights}")

        # Set model
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])

        #
        model = fit(model=model, processor=processor, train_loader=train_loader, val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer, fold=fold)

        acc = infer(model=model, infer_loader=val_loader)
        all_scores.append(acc)

        print(f"Fold {fold + 1} -  Accuracy (val) {acc}")

    print(f"Mean Accuracy: {np.mean(all_scores)}")


def train(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict):
    """

    :param inputs_images:
    :param labels_images:
    :param hp:
    :return:
    """
    if config.ALGORITHM != "ViT":
        raise ValueError("Bad ALGORITHM value")

    global best_acc
    best_acc = -np.inf
    np.random.seed(4)
    k_cross_validation(inputs_images, labels_images, hp)
