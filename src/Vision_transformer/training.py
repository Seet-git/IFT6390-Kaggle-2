from datetime import datetime

import pytz
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.amp import GradScaler, autocast

import config

import numpy as np
from tqdm import tqdm

from src.Vision_transformer.save_transformer import save_model_base
from src.Vision_transformer.utils import convert_to_rgb, OCTDataset, balance_validation_set
import wandb

logging.set_verbosity_error()
best_acc = -np.inf
patience = 3


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


def fit(model, processor, train_loader, val_loader, criterion, optimizer, fold, scheduler, hp):
    global best_acc
    best_acc = -np.inf
    no_improvement = 0
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(hp["epochs"]):
        model.train()
        train_loss = 0
        current_time = datetime.now().strftime("%H:%M")
        for images, labels in tqdm(train_loader, desc=f"Starting epoch {epoch + 1}/{hp['epochs']} at {current_time}"):
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
            scheduler.step()

        train_loss /= len(train_loader)
        acc_train = infer(model, train_loader)
        acc_val = infer(model, val_loader)

        if config.WANDB_ACTIVATE:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": acc_train,
                "val_accuracy": acc_val,
                "epoch": epoch + 1,
                "fold": fold,
            })

        print(
            f"\tEpoch {epoch + 1} - Loss {train_loss} - Accuracy (train) {acc_train} - Accuracy (val) {acc_val}")

        if best_acc < acc_val:
            print("New best model !")
            best_acc = acc_val
            save_model_base(model, processor, fold, epoch)
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
        print("Run config:")
        print(f"\tDevice {config.DEVICE}")

        # Set inputs and labels
        inputs_train, inputs_val = [inputs_images[i] for i in train_idx], [inputs_images[i] for i in val_idx]
        labels_train, labels_val = [labels_images[i] for i in train_idx], [labels_images[i] for i in val_idx]
        print(f"\tFold {fold + 1}: Train class distribution: {np.bincount(labels_train)}")
        print(f"\tFold {fold + 1}: Val class distribution: {np.bincount(labels_val)}")

        inputs_val, labels_val = balance_validation_set(inputs_val, labels_val)

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
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
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
        model = fit(model=model, processor=processor, train_loader=train_loader, val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, fold=fold, hp=hp)

        if model == None:
            all_scores.append(0)
            break

        acc = infer(model=model, infer_loader=val_loader)
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
    if config.ALGORITHM != "ViT":
        raise ValueError("Bad ALGORITHM value")

    global best_acc
    best_acc = -np.inf
    np.random.seed(4)
    montreal_timezone = pytz.timezone('America/Montreal')
    current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

    if config.WANDB_ACTIVATE:
        wandb.init(
            project=f"{config.ALGORITHM}-Training",
            config=hp,
            name=f"{config.ALGORITHM}-{hp['optimizer']} - {current_time} - Trial {trial}",
        )
        print("")

    config.OUTPUT_HP_PATH = f"{config.ALGORITHM}-{hp['optimizer']} - {current_time}".replace(":", "-")
    k_cross_validation(inputs_images, labels_images, hp)

    if config.WANDB_ACTIVATE:
        wandb.finish()
        print("\n")
