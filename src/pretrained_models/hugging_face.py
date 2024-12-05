from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np
import config
from torch.utils.data import Dataset
import os

from src.pretrained_models.save_models import save_model_hf
from src.pretrained_models.utils import convert_to_rgb, balance_validation_set


class OCTDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        encoding = self.processor(image, return_tensors="pt")

        item = {
            "pixel_values": encoding["pixel_values"].squeeze(0)
        }

        if self.labels is not None:
            label = self.labels[idx]
            item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


def preprocessing():
    """
    Préparer le modèle et le processeur pour le fine-tuning.
    """
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=4
    )
    return processor, model


def train_with_trainer(model, processor, train_dataset, val_dataset, hp, fold):
    """
    Fine-tuning avec Hugging Face Trainer.
    """
    os.makedirs(f"./temp/{config.ALGORITHM}/{fold}", exist_ok=True)
    training_args = TrainingArguments(
        output_dir=f"./temp/{config.ALGORITHM}/{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=hp['batch_size'],
        num_train_epochs=hp['epochs'],
        learning_rate=hp['lr'],
        save_total_limit=1,
        warmup_steps=int(0.1 * (len(train_dataset) // hp['batch_size'])),
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
        }
    )

    trainer.train()
    save_model_hf(trainer, fold)
    return trainer


def infer(model, infer_loader):
    """
    Effectuer une évaluation sur le jeu de test.
    """
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


def k_cross_validation(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict, n_split: int = 5):
    """
    Validation croisée avec Hugging Face Trainer.
    """
    inputs_images = [convert_to_rgb(img) for img in inputs_images]
    all_scores = []
    processor = None

    k_fold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(inputs_images, labels_images)):
        print(f"Fold {fold + 1}/{n_split}")

        # Préparation
        processor, model = preprocessing()
        model.to(config.DEVICE)

        inputs_train, inputs_val = [inputs_images[i] for i in train_idx], [inputs_images[i] for i in val_idx]
        labels_train, labels_val = [labels_images[i] for i in train_idx], [labels_images[i] for i in val_idx]

        inputs_val, labels_val = balance_validation_set(inputs_val, labels_val)

        # Datasets
        train_dataset = OCTDataset(inputs_train, labels_train, processor)
        val_dataset = OCTDataset(inputs_val, labels_val, processor)

        # Entraînement
        trainer = train_with_trainer(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            hp=hp,
            fold=fold
        )

        acc = infer(trainer.model, DataLoader(val_dataset, batch_size=hp['batch_size']))
        all_scores.append(acc)

        print(f"Fold {fold + 1} - Accuracy (val): {acc}")

    print(f"Mean Accuracy: {np.mean(all_scores)}")


def train(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict):
    """
    Lancer l'entraînement principal avec validation croisée.
    """
    if config.ALGORITHM != "model_ViT":
        raise ValueError("Bad ALGORITHM value")

    np.random.seed(4)
    k_cross_validation(inputs_images, labels_images, hp)
