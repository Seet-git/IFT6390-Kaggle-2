import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from tqdm import tqdm

# Convert image 2D into 3D with 3 channels
def convert_to_rgb(image):
    return Image.fromarray((np.stack([image] * 3, axis=-1) * 255).astype('uint8'))

from src.extract_data import load_data

images, labels, _ = load_data(10)

images = [convert_to_rgb(img) for img in images]

# Split data
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)


class OCTDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        encoding = self.processor(image, return_tensors="pt")
        return encoding['pixel_values'].squeeze(0), torch.tensor(label)

# Charger le processeur et le modèle ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=4
)

# Load dataset
train_dataset = OCTDataset(train_images, train_labels, processor)
val_dataset = OCTDataset(val_images, val_labels, processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss {val_loss:.4f}, Accuracy {accuracy:.4f}")

# Sauvegarder le modèle
model.save_pretrained("vit-oct-classification")
processor.save_pretrained("vit-oct-classification")
