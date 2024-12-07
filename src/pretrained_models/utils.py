from PIL import Image
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, processor=None, transform=None):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.processor:
            # ViT processor
            inputs = self.processor(images=image, return_tensors="pt")
            image = inputs["pixel_values"].squeeze(0)
        elif self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image


def convert_to_rgb(image):
    """
    Convert 2D image into 3D with three channels
    """
    return Image.fromarray((np.stack([image] * 3, axis=-1) * 255).astype('uint8'))


def balance_validation_set(inputs_val, labels_val):
    labels_val = np.array(labels_val)

    # Get minority class
    class_counts = np.bincount(labels_val)
    min_count = np.min(class_counts)

    # Balance index
    balanced_indices = []
    for classe in range(len(class_counts)):
        # Get index class
        cls_indices = np.where(labels_val == classe)[0]
        # Sample to balance
        balanced_indices.extend(np.random.choice(cls_indices, size=min_count, replace=False))

    # Shuffle index
    np.random.shuffle(balanced_indices)

    # Get balanced data
    balanced_inputs = [inputs_val[i] for i in balanced_indices]
    balanced_labels = [labels_val[i] for i in balanced_indices]

    return balanced_inputs, balanced_labels


def plot_confusion_matrix(y_true, y_pred):
    class_names = ["CN", "DME", "Drusen", "Healthy"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


def apply_smote_to_images(inputs, labels):
    smote = SMOTE(random_state=1, k_neighbors=2)
    # PIL to numpy
    flattened_inputs = np.array([np.array(img) for img in inputs]).reshape(len(inputs), -1)

    # SMOTE
    inputs_resampled, labels_resampled = smote.fit_resample(flattened_inputs, labels)
    original_shape = np.array(inputs[0]).shape

    # numpy to PIL
    inputs_resampled = [
        Image.fromarray(img.reshape(original_shape).astype(np.uint8)) for img in inputs_resampled
    ]

    return inputs_resampled, labels_resampled


def generate_bottom_mask(image_size=(224, 224), crop_ratio=0.3):
    mask = torch.zeros(image_size)
    top = int(image_size[0] * (1 - crop_ratio))
    mask[top:, :] = 1  # 1 sur la région inférieure
    return mask
