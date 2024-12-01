from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# Override Dataset class for ViT
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

        if self.labels is not None:
            label = self.labels[idx]
            return encoding['pixel_values'].squeeze(0), torch.tensor(label)
        else:
            return encoding['pixel_values'].squeeze(0)


def convert_to_rgb(image):
    """
    Convert 2D image into 3D with 3 channels
    :param image:
    :return:
    """
    return Image.fromarray((np.stack([image] * 3, axis=-1) * 255).astype('uint8'))


def balance_validation_set(inputs_val, labels_val):
    """
    Balance validation
    """
    labels_val = np.array(labels_val)

    # Get minority class
    class_counts = np.bincount(labels_val)
    min_count = np.min(class_counts)

    # Rééquilibrer les indices
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
