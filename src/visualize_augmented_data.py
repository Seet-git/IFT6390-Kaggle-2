import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from src.extract_data import load_data
from src.pretrained_models.training import get_default_transform, OCTDataset
import matplotlib.pyplot as plt

from src.pretrained_models.utils import convert_to_rgb, denormalize

images_train, labels_train, _ = load_data(10)
images_train = [convert_to_rgb(img) for img in images_train]

train_dataset = OCTDataset(images_train, labels_train, processor=None, transform=get_default_transform(transform=True))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

data_iter = iter(train_loader)

image_to_save = None
for i, (images, labels) in enumerate(data_iter):
    if i == 8:  # 9e image
        image_to_save = images[0]
        break

if image_to_save is not None:
    original_image = denormalize(image_to_save)
    original_image = (original_image - original_image.min()) / (
            original_image.max() - original_image.min())
    plt.figure()
    plt.imshow(original_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./plots/ref_augmented.png", format="png",
                bbox_inches='tight', pad_inches=0)

    plt.show()
    print('Image saved')
