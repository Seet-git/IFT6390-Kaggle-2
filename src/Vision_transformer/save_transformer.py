import os

import config


def save_model(model, processor, fold, epoch):
    # Sauvegarder le modèle
    os.makedirs(f"./hyperparameters/{config.ALGORITHM}", exist_ok=True)
    model.save_pretrained(f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}_{fold}_{epoch}")
    processor.save_pretrained(f"./hyperparameters/{config.ALGORITHM}/processor_{config.ALGORITHM}")
