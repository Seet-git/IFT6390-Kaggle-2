import os

import config


def save_model_hf(trainer, fold):
    os.makedirs(f"./hyperparameters/{config.ALGORITHM}", exist_ok=True)
    trainer.save_model_base(f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}_{fold}_hf")


def save_model_base(model, processor, fold, epoch):
    # Sauvegarder le modèle
    os.makedirs(f"./hyperparameters/{config.ALGORITHM}/{config.OUTPUT_HP_PATH}", exist_ok=True)
    model.save_pretrained(
        f"./hyperparameters/{config.ALGORITHM}/{config.OUTPUT_HP_PATH}/model_{config.ALGORITHM}_f{fold}_e{epoch}")
    processor.save_pretrained(
        f"./hyperparameters/{config.ALGORITHM}/{config.OUTPUT_HP_PATH}/processor_{config.ALGORITHM}")
