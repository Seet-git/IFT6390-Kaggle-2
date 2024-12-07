import os
import torch
import config


def save_model_hf(trainer, fold):
    """
    Hugging face models
    """
    os.makedirs(f"./hyperparameters/{config.ALGORITHM}", exist_ok=True)
    trainer.save_model_base(f"./hyperparameters/{config.ALGORITHM}/model_{config.ALGORITHM}_{fold}_hf")


def save_models_base(model, processor, fold, epoch):
    """
    Torchvision and transformers models
    """
    # Set path
    path = f"./hyperparameters/{config.ALGORITHM}/{config.OUTPUT_HP_PATH}"
    os.makedirs(path, exist_ok=True)

    if hasattr(model, "save_pretrained"):  # Transformers
        model.save_pretrained(f"{path}/model_{config.ALGORITHM}_f{fold}_e{epoch}")
        if processor:
            processor.save_pretrained(f"{path}/processor_{config.ALGORITHM}")
    else:  # torchvision models
        torch.save(model.state_dict(), f"{path}/model_{config.ALGORITHM}_f{fold}_e{epoch}.pth")
