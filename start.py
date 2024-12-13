import torch

import config
from src.pretrained_models.bayesian_opti import bayesian_optimization
from src.pretrained_models.evaluation import predict
from src.pretrained_models.hugging_face import train as train_hf
from src.pretrained_models.training import train as train_base
from src.extract_data import load_data


def main():
    images_train, labels_train, images_test = load_data()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    hp_base_vit = {
        "algo": "ViT",
        "epochs": 20,
        "optimizer": "adamW",
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "scheduler": "cosine"
    }

    hp_base_mobile_net_v3 = {
        "algo": "MobileNetV3",
        "epochs": 20,
        "optimizer": "sgd",
        "batch_size": 64,
        "lr": 0.045,
        "momentum": 0.9,
        "weight_decay": 4e-5
    }

    hp_base_efficient_net_b0 = {
        "algo": "EfficientNet-B0",
        "epochs": 20,
        "optimizer": "adam",
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.00,
        "scheduler": "cosine"
    }

    hp_base_dense_net_121 = {
        "algo": "DenseNet121",
        "epochs": 20,
        "optimizer": "sgd",
        "batch_size": 64,
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.01
    }

    hp_base_resnet = {
        "algo": "ResNet18",
        "epochs": 20,
        "optimizer": "sgd",
        "batch_size": 64,
        "lr": 0.00002,
        "momentum": 0.9,
        "weight_decay": 2e-05
    }

    hp_base_vgg = {
        "algo": "vgg16",
        "epochs": 20,
        "optimizer": "sgd",
        "batch_size": 64,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0,
        "scheduler": "lr"
    }

    hp_base_hybrid = {
        "algo": "HybridResNetEfficientNet",
        "epochs": 20,
        "optimizer": "adam",
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.00,
        "scheduler": "cosine"
    }

    # Bayesian optimization
    # bayesian_optimization("EfficientNet-B0")

    # Train model
    train_base(images_train, labels_train, hp_base_hybrid)

    # Predict
    # predict(images_test, hp_base_hybrid)

    # Hugging face
    hp_hf = {
        "algo": "ViT",
        "epochs": 20,
        "optimizer": "adamW",
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "scheduler": "cosine"
    }

    # train_hf(images_train, labels_train, hp_hf)


if __name__ == "__main__":
    main()
