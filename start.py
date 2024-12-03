import config
from src.Vision_transformer.evaluation import predict
from src.Vision_transformer.hugging_face import train as train_hf
from src.Vision_transformer.training import train as train_base
from src.extract_data import load_data


def main():

    images_train, labels_train, images_test = load_data(10)

    hp_base = {
        "algo": "ResNet18",
        "epochs": 20,
        "optimizer": "adamW",
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.01
    }

    train_base(images_train, labels_train, hp_base)
    predict(images_test, hp_base)

    # Hugging face
    # train_hf(images_train, labels_train, hp_hf)


if __name__ == "__main__":
    main()
