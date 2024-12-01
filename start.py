import config
from src.Vision_transformer.evaluation import predict
from src.Vision_transformer.training import train
from src.extract_data import load_data


def main():
    config.ALGORITHM = "ViT"
    images_train, labels_train, images_test = load_data()
    hp = {
        "batch_size": 128,
        "lr": 0.001,
    }
    train(images_train, labels_train, hp)
    predict(images_test, hp)


if __name__ == "__main__":
    main()
