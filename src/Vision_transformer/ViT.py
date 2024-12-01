from src.Vision_transformer.training import train
from src.extract_data import load_data


def main():
    images, labels, _ = load_data(10)
    hp = {
        "batch_size": 16,
        "lr": 5e-5,
    }
    train(images, labels, hp)