import numpy as np
from training import k_cross_validation
from utils import create_one_hot


def main():
    np.random.seed(4)
    # Data loading
    data_train = np.load('../../data/train_data.pkl', allow_pickle=True)
    data_test = np.load('../../data/test_data.pkl', allow_pickle=True)

    # Extract data
    images_train = np.array(data_train['images'])
    labels_train = np.array(data_train['labels'])
    images_test = np.array(data_train['images'])

    one_hot_train = create_one_hot(labels_train)

    hp = {
        "input_size": 28 * 28,
        "hidden_size1": 128,
        "hidden_size2": 64,
        "dropout_rate": 0.5,
        "batch_size": 32,
        "eta": 0.001,
        "epochs": 10
    }

    k_cross_validation(inputs_images=images_train, one_hot=one_hot_train, hp=hp, n_split=5)


if __name__ == "__main__":
    main()
