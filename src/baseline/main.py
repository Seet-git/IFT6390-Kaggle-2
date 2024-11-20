import numpy as np
from optimizer import grid_search


def main():
    # Data loading
    data_train = np.load('../../data/train_data.pkl', allow_pickle=True)
    data_test = np.load('../../data/test_data.pkl', allow_pickle=True)

    # Extract data
    images_train = np.array(data_train['images'])
    labels_train = np.array(data_train['labels'])
    images_test = np.array(data_train['images'])

    grid_search(inputs_images=images_train, labels_images=labels_train)


if __name__ == "__main__":
    main()
