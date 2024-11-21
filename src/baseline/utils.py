import numpy as np


def split_fold(inputs_images: np.ndarray, n_split: int) -> list:
    """
    :param inputs_images:
    :param n_split:
    :return:
    """
    dim = inputs_images.shape[0]
    index_tab = np.arange(dim)
    folds = np.array_split(index_tab, n_split)
    return folds


def split_batch(inputs_images: np.ndarray, batch_size: int) -> list:
    """
    Split data into batches
    :param inputs_images:
    :param batch_size:
    :return:
    """
    dim = inputs_images.shape[0]
    index_tab = np.arange(dim)
    batches = [index_tab[i:i + batch_size] for i in range(0, dim, batch_size)]
    return batches


def create_one_hot(labels):
    """
    One hot matrix
    :param labels:
    :return:
    """
    one_vs_all = np.zeros((labels.shape[0], 4))

    for i, label in enumerate(labels):
        one_vs_all[i, int(label)] = 1

    return one_vs_all
