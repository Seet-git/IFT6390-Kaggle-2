import numpy as np

def f1_score(y_true: list, y_pred: list):
    """
    Calculate macro F1 score
    :param y_true:
    :param y_pred:
    :return:
    """
    # Vérifier les dimensions
    assert len(y_true) == len(y_pred), f"Shape: {len(y_true)} != {len(y_pred)}"

    # Récupérer les labels uniques
    labels = np.unique(y_true)

    # Initialiser les scores F1 pour chaque classe
    f1_scores = []

    for label in labels:
        # Vrais positifs
        vrai_pos = np.sum((y_true == label) & (y_pred == label))

        # Faux positifs
        faux_pos = np.sum((y_true != label) & (y_pred == label))

        # Faux négatifs
        faux_neg = np.sum((y_true == label) & (y_pred != label))

        # Précision (precision) et rappel (recall)
        precision = vrai_pos / (vrai_pos + faux_pos) if (vrai_pos + faux_pos) > 0 else 0
        recall = vrai_pos / (vrai_pos + faux_neg) if (vrai_pos + faux_neg) > 0 else 0

        # F1 score pour la classe
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # F1 score macro : moyenne des F1 scores des classes
    f1_macro = np.mean(f1_scores)
    return f1_macro


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
