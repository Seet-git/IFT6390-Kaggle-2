import numpy as np

from model import MLP_Hidden1
from utils import split_fold, split_batch, create_one_hot


def infer(model, x_inputs, y_labels):
    y_pred = model.forward(x_inputs)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_labels, axis=1)
    return np.mean(y_pred_classes == y_true_classes)


def fit(model: MLP_Hidden1, x_train, y_train, x_val, y_val, hp):
    """
    Fit model
    :param model:
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param hp:
    :return:
    """
    # Loop on all epochs
    for epoch in range(hp["epochs"]):
        loss = 0
        batches = split_batch(x_train, hp["batch_size"])

        for batch in batches:
            x_batch, y_batch = x_train[batch], y_train[batch]

            y_pred = model.forward(x_batch)
            loss += model.loss(y_batch, y_pred)
            model.backward()

        acc_train = infer(model, x_train, y_train)
        acc_val = infer(model, x_val, y_val)

        # Compute results
        print(
            f"\tEpoch {epoch + 1} - Loss {loss / hp['batch_size']} - Accuracy (train) {acc_train} - Accuracy (val) {acc_val}")
    print("")


def k_cross_validation(inputs_images: np.ndarray, one_hot: np.ndarray, hp: dict, n_split: int):
    """
    :param inputs_images:
    :param one_hot:
    :param hp: dictionary hyperparameters
    :param n_split:
    :return:
    """
    inputs_images = inputs_images.reshape(inputs_images.shape[0], -1)
    acc_scores = []
    model = None
    kf_index = split_fold(inputs_images, n_split)

    i = 0
    print(f"Hyper-paramètres: {hp}\n")
    for fold in kf_index:
        print(f"Fold {i + 1} / {n_split}")

        # Set inputs and labels
        inputs_train, inputs_val = np.delete(inputs_images, fold, axis=0), inputs_images[fold]
        labels_train, labels_val = np.delete(one_hot, fold, axis=0), one_hot[fold]

        # Set model
        model = MLP_Hidden1(input_size=hp['input_size'], hidden_layer=hp['hidden_size1'], eta=hp['eta'])

        fit(model=model, x_train=inputs_train, y_train=labels_train, x_val=inputs_val, y_val=labels_val, hp=hp)

        acc = infer(model=model, x_inputs=inputs_val, y_labels=labels_val)
        acc_scores.append(acc)

        print(f"\tAccuracy (val) {acc}\n")

        i += 1
    return np.mean(acc_scores), model


def train(inputs_images: np.ndarray, labels_images: np.ndarray, hp: dict):
    """

    :param hp:
    :param inputs_images:
    :param labels_images:
    :return:
    """
    np.random.seed(4)
    one_hot = create_one_hot(labels_images)

    _, model = k_cross_validation(inputs_images, one_hot, hp, 5)

    return model
