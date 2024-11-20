import numpy as np

from model import MLP_Hidden2, MLP_Hidden1
from utils import split_fold, split_batch, create_one_hot, accuracy


def infer(model, x_inputs, y_labels, hp):
    # Initialisation
    y_score = []
    y_true = []

    model.eval()
    batches = split_batch(x_inputs, hp["batch_size"])

    for batch in batches:
        x_batch, y_batch = x_inputs[batch], y_labels[batch]
        y_pred = model.forward(x_batch)
        y_score.extend(np.argmax(y_pred, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))

    y_score = np.array(y_score)
    y_true = np.array(y_true)
    return accuracy(y_score, y_true)


def fit(model: MLP_Hidden1, x_train, y_train, hp):
    """
    Fit model
    :param model:
    :param x_train:
    :param y_train:
    :param hp:
    :return:
    """
    # Loop on all epochs
    for epoch in range(hp["epochs"]):
        model.train()
        loss = 0
        batches = split_batch(x_train, hp["batch_size"])

        for batch in batches:
            x_batch, y_batch = x_train[batch], y_train[batch]

            y_pred = model.forward(x_batch)
            loss += model.loss(y_batch, y_pred)
            model.backward()

        acc_train = infer(model, x_train, y_train, hp)

        # Compute results
        print(f"\tEpoch {epoch + 1} - Loss {loss / hp['batch_size']} - Accuracy (train) {acc_train}")
    print("")


def k_cross_validation(inputs_images: np.ndarray, one_hot: np.ndarray, hp: dict, n_split: int):
    """
    :param inputs_images:
    :param one_hot:
    :param hp: dictionary hyperparameters
    :param n_split:
    :return:
    """
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
        # model = MLP_Hidden2(input_size=hp['input_size'], hidden_layer1=hp['hidden_size1'],
        #                     hidden_layer2=hp['hidden_size2'],
        #                     dropout_rate=hp['dropout_rate'], batch_size=hp['batch_size'], eta=hp['eta'])

        model = MLP_Hidden1(input_size=hp['input_size'], hidden_layer=hp['hidden_size1'],
                            dropout_rate=hp['dropout_rate'], batch_size=hp['batch_size'], eta=hp['eta'])

        fit(model=model, x_train=inputs_train, y_train=labels_train, hp=hp)

        acc = infer(model=model, x_inputs=inputs_val, y_labels=labels_val, hp=hp)
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

    _, model = k_cross_validation(inputs_images, one_hot, hp, 2)

    return model
