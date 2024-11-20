import numpy as np
from model import MLP_Hidden2
from utils import split_fold, split_batch


def fit(model: MLP_Hidden2, x_train, y_train, hp):
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

        # Compute results
        # train_loss = compute_loss(model, train_dataset)
        # val_loss = compute_loss(model, val_loader)
        # f1_train, _, _ = infer(model, train_dataset, infer_threshold)
        # f1_val, _, _, = infer(model, val_loader, infer_threshold)

        print(f"\tEpoch {epoch + 1} - Loss {loss / hp['batch_size']}")
    print("\n")


def k_cross_validation(inputs_images: np.ndarray, one_hot: np.ndarray, hp: dict, n_split: int):
    """
    :param inputs_images:
    :param one_hot:
    :param hp: dictionary hyperparameters
    :param n_split:
    :return:
    """
    f1_scores = []
    kf_index = split_fold(inputs_images, n_split)

    i = 0
    print(f"Hyper-paramètres: {hp}\n")
    for fold in kf_index:
        print(f"Fold {i + 1} / {n_split}")

        # Set inputs and labels
        inputs_train, inputs_val = np.delete(inputs_images, fold, axis=0), inputs_images[fold]
        labels_train, labels_val = np.delete(one_hot, fold, axis=0), one_hot[fold]

        # Set model
        model = MLP_Hidden2(input_size=hp['input_size'], hidden_layer1=hp['hidden_size1'],
                            hidden_layer2=hp['hidden_size2'],
                            dropout_rate=hp['dropout_rate'], batch_size=hp['batch_size'], eta=hp['eta'])

        # TODO: fit, infer
        fit(model=model, x_train=inputs_train, y_train=labels_train, hp=hp)

        i += 1
    return 0
