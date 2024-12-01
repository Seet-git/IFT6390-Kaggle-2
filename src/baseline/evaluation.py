import numpy as np
from model import MLP_Hidden1
from utils import split_batch


def export_output(output_pred: list):
    np.savetxt(
        '../../output/baseline.csv',
        np.column_stack((np.arange(1, len(output_pred) + 1), output_pred)),
        delimiter=',',
        header='ID,label',
        comments='',
        fmt='%d'
    )

    print("Prediction saved !")


def predict(predict_images: np.ndarray, model: MLP_Hidden1, hp: dict):
    """
    :param hp
    :param model: 
    :param predict_images:
    :return: 
    """
    # Initialisation
    output_pred = []
    model.eval()
    batches = split_batch(predict_images, hp["batch_size"])

    for batch in batches:
        x_batch = predict_images[batch]
        y_pred = model.forward(x_batch)
        output_pred.extend(np.argmax(y_pred, axis=1))

    export_output(output_pred)
