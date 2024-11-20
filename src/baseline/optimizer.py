from training import k_cross_validation
from utils import create_one_hot
import numpy as np


def grid_search(inputs_images: np.ndarray, labels_images: np.ndarray):
    np.random.seed(4)
    one_hot = create_one_hot(labels_images)

    hidden_size1 = [64, 128, 256]
    hidden_size2 = [32, 64, 128]
    dropout_rate = [0.2, 0.5, 0.8]
    batch_size = [16, 32, 64]
    eta = [0.001, 0.0001, 0.01]
    epochs = [5, 10, 20]

    grid = np.array(np.meshgrid(hidden_size1, hidden_size2, dropout_rate, batch_size, eta, epochs)).T.reshape(-1, 6)

    best_score = -np.inf
    best_hp = None

    for i in grid:
        hp = {
            "input_size": 28 * 28,  # Constant
            "hidden_size1": int(i[0]),
            "hidden_size2": int(i[1]),
            "dropout_rate": i[2],
            "batch_size": int(i[3]),
            "eta": i[4],
            "epochs": int(i[5])
        }

        f1 = k_cross_validation(inputs_images, one_hot, hp, 5)

        if f1 > best_score:
            best_score = f1
            best_hp = hp
            print(f"New best F1-score {f1}")


        print(f"Best hyperparameters {best_hp}")

    print(f"\nGRID SEARCH - Best F1-score {best_score} \n"
          f"Hyper-parameters {best_hp}")
