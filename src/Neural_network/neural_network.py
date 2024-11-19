import os

import pandas as pd
import numpy as np
import config
from src.Neural_network.bayesian_optimization import bayesian_optimization
from src.Neural_network.predict import predict


def main():
    if config.ALGORITHM not in ["Perceptron", "MLP_H1", "MLP_H2"]:
        raise ValueError("Bad ALGORITHM value")

    print(f"device: {config.DEVICE}")
    config.LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs/")

    # Load data
    config.INPUTS_DOCUMENTS = np.load(f"../../{config.DATA_PATH}data_train.npy", allow_pickle=True)
    config.LABELS_DOCUMENTS = pd.read_csv(f"../../{config.DATA_PATH}label_train.csv").to_numpy()[:, 1]
    config.TEST_DOCUMENTS = np.load(f"../../{config.DATA_PATH}data_test.npy", allow_pickle=True)
    config.VOCAB = np.load(f"../../{config.DATA_PATH}vocab_map.npy", allow_pickle=True)

    # Bayesian optimization
    bayesian_optimization(n_trials=config.N_TRIALS)

    # Predict
    predict()


if __name__ == '__main__':
    main()
