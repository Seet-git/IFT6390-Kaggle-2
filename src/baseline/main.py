import numpy as np
from optimizer import grid_search
from evaluation import predict
from training import train


def main():
    # Data loading
    data_train = np.load('../../data/train_data.pkl', allow_pickle=True)
    data_test = np.load('../../data/test_data.pkl', allow_pickle=True)

    # Extract data
    images_train = np.array(data_train['images'])
    labels_train = np.array(data_train['labels'])
    images_test = np.array(data_test['images'])

    # Grid search and return best hyperparameters in a python file
    # grid_search(inputs_images=images_train, labels_images=labels_train)

    # Load best hyper-parameters
    hp = {
        "input_size": 784,
        "hidden_size1": 64,
        "hidden_size2": 32,
        "dropout_rate": 0.2,
        "batch_size": 16,
        "eta": 0.001,
        "epochs": 5
    }

    # Train model
    model_train = train(images_train, labels_train, hp)

    # Predict on a test set
    predict(predict_images=images_test, model=model_train, hp=hp)


if __name__ == "__main__":
    main()
