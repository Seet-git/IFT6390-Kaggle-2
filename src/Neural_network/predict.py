import os

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import config
from src.Neural_network.training import evaluation
from src.scripts.load import load_hyperparams


def predict():
    config.WANDB_ACTIVATE = False
    hyperparameters = load_hyperparams()
    model, X_test, mean_f1_score = evaluation(hyperparameters, save_file=True)

    print(f"F1 score moyen obtenu : {mean_f1_score:.4f}")

    save_model_predictions(model, X_test, threshold=hyperparameters.infer_threshold,
                           batch_size=hyperparameters.batch_size)


def save_model_predictions(model, data_test, threshold, batch_size):
    if not os.path.exists(f"../../output/{config.ALGORITHM}"):
        os.makedirs(f"../../output/{config.ALGORITHM}")

    # Initialisation
    model.eval()
    predictions = []

    # Define dataloader
    test_tensor = torch.tensor(data_test, dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute prediction
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(config.DEVICE)
            outputs = model(inputs)
            pred = (torch.sigmoid(outputs).view(-1) > threshold).float()
            predictions.extend(pred.cpu().numpy())

    # Save model
    df_pred = pd.DataFrame(predictions, columns=['label'])
    df_pred['label'] = df_pred['label'].astype(int)
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'../../{config.PREDICTION_PATH}/{config.ALGORITHM}/{config.PREDICTION_FILENAME}.csv')
