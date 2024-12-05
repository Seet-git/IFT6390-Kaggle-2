import os
import optuna
from src.pretrained_models.training import train as train_base
from src.extract_data import load_data
import numpy as np
import config
import pytz
import urllib.parse
from datetime import datetime


def export_trial_to_csv(study, trial):
    # Convert study trials to a DataFrame
    df = study.trials_dataframe()

    os.makedirs(config.LOG_PATH, exist_ok=True)

    # Save dataframe
    df.to_csv(f'{config.LOG_PATH}/log_{config.ALGORITHM}.csv', index=False)


def objective(trial):
    """
    Fonction objectif pour Optuna.
    """
    # Chargement des données
    images_train, labels_train, _ = load_data(10)

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    transforms = trial.suggest_categorical("transform", [True, False])

    # Configuration des hyper-paramètres
    hp = {
        "algo": "EfficientNet-B0",
        "epochs": 5,
        "optimizer": "adam",
        "batch_size": 64,
        "lr": lr,
        "weight_decay": 0,
        "transform" : transforms
    }

    f1_valid = train_base(images_train, labels_train, hp)

    mean_f1 = np.mean(f1_valid)
    return mean_f1


def bayesian_optimization(n_trial: int = 10):
    montreal_timezone = pytz.timezone('America/Montreal')
    current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")
    storage_url = f"mysql+pymysql://{config.USER}:{urllib.parse.quote(config.PASSWORD)}@{config.ENDPOINT}/{config.DATABASE_NAME}"
    print(f"Connexion to {storage_url} ...")
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=f"{config.ALGORITHM} Optimizer - {current_time}",
        sampler=optuna.samplers.TPESampler(seed=1)
    )

    # Lancez l'optimisation
    study.optimize(objective, n_trials=n_trial)

    # Affichez les meilleurs résultats
    print("Meilleurs paramètres : ", study.best_params)
    print("Meilleure précision : ", study.best_value)

    # Sauvegarder l'étude pour des analyses ultérieures
    study.trials_dataframe().to_csv("optuna_results.csv")
