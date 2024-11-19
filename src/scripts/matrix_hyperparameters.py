import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config


def plot_hyperparameter_correlation_matrix(study, trial):
    df = pd.read_csv(f'{config.LOG_PATH}/log_{config.ALGORITHM}.csv')

    # Select columns between 'duration' and 'state'
    start_index = df.columns.get_loc('duration') + 1
    end_index = df.columns.get_loc('state')
    hyperparameters = df.iloc[:, start_index:end_index]

    # Pretty titles
    hyperparameters.columns = [col.replace('params', '').replace('_', ' ').title() for col in hyperparameters.columns]

    # Corr matrix
    hyperparameters = hyperparameters.select_dtypes(include=[np.number])  # Prevent string
    corr_matrix = hyperparameters.corr()

    # If value have nan
    if corr_matrix.isna().all().all():
        return

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(f"{config.ALGORITHM.replace('_', ' ')} - Matrice de corrélation des hyper-paramètres")
    plt.savefig(f"{config.LOG_PATH}/{config.ALGORITHM}_matrix_corr.svg", format='svg', bbox_inches='tight')