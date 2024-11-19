import os

import numpy as np
import matplotlib.pyplot as plt
import config
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score
import seaborn as sns

from torch.utils.data import WeightedRandomSampler

from src.Neural_network.models import *


def set_seed(seed: int):
    """
    Ensures that the experiment is reproducible
    :param seed: seed number
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_sampler(y_train):
    # Count class
    class_counts = [(y_train == 0).sum().item(), y_train.sum().item()]

    weights = 1 / torch.tensor(class_counts, dtype=torch.float, device=y_train.device)

    y_train = y_train.to(weights.device)

    sample_weights = weights[y_train.long()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_model(input_size, hp):
    if input_size <= 0:
        raise ValueError(f"Invalid input_size: {input_size}. It must be positive.")
    if config.ALGORITHM == "MLP_H2":
        return MLP_H2(input_size, hp.hidden_layer1, hp.hidden_layer2, hp.dropout_rate)
    elif config.ALGORITHM == "MLP_H1":
        return MLP_H1(input_size, hp.hidden_layer, hp.dropout_rate)
    elif config.ALGORITHM == "Perceptron":
        return Perceptron(input_size)
    else:
        raise ValueError("Bad ALGORITHM value")


def plot_all_visualizations(y_true_list, y_scores_list, y_pred_list):
    if not os.path.exists(f"../../plots/{config.ALGORITHM}"):
        os.makedirs(f"../../plots/{config.ALGORITHM}")

    # Courbes ROC
    plt.figure(figsize=(10, 6))
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrai positif')
    plt.title('ROC Curve - Tous les folds')
    plt.legend(loc="lower right")
    plt.savefig(f"../../plots/{config.ALGORITHM}/{config.PREDICTION_FILENAME}_roc_curve.svg", format="svg")
    plt.show()

    # Courbes de Précision-Rappel
    plt.figure(figsize=(10, 6))
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, lw=2, label=f'Fold {fold + 1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Tous les folds')
    plt.legend(loc="lower left")
    plt.savefig(f"../../plots/{config.ALGORITHM}/{config.PREDICTION_FILENAME}_precision_recall_curve.svg", format="svg")
    plt.show()

    # 3. Matrice de Confusion
    cm = confusion_matrix(y_true_list[-1], y_pred_list[-1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Dernier Fold)')
    plt.savefig(f"../../plots/{config.ALGORITHM}/{config.PREDICTION_FILENAME}_confusion_matrix.svg", format="svg")
    plt.show()

    # 4. Bar Plot des Métriques
    f1 = f1_score(y_true_list[-1], y_pred_list[-1], average="macro")
    precision = precision_score(y_true_list[-1], y_pred_list[-1], average="macro", zero_division=0)
    recall = recall_score(y_true_list[-1], y_pred_list[-1], average="macro", zero_division=0)
    accuracy = accuracy_score(y_true_list[-1], y_pred_list[-1])
    metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Metriques')
    plt.ylabel('Scores')
    plt.title('Métriques de performances (Dernier Fold)')
    plt.savefig(f"../../plots/{config.ALGORITHM}/{config.PREDICTION_FILENAME}_performance_metrics.svg", format="svg")
    plt.ylim(0, 1)

    plt.show()
