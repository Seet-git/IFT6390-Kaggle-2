import os
import numpy as np
import seaborn as sns
import pandas as pd
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, \
    average_precision_score, classification_report
import matplotlib.pyplot as plt

from src.pretrained_models.utils import denormalize


def encode_one_hot(y):
    y_np = np.array(y)
    one_hot = np.zeros((y_np.size, 4))
    one_hot[np.arange(y_np.size), y_np] = 1
    return one_hot


def plot_grad_cam(images, idx, y_true, y_pred, efficientnet_cam, resnet_cam):
    if not os.path.exists(f"./plots/{config.ALGORITHM}/Cam_grad"):
        os.makedirs(f"./plots/{config.ALGORITHM}/Cam_grad")
    # Dénormalize
    original_image = denormalize(images[idx])
    original_image = (original_image - original_image.min()) / (
            original_image.max() - original_image.min())

    # EfficientNet CAM
    eff_targets = [ClassifierOutputTarget(y_pred[idx].item())]
    eff_grayscale_cam = efficientnet_cam(input_tensor=images[idx].unsqueeze(0), targets=eff_targets)[
        0]
    eff_visualization = show_cam_on_image(original_image, eff_grayscale_cam, use_rgb=True)

    # ResNet CAM
    res_targets = [ClassifierOutputTarget(y_pred[idx].item())]
    res_grayscale_cam = resnet_cam(input_tensor=images[idx].unsqueeze(0), targets=res_targets)[0]
    res_visualization = show_cam_on_image(original_image, res_grayscale_cam, use_rgb=True)

    # Plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Image Originale (Classe {y_true[idx].item()})", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(eff_visualization)
    plt.title(f"EfficientNet Grad-CAM (Prediction {y_pred[idx].item()})", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(res_visualization)
    plt.title(f"ResNet Grad-CAM (Prediction {y_pred[idx].item()})", fontsize=16)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"./plots/{config.ALGORITHM}/Cam_grad/cam_grad_{idx}.svg", format="svg",
                bbox_inches='tight')
    plt.show()


def plot_classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["CN", "DME", "Drusen", "Healthy"], output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    classes = ["CN", "DME", "Drusen", "Healthy"]
    data = {metric: [report[cls][metric] for cls in classes] for metric in metrics}

    df = pd.DataFrame(data, index=classes)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(classes))

    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, df[metric], bar_width, label=metric.capitalize())

    plt.ylabel('Score')
    plt.xticks(index + bar_width, classes)
    plt.legend(title='Metrics', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(f"./plots/{config.ALGORITHM}/classification_metrics.svg", format="svg",
                bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    class_names = ["CN", "DME", "Drusen", "Healthy"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(f"./plots/{config.ALGORITHM}/confusion_matrix.png", format="png",
                bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    y_true_one_hot = encode_one_hot(y_true)
    y_pred_one_hot = encode_one_hot(y_pred)

    class_names = ["CN", "DME", "Drusen", "Healthy"]

    for i in range(4):
        precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
        average_precision = average_precision_score(y_true_one_hot[:, i], y_pred_one_hot[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP = {average_precision:.2f})")

    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbe Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(f"./plots/{config.ALGORITHM}/precision_recall_curve.svg", format="svg",
                bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_pred):
    # Courbes ROC
    plt.figure(figsize=(10, 6))
    y_true_one_hot = encode_one_hot(y_true)
    y_pred_one_hot = encode_one_hot(y_pred)

    class_names = ["CN", "DME", "Drusen", "Healthy"]

    for i in range(4):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrai positif')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"./plots/{config.ALGORITHM}/roc_curve.svg", format="svg",
                bbox_inches='tight')
    plt.show()


def plot_visualisations(y_true, y_pred):
    if not os.path.exists(f"./plots/{config.ALGORITHM}"):
        os.makedirs(f"./plots/{config.ALGORITHM}")

    plot_confusion_matrix(y_true, y_pred)  # Matrix confusion
    plot_roc_curve(y_true, y_pred)  # Roc curve
    plot_precision_recall_curve(y_true, y_pred)  # Precision-Recall curve
    plot_classification_metrics(y_true, y_pred)  # Classification metrics bar plots
