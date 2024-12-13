import pickle
import matplotlib.pyplot as plt
import numpy as np

from src.extract_data import load_data

_, labels, _ = load_data(path="../")
class_counts = np.bincount(labels)
classes = np.arange(len(class_counts))
frequencies = class_counts
class_names = {
    0: 'CN',
    1: 'DME',
    2: 'Drusen',
    3: 'Healthy'
}

class_labels = [class_names[cls] for cls in classes]
plt.figure(figsize=(10, 6))
plt.grid(axis='y', zorder=0)
bars = plt.bar(class_labels, frequencies, color=['#4682B4', '#CD5C5C', '#FFA07A', '#ADD8E6'], edgecolor='black', zorder=3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, yval, ha='center', va='bottom', fontsize=12)
plt.ylabel("Nombre d'exemples", fontsize=14)
plt.title("Distribution des Classes", fontsize=16)
plt.xticks(rotation=0, ha='center', fontsize=14)
plt.tight_layout()
plt.savefig(f"../plots/class_imbalance.svg", format="svg",
            bbox_inches='tight')

plt.show()
