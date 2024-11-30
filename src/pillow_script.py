# Source: Chat-GPT

import os
import numpy as np
from PIL import Image

data_test = np.load('../data/test_data.pkl', allow_pickle=True)

# Extract data
images = np.array(data_test['images'])

# Normaliser entre 0 et 255 si nécessaire
if images.max() <= 1.0:  # Si les valeurs sont entre 0 et 1
    images = images * 255
images = images.astype(np.uint8)  # Convertir en entier 8 bits

# Dossier pour sauvegarder les images_test
output_folder = "../images_test"
os.makedirs(output_folder, exist_ok=True)

# Sauvegarder les 10 premières images_test
for i in range(1000):
    image_array = images[i]  # Extraire l'image du tableau Numpy
    image = Image.fromarray(image_array, mode="L")  # Assurer que c'est en niveaux de gris
    image_path = os.path.join(output_folder, f"image_{i+1}.png")
    image.save(image_path)
    print(f"Image {i+1} sauvegardée : {image_path}")